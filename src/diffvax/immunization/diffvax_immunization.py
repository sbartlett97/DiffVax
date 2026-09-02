"""DiffVax immunization against diffusion attacks — v2 unified training loop.

Integrates all seven phases of the DiffVax v2 implementation plan:
  Phase 1: EoT augmentation (src/diffvax/eot.py)
  Phase 2: CLIP-based disruption loss (src/diffvax/losses/clip_loss.py)
  Phase 3: SD3/FLUX 16-ch VAE surrogate (src/diffvax/sd3_attack.py)
  Phase 4: Multi-resolution curriculum (src/diffvax/curriculum.py)
  Phase 5: Adaptive ensemble weighting (src/diffvax/attack_manager.py)
  Phase 6: Flat-minima regularization (src/diffvax/losses/flat_minima.py)
  Phase 7: Cross-attention disruption loss (src/diffvax/losses/attention_loss.py)

All new features are gated by config flags. Setting all enabled flags to False
reproduces the original v1 behaviour exactly (loss1 + loss2 only).
"""

import random
import traceback

import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

from diffvax.model import NestedUNet
from diffvax.reporter import TrainingReporter
from diffvax.distributed import (
    all_reduce_mean, any_rank_true, get_local_rank, get_rank, get_world_size,
    is_distributed, is_main_process,
)
from diffvax.utils import (
    set_seed_lib, load_image, resolve_device, resolve_dtype, empty_cache,
    make_generator,
)

# GradScaler is a passthrough (scale/unscale_/step are no-ops around the
# optimizer) whenever it's constructed with enabled=False. It is CUDA-only
# and only needed for fp16 (loss scaling prevents fp16 underflow); bf16
# (MPS) and fp32 (CPU) share fp32's exponent range and need no scaling, so
# it is disabled there deliberately, not merely because CUDA is absent.
try:
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
except (AttributeError, TypeError):  # torch < 2.3
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


class ImmunizationDataset(torch.utils.data.Dataset):
    """Dataset for immunization training — streams images from disk on demand.

    Supports dynamic resolution updates via set_resolution() for Phase 4
    multi-resolution curriculum training. When set_resolution() is called
    between epochs, the next __getitem__ call uses the updated size.
    """

    def __init__(self, entries, data_dir, images_subdir, masks_subdir, size,
                 dtype=torch.float16):
        self.entries = entries          # list of {"image_name", "prompt", "flux_prompt"}
        self.data_dir = data_dir
        self.images_subdir = images_subdir
        self.masks_subdir = masks_subdir
        self._current_size = size       # (H, W) tuple, updated by set_resolution()
        self._dtype = dtype             # fp16 on GPU, fp32 for CPU debugging

    @property
    def size(self):
        return self._current_size

    def set_resolution(self, resolution: int) -> None:
        """Update the target load resolution for subsequent __getitem__ calls.

        Args:
            resolution: Square resolution in pixels (must be multiple of 16).
        """
        self._current_size = (resolution, resolution)

    def __getitem__(self, index):
        entry = self.entries[index]
        mask_type = None
        if "mask_types_available" in entry:
            mask_type = random.choice(entry["mask_types_available"])
        image = load_image(
            entry["image_name"], self.data_dir, is_mask=False,
            images_subdir=self.images_subdir, masks_subdir=self.masks_subdir,
            size=self._current_size,
        )
        image_mask = load_image(
            entry["image_name"], self.data_dir, is_mask=True,
            images_subdir=self.images_subdir, masks_subdir=self.masks_subdir,
            size=self._current_size, mask_type=mask_type,
        )
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)) / 127.5 - 1.0
        mask_np = np.array(image_mask.convert("L"), dtype=np.uint8)
        mask_t = torch.from_numpy((mask_np >= 128).astype(np.float32)[None])
        return (
            img_t.to(self._dtype),
            mask_t.to(self._dtype),
            entry["prompt"],
            entry["flux_prompt"],
        )

    def __len__(self):
        return len(self.entries)


def _select_loss1_weight(attack_model, used_mask_this_batch, mask_batch, ones):
    """loss1 should be weighted toward the mask region whenever a mask was
    actually used to produce img_out this batch — either because the
    surrogate is mask-ONLY (is_inpainting, e.g. SD 1.5) or because this
    particular call opted into the masked/RePaint path (e.g. SD3.5). In both
    cases img_out's unmasked region is a verbatim copy of img_adv, so an
    unweighted loss1 would waste gradient budget "disrupting" pixels the
    surrogate never touched.
    """
    if attack_model.loss_uses_mask_weighting or used_mask_this_batch:
        return mask_batch
    return ones


def _weighted_l1(diff: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean absolute value of `diff` (B, C, H, W), weighted by `weight`
    (B, 1, H, W broadcasting over channels).

    `weight.sum()` counts spatial positions only (weight has 1 channel), but
    `diff` has C channels — dividing by `weight.sum()` alone (as this used to)
    summed the numerator over C times as many elements as the denominator
    counted, inflating the result by exactly C (verified against the observed
    loss1=3.000 plateau with a noise target: true mean abs diff ~=1.0, x3 bug
    -> 3.000). Scale the normalizer by the channel count to fix.
    """
    channels = diff.shape[1]
    return (diff * weight).norm(p=1) / (weight.sum() * channels)


class DiffVaxImmunization:
    def __init__(
        self,
        attack_model=None,
        config=None,
        load_existing=False,
        existing_iter_num=0,
        load_path=None,
        output_dir=None,
        attack_manager=None,
    ):
        self.model_name = "DiffVaxImmunization"
        self.step_size = 1
        self.eps = 32 / 255
        self.clamp_min = -1
        self.clamp_max = 1
        self.output_dir = output_dir or "outputs"
        self._config = config or {}
        self.reporter = TrainingReporter(self._config, self.output_dir)

        # Support both single attack_model (backward compat) and attack_manager
        if attack_manager is not None:
            self.attack_manager = attack_manager
            self.attack_model = None
        elif attack_model is not None:
            from diffvax.attack_manager import AttackModelManager
            self.attack_manager = AttackModelManager(
                models={"sd": attack_model},
                probabilities={"sd": 1.0},
            )
            self.attack_model = attack_model
        else:
            raise ValueError("Either attack_model or attack_manager must be provided")

        # ---- Distributed (multi-GPU) topology ----
        # Each rank owns ONE frozen surrogate for the whole run and a replica
        # of the NestedUNet under DDP; only the ~9M-param NestedUNet gradients
        # are all-reduced. See src/diffvax/distributed.py for the rationale.
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_distributed = is_distributed()

        # Device-agnostic: CUDA > MPS (Apple Silicon) > CPU (tests/debugging).
        # Under distributed CUDA each rank must bind to its OWN GPU, otherwise
        # every rank would pile onto cuda:0.
        self.device = resolve_device()
        if self.is_distributed and self.device.type == "cuda":
            self.device = torch.device(f"cuda:{get_local_rank()}")
            torch.cuda.set_device(self.device)

        # Non-main ranks must not contend for the same log file or fire
        # duplicate webhooks; give them a rank-suffixed log and no webhook.
        if not is_main_process():
            self.reporter.webhook_url = None
            _base, _ext = os.path.splitext(self.reporter.log_path)
            self.reporter.log_path = f"{_base}_rank{self.rank}{_ext}"

        # H6: configurable filter counts — default [32,64,128,256,512] (~1.8M params);
        # set nb_filter: [64,128,256,512,1024] in config for the larger ~7M variant.
        _nb_filter = config.get("nb_filter") or None
        unetmodel = NestedUNet(num_classes=3, nb_filter=_nb_filter)
        unetmodel = unetmodel.to(self.device)

        self.load_existing = load_existing
        self.existing_iter_num = existing_iter_num

        # Load weights BEFORE wrapping in DDP: the checkpoint has no "module."
        # prefix, and DDP broadcasts rank 0's parameters at construction so
        # every rank starts from identical weights either way.
        if self.load_existing:
            if not load_path:
                raise ValueError("load_existing=True but no load_path was provided")
            unetmodel.load_state_dict(torch.load(load_path, weights_only=True))

        for param in unetmodel.parameters():
            param.requires_grad = True

        # _unet_module is always the raw NestedUNet — used for state_dict()
        # (no "module." prefix), push_to_hub(), and flat-minima gradient
        # surgery. self.unetmodel is what the training loop CALLS, so it must
        # be the DDP wrapper when distributed or the gradient hooks never fire.
        self._unet_module = unetmodel
        if self.is_distributed:
            from torch.nn.parallel import DistributedDataParallel

            self.unetmodel = DistributedDataParallel(
                unetmodel,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
            )
        else:
            self.unetmodel = unetmodel

        learning_rate = config["learning_rate"]
        self.optimizer = torch.optim.Adam(
            self._unet_module.parameters(), lr=learning_rate
        )
        self.model = self._unet_module

        self.generator = make_generator(self.device)

        # ---- Phase 1: EoT augmentation ----
        self._eot = None
        if self._config.get("eot", {}).get("enabled", False):
            from diffvax.eot import DifferentiableEoT
            self._eot = DifferentiableEoT(self._config)

        # ---- Phase 2+: LossComposer (CLIP, spectral, future terms) ----
        # Instantiate if any optional loss term is enabled.
        self._loss_composer = None
        _any_extra_loss = (
            self._config.get("clip_loss", {}).get("enabled", False)
            or self._config.get("spectral_loss", {}).get("enabled", False)
        )
        if _any_extra_loss:
            from diffvax.losses import LossComposer
            self._loss_composer = LossComposer(self._config)

        # ---- Phase 4: Resolution curriculum ----
        from diffvax.curriculum import ResolutionCurriculum
        self._curriculum = ResolutionCurriculum(self._config)

        # ---- Phase 6: Flat-minima regularization ----
        self._flat_minima = None
        if self._config.get("flat_minima", {}).get("enabled", False):
            from diffvax.losses.flat_minima import FlatMinimaRegularizer
            self._flat_minima = FlatMinimaRegularizer(self._config)

        # ---- Phase 7: Cross-attention disruption loss ----
        self._attention_loss = None
        if self._config.get("attention_loss", {}).get("enabled", False):
            from diffvax.losses.attention_loss import AttentionDisruptionLoss
            self._attention_loss = AttentionDisruptionLoss(self._config)

        # H7: fixed target for loss1 — cached per output shape so the optimizer has a
        # consistent direction. A per-batch RANDOM target has E[|x-t|]=1 for all x,
        # making loss1 a constant in expectation and its gradient pure noise.
        #
        # Two target sources, both cached identically by output shape:
        #   - noise_target.image_path set: a fixed real image (e.g. an unrelated
        #     photo). Chosen over a random ±1 noise pattern because comparing a
        #     smooth, spatially-correlated generated image against independent
        #     per-pixel random noise via mean L1 distance saturates near 1.0 for
        #     virtually ANY image content (law of large numbers over ~10^5-10^6
        #     pixels) — that objective can't distinguish a disrupted output from
        #     an undisrupted one. A real photo has genuine spatial structure a
        #     generated image can actually be pushed toward or away from.
        #   - noise_target.image_path unset (default): the original random ±1
        #     noise pattern, preserved for backward compatibility.
        self._fixed_noise_target: dict = {}  # shape -> tensor
        self._target_image_source = None  # PIL.Image, loaded once if configured
        _target_image_path = self._config.get("noise_target", {}).get("image_path")
        if _target_image_path:
            self._target_image_source = Image.open(_target_image_path).convert("RGB")

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    def immunize_img(self, img, img_mask, epsilon=32):
        """Apply immunization perturbation to image."""
        img_f = img.float().to(self.device)
        # Raw module, not the DDP wrapper: this is an inference helper, and
        # DDP's forward would try to synchronise gradients that nobody wants.
        unet_out = self._unet_module(img_f)
        unet_out = unet_out.to(dtype=img.dtype)
        img_adv = torch.clamp(img + unet_out, self.clamp_min, self.clamp_max)
        return img_adv, unet_out

    def _model_card_kwargs(self, checkpoint_type: str, epoch: int, loss_value: float) -> dict:
        """Build the dynamic fields for NestedUNet's Hub model card template
        (see model.py::_MODEL_CARD_TEMPLATE) from this run's actual config,
        so a checkpoint downloaded from the Hub documents what it was
        actually trained against instead of generic Mixin boilerplate.
        """
        cfg = self._config

        surrogate_parts = []
        if cfg.get("sd_probability", 0) > 0:
            surrogate_parts.append(
                f"SD 1.5 inpainting ({cfg.get('attack_model_link')}, "
                f"p={cfg.get('sd_probability')})"
            )
        if cfg.get("sd3_probability", 0) > 0 and cfg.get("sd3_model_link"):
            surrogate_parts.append(
                f"SD3/3.5 ({cfg.get('sd3_model_link')}, p={cfg.get('sd3_probability')})"
            )
        if cfg.get("flux_probability", 0) > 0 and cfg.get("flux_model_link"):
            surrogate_parts.append(
                f"FLUX ({cfg.get('flux_model_link')}, p={cfg.get('flux_probability')})"
            )
        surrogates = "; ".join(surrogate_parts) or "none configured"

        curriculum_cfg = cfg.get("curriculum", {})
        if curriculum_cfg.get("enabled", False):
            resolution_info = " → ".join(
                f"{s['resolution']}px (until epoch {s['until_epoch']})"
                for s in curriculum_cfg.get("stages", [])
            )
        else:
            resolution_info = f"{cfg.get('resolution', 512)}px (static)"

        loss_term_labels = []
        for key, label in [
            ("eot", "EoT augmentation"),
            ("clip_loss", "CLIP disruption"),
            ("spectral_loss", "Spectral concentration"),
            ("latent_loss", "Latent-space disruption"),
            ("attention_loss", "Attention disruption"),
            ("flat_minima", "Flat-minima regularization"),
            ("adaptive_ensemble", "Adaptive ensemble weighting"),
            ("noise_target", "Fixed target for loss1"),
        ]:
            if cfg.get(key, {}).get("enabled", False):
                loss_term_labels.append(label)
        masked_prob = cfg.get("sd3_attack", {}).get("masked_attack_probability", 0.0)
        if masked_prob > 0:
            loss_term_labels.append(f"masked/inpainting-style RePaint attack (p={masked_prob})")
        loss_terms = ", ".join(loss_term_labels) or "none (v1 baseline: loss1 + loss2 only)"

        hyperparams = (
            f"alpha={cfg.get('alpha')}, beta={cfg.get('beta')}, "
            f"learning_rate={cfg.get('learning_rate')}, "
            f"num_inference_steps={cfg.get('num_inference_steps')}"
        )

        return {
            "model_name": f"DiffVax NestedUNet ({cfg.get('project_name', 'diffvax')})",
            "surrogates": surrogates,
            "resolution_info": resolution_info,
            "loss_terms": loss_terms,
            "hyperparams": hyperparams,
            "checkpoint_type": checkpoint_type,
            "epoch": epoch,
            "loss_value": f"{loss_value:.5f}",
        }

    def _load_target_image_tensor(self, shape, dtype, device):
        """Resize the fixed H7 target image to match img_out's spatial shape
        and normalize to [-1, 1] (same convention as utils.load_image), for
        caching by shape identically to the random-noise target path."""
        b, _, h, w = shape
        img = self._target_image_source.resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        t = torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)
        t = t.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
        return t.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_immunization_all_images_batch(
        self,
        entries,
        data_dir,
        images_subdir,
        masks_subdir,
        size,
        target_image=None,
        iter_num=2000,
        SEED=5,
        batch_size=2,
        alpha=1,
        loss_type="l2",
        sd_target_resolutions=None,
        strength_range=None,
    ):
        if sd_target_resolutions is None:
            sd_target_resolutions = [512]
        if strength_range is None:
            strength_range = [0.5, 1.0]
        # Offset the seed per rank so EoT augmentation, denoising strength, and
        # mask-variant draws differ across ranks — identical streams would make
        # the ensemble gradient less diverse for no benefit. This does NOT
        # perturb data partitioning (DistributedSampler carries its own seed)
        # nor the H7 fixed noise target (seeded from its own generator).
        set_seed_lib(SEED + self.rank)

        # Memory-efficient SDPA backward is not implemented in all PyTorch
        # versions; disable it so the runtime prefers flash attention (which
        # supports first-order backward) with math as a fallback. This toggle
        # is CUDA-specific (selects among CUDA SDPA kernels); MPS uses its own
        # Metal attention kernel and has no equivalent backend-selection knob.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        total_iter = 0

        models_dir = os.path.join(self.output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        existing_folders = [
            d for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d)) and d.isdigit()
        ]
        last_idx = max([int(x) for x in existing_folders], default=0) + 1
        run_dir = os.path.join(models_dir, str(last_idx))
        os.makedirs(run_dir, exist_ok=True)

        if self.load_existing:
            path_of_models = os.path.join(
                models_dir,
                f"sd15_all_images_half_mult_img_mult_prompt_immunization_model_"
                f"{self.model_name}_iter_{iter_num + self.existing_iter_num}_"
                f"alpha_{alpha}_loss_{loss_type}",
            )
        else:
            path_of_models = os.path.join(
                run_dir,
                f"sd15_all_images_{self.model_name}_iter_{iter_num}_alpha_{alpha}"
                f"_loss_{loss_type}_batch_{batch_size}",
            )

        # Pull adaptive ensemble settings
        adaptive_cfg = self._config.get("adaptive_ensemble", {})
        adaptive_enabled = adaptive_cfg.get("enabled", False)
        update_period = int(adaptive_cfg.get("update_period", 50))

        # Pull flat-minima lambda (Phase 6)
        lambda_flat = float(
            self._config.get("flat_minima", {}).get("lambda_flat", 0.01)
        )

        # Pull attention loss weight (Phase 7)
        attn_weight = float(
            self._config.get("attention_loss", {}).get("weight", 0.3)
        )
        attn_only_with_dit = bool(
            self._config.get("attention_loss", {}).get("only_with_dit", True)
        )
        dit_model_names = {"sd3", "flux"}
        num_inference_steps = int(self._config.get("num_inference_steps", 4))
        # Fraction of SD3(.5) batches routed through the masked/inpainting-
        # style RePaint attack instead of full-image img2img, using the SAME
        # resident pipeline (see SD3Attack.attack()'s `mask` handling). 0.0
        # (default) preserves exact behaviour for every config that doesn't
        # set this.
        masked_attack_probability = float(
            self._config.get("sd3_attack", {}).get("masked_attack_probability", 0.0)
        )
        # Confine the perturbation itself to the subject region (dataset mask
        # convention: 1=background, 0=subject). False (default) preserves
        # exact full-image behaviour for every config that doesn't set this.
        perturbation_mask_gating = bool(
            self._config.get("perturbation_mask_gating", False)
        )

        dataset = ImmunizationDataset(
            entries, data_dir, images_subdir, masks_subdir, size,
            dtype=resolve_dtype(self.device),
        )
        dl_cfg = self._config.get("dataloader", {})
        num_workers = int(dl_cfg.get("num_workers", 4))

        # Holds the active DistributedSampler so the epoch loop can call
        # set_epoch() on it (without which every epoch reuses one shuffle).
        self._sampler = None

        def _make_dataloader(ds, bs):
            """Create a fresh DataLoader with the given dataset and batch size.

            Must be recreated on every curriculum stage change so that:
            1. Worker subprocesses are forked after set_resolution() and pick up
               the new _current_size (stale workers hold the old value).
            2. The new per-stage batch size is applied immediately.

            Under DDP a DistributedSampler partitions the dataset so each rank
            sees a disjoint slice, with drop_last=True so every rank runs the
            SAME number of batches — an uneven count would leave some ranks
            waiting forever in the gradient all-reduce for peers that already
            finished the epoch.
            """
            pf = int(dl_cfg.get("prefetch_factor", 4)) if num_workers > 0 else None
            sampler = None
            if self.is_distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    ds,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    drop_last=True,
                )
            self._sampler = sampler
            return torch.utils.data.DataLoader(
                ds, batch_size=bs,
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=pf,
                drop_last=self.is_distributed,
            )

        # Initialise with the epoch-0 curriculum resolution and batch size.
        _curr_resolution = self._curriculum.get_resolution(0)
        _curr_dl_batch = self._curriculum.get_batch_size(0)
        dataset.set_resolution(_curr_resolution)
        dataloader = _make_dataloader(dataset, _curr_dl_batch)

        # ---- Hub + reporting setup ----
        # self.reporter is initialised in __init__; reset events for this run
        # so the JSON log reflects only the current training session.
        self.reporter._events = []

        hub_cfg = self._config.get("hub", {})
        hub_enabled = hub_cfg.get("enabled", False)
        hub_repo_id = hub_cfg.get("repo_id") or None
        hub_private = bool(hub_cfg.get("private", True))
        hub_token = hub_cfg.get("token") or os.environ.get("HF_TOKEN") or None
        hub_upload_every_n = int(hub_cfg.get("upload_every_n_epochs", 10000))

        best_loss = float("inf")
        best_model_path: str = path_of_models + "_best.pth"
        epoch_avg_loss = float("inf")  # updated each epoch for use after the loop

        batch_iter_count = 0  # global batch counter for adaptive weight updates

        for epoch_i in range(iter_num):
            # ---- Phase 4: curriculum resolution + batch-size update ----
            curriculum_resolution = self._curriculum.get_resolution(epoch_i)
            curriculum_batch_size = self._curriculum.get_batch_size(epoch_i)
            if (curriculum_resolution != _curr_resolution
                    or curriculum_batch_size != _curr_dl_batch):
                dataset.set_resolution(curriculum_resolution)
                _curr_resolution = curriculum_resolution
                _curr_dl_batch = curriculum_batch_size
                # Recreate DataLoader: spawns fresh workers that inherit the
                # updated _current_size, and applies the new per-stage batch size.
                dataloader = _make_dataloader(dataset, _curr_dl_batch)
                if is_main_process():
                    tqdm.write(
                        f"[Curriculum] Stage change → {curriculum_resolution}px "
                        f"batch={curriculum_batch_size}"
                    )

            # Reshuffle each rank's slice per epoch. Without set_epoch() the
            # DistributedSampler yields the identical permutation every epoch.
            if self._sampler is not None:
                self._sampler.set_epoch(epoch_i)

            pbar = tqdm(
                enumerate(dataloader), total=len(dataloader),
                disable=not is_main_process(),
            )
            epoch_losses = []
            epoch_losses1 = []
            epoch_losses2 = []
            per_model_losses = {}  # {model_name: {"loss": [], "loss1": [], "loss2": []}}

            for i, (img_batch, mask_batch, prompt_batch, flux_prompt_batch) in enumerate(dataloader):
                self.optimizer.zero_grad()
                losses = []
                losses1 = []
                losses2 = []
                cur_iter = i + self.existing_iter_num

                # Select model for this batch
                model_name, attack_model = self.attack_manager.select_and_load()

                # Pick prompt set based on active model
                cur_prompt = flux_prompt_batch if model_name == "flux" else prompt_batch

                img_batch = img_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)

                mask_batch.requires_grad = False
                # img_batch.requires_grad_() removed — no code consumes input-image
                # gradients; enabling it wastes memory tracking the input graph.

                ones = torch.ones_like(mask_batch)
                # Real face masks can be all-background (face detection found
                # nothing, e.g. face_alexander-lunyov...png in the validation
                # set) — 1.0 - mask_batch is then exactly zero, which would
                # make loss2's weighted normalizer divide by zero (NaN, trips
                # the NaN/Inf guard below and aborts the whole run). Same
                # hazard class as the existing mask_batch.sum() > 0 guard for
                # use_masked_sd3 — fall back to whole-image gating for this
                # one batch rather than let a single bad mask kill the run.
                subject_region = 1.0 - mask_batch
                gate_to_subject = perturbation_mask_gating and subject_region.sum() > 0

                # Perturbation: full image by default. When gate_to_subject,
                # confine it to the subject region — dataset mask convention
                # is 1=background, 0=subject, so subject_region selects the
                # subject. Matches this repo's pre-"full image immunization"
                # behavior (mask-gated perturbation), restored deliberately
                # rather than left as an experiment: a perturbation the
                # attacker's model can zero out entirely (background,
                # discarded by inpainting-style masking) wastes budget that's
                # better spent concentrated on the subject, which every
                # attack path here treats as "known"/preserved content and
                # therefore actually propagates.
                img_f = img_batch.float()
                # Call the module, not .forward(): DDP's gradient-sync hooks
                # live in __call__, so calling .forward() directly would
                # silently skip the all-reduce and let ranks diverge.
                unet_out = self.unetmodel(img_f)
                unet_out = unet_out.to(dtype=img_batch.dtype)
                if gate_to_subject:
                    unet_out = unet_out * subject_region
                img_adv = torch.clamp(
                    img_batch + unet_out, self.clamp_min, self.clamp_max
                )

                # ---- Phase 1: EoT augmentation ----
                # EoT is applied to img_adv before passing to the attack model.
                # The noise loss (loss2) is still computed against the un-augmented
                # img_adv to penalize perturbation magnitude in pixel space.
                if self._eot is not None:
                    img_adv_aug = self._eot(img_adv)
                else:
                    img_adv_aug = img_adv

                # Sample strength for this batch
                strength = random.uniform(*strength_range)

                # ---- Phase 7: Register attention hooks for DiT models ----
                attn_active = (
                    self._attention_loss is not None
                    and (not attn_only_with_dit or model_name in dit_model_names)
                )
                if attn_active:
                    transformer = getattr(
                        getattr(attack_model, "pipe", None), "transformer", None
                    )
                    if transformer is not None:
                        self._attention_loss.register_hooks(transformer)

                # Differentiable resize and attack (Phases 3 + 4)
                h, w = img_batch.shape[2], img_batch.shape[3]
                actual_bs = img_batch.shape[0]

                # H-mask: per-batch coin flip routing this SD3.5 call through
                # the masked/RePaint attack instead of full-image img2img,
                # using the SAME resident pipeline. mask_batch.sum() > 0 guards
                # against a degenerate all-zero mask draw: loss1_weight_norm's
                # sum would be exactly 0 in that case, producing a NaN loss
                # that trips the NaN/Inf guard below and aborts the run — an
                # empty mask just falls through to the ordinary full-image path.
                use_masked_sd3 = (
                    not attack_model.is_inpainting
                    and getattr(attack_model, "supports_masked_attack", False)
                    and mask_batch.sum() > 0
                    and random.random() < masked_attack_probability
                )

                if attack_model.is_inpainting:
                    # Inpainting model: mask-conditioned, multi-resolution downsample.
                    # Apply mask on GPU: zero out inpaint region so the pipeline sees a
                    # masked image (inpaint region = 0) as expected by inpainting models.
                    attack_mask = mask_batch
                    sd_target = random.choice(sd_target_resolutions)
                    if sd_target < h:
                        img_adv_resized = F.interpolate(
                            img_adv_aug, (sd_target, sd_target),
                            mode="bilinear", align_corners=False,
                        )
                        mask_resized = F.interpolate(
                            attack_mask, (sd_target, sd_target), mode="nearest"
                        )
                        img_adv_sd = img_adv_resized * (1.0 - mask_resized)
                        img_out_small = attack_model.attack(
                            prompt=cur_prompt,
                            image=img_adv_sd,
                            mask=mask_resized,
                            height=sd_target,
                            width=sd_target,
                            num_inference_steps=num_inference_steps,
                            batch_size=actual_bs,
                            strength=strength,
                        )
                        img_out = F.interpolate(
                            img_out_small, (h, w), mode="bilinear", align_corners=False
                        )
                    else:
                        img_adv_sd = img_adv_aug * (1.0 - attack_mask)
                        img_out = attack_model.attack(
                            prompt=cur_prompt,
                            image=img_adv_sd,
                            mask=attack_mask,
                            height=h,
                            width=w,
                            num_inference_steps=num_inference_steps,
                            batch_size=actual_bs,
                            strength=strength,
                        )
                elif use_masked_sd3:
                    # Masked/inpainting-style attack via the SAME SD3.5
                    # pipeline instance (no second load). Unlike the
                    # is_inpainting branch above, the FULL (never pre-zeroed)
                    # image is passed through — attack() VAE-encodes it whole
                    # as the RePaint "known content" source and blends in
                    # latent space internally; the mask only needs to stay in
                    # lockstep resolution with the image.
                    native_res = attack_model.native_resolution
                    if h > native_res:
                        img_input = F.interpolate(
                            img_adv_aug, (native_res, native_res),
                            mode="bilinear", align_corners=False,
                        )
                        mask_input = F.interpolate(
                            mask_batch, (native_res, native_res), mode="nearest"
                        )
                        img_out_native = attack_model.attack(
                            prompt=cur_prompt,
                            image=img_input,
                            mask=mask_input,
                            height=native_res,
                            width=native_res,
                            num_inference_steps=num_inference_steps,
                            batch_size=actual_bs,
                            strength=strength,
                        )
                        img_out = F.interpolate(
                            img_out_native, (h, w), mode="bilinear", align_corners=False
                        )
                    else:
                        img_out = attack_model.attack(
                            prompt=cur_prompt,
                            image=img_adv_aug,
                            mask=mask_batch,
                            height=h,
                            width=w,
                            num_inference_steps=num_inference_steps,
                            batch_size=actual_bs,
                            strength=strength,
                        )
                else:
                    # FLUX, SD3, or other full-image models
                    # Phase 4: native-resolution dispatch
                    native_res = attack_model.native_resolution
                    if h > native_res:
                        # Resize down to native resolution for the attack pass
                        img_input = F.interpolate(
                            img_adv_aug, (native_res, native_res),
                            mode="bilinear", align_corners=False,
                        )
                        img_out_native = attack_model.attack(
                            prompt=cur_prompt,
                            image=img_input,
                            height=native_res,
                            width=native_res,
                            num_inference_steps=num_inference_steps,
                            batch_size=actual_bs,
                            strength=strength,
                        )
                        img_out = F.interpolate(
                            img_out_native, (h, w), mode="bilinear", align_corners=False
                        )
                    else:
                        img_out = attack_model.attack(
                            prompt=cur_prompt,
                            image=img_adv_aug,
                            height=h,
                            width=w,
                            num_inference_steps=num_inference_steps,
                            batch_size=actual_bs,
                            strength=strength,
                        )

                # ---- Loss computation ----
                resolution = h
                # H7: fixed target for loss1 (Mist insight, arXiv:2305.12683).
                # Target is cached per shape so the optimizer has a consistent
                # direction across batches — a per-batch RANDOM target has
                # E[|x-t|]=1 for all x, making loss1 a constant in expectation.
                # See __init__ for why a real image (noise_target.image_path)
                # is preferred over random ±1 noise: comparing a smooth
                # generated image against independent per-pixel noise via mean
                # L1 distance saturates near 1.0 regardless of content.
                if self._config.get("noise_target", {}).get("enabled", False):
                    shape_key = tuple(img_out.shape)
                    if shape_key not in self._fixed_noise_target:
                        if self._target_image_source is not None:
                            self._fixed_noise_target[shape_key] = (
                                self._load_target_image_tensor(
                                    img_out.shape, img_out.dtype, img_out.device
                                )
                            )
                        else:
                            # torch.randint's generator= must match the output
                            # tensor's device on CUDA; MPS generator support is
                            # unreliable (see make_generator), so always sample
                            # on CPU and move — cheap, one-time, cached by shape.
                            g = torch.Generator(device="cpu").manual_seed(1234)
                            self._fixed_noise_target[shape_key] = (
                                torch.randint(
                                    0, 2, img_out.shape,
                                    generator=g, device="cpu",
                                    dtype=img_out.dtype,
                                ) * 2 - 1
                            ).to(img_out.device)
                    target_image_t = self._fixed_noise_target[shape_key]
                else:
                    target_image_t = torch.zeros_like(img_out)

                loss1_weight = _select_loss1_weight(
                    attack_model, use_masked_sd3, mask_batch, ones
                )

                loss1_weight_norm = loss1_weight / resolution
                # When the perturbation is gated to the subject this batch,
                # the imperceptibility penalty must be measured over that
                # same region — otherwise it's diluted by background pixels
                # that are guaranteed exactly zero, making alpha silently
                # mean a much weaker per-visible-pixel budget than the config
                # states. Uses the same gate_to_subject/subject_region as the
                # perturbation itself so the two can never disagree (e.g. on
                # the degenerate all-background-mask fallback above).
                loss2_weight = subject_region if gate_to_subject else ones
                loss2_weight_norm = loss2_weight / resolution

                # Existing losses (unchanged in intent; see _weighted_l1 for
                # the channel-count fix to their normalization)
                loss1 = _weighted_l1(img_out - target_image_t, loss1_weight_norm)
                loss2 = _weighted_l1(alpha * (img_adv - img_batch), loss2_weight_norm)

                # ---- Phase 2+: Extra losses (CLIP, spectral, …) ----
                loss_clip_val = 0.0
                loss_spectral_val = 0.0
                if self._loss_composer is not None and self._loss_composer.has_terms():
                    loss_extra, extra_breakdown = self._loss_composer.compute(
                        img_orig=img_batch,
                        img_adv=img_adv,
                        img_out=img_out,
                        prompts=cur_prompt,
                    )
                    loss_clip_val = extra_breakdown.get("clip", 0.0)
                    loss_spectral_val = extra_breakdown.get("spectral", 0.0)
                else:
                    loss_extra = torch.tensor(0.0, device=self.device)

                # ---- H8: VAE latent-space disruption loss ----
                # Compute in latent space using the active attack model's VAE.
                # VAE encode-only is ~10x cheaper than a full denoising pass —
                # this can run on every batch regardless of which surrogate is active.
                # The term is the cosine SIMILARITY between clean and adversarial
                # latents: minimizing it pushes the latents apart. (An earlier
                # version added 1 - cos_sim, which rewarded identical latents.)
                loss_latent = torch.tensor(0.0, device=self.device)
                latent_loss_weight = float(
                    self._config.get("latent_loss", {}).get("weight", 1.0)
                )
                if self._config.get("latent_loss", {}).get("enabled", False):
                    _vae = attack_model.get_vae()
                    if _vae is not None:
                        from diffvax.losses.latent_loss import latent_disruption_loss
                        loss_latent = latent_disruption_loss(
                            _vae, img_batch, img_adv
                        )

                # ---- Phase 7: Attention disruption loss ----
                loss_attn = torch.tensor(0.0, device=self.device)
                if attn_active:
                    loss_attn = self._attention_loss.compute()
                # Always remove hooks after the forward pass to avoid
                # stale hooks accumulating when the next batch uses a different model.
                if self._attention_loss is not None:
                    self._attention_loss.remove_hooks()

                # Aggregate loss
                loss = (
                    loss1
                    + loss2
                    + loss_extra
                    + attn_weight * loss_attn
                    + latent_loss_weight * loss_latent
                )

                # Log scalar values (after building the full computation graph)
                loss1_val = loss1.item()
                loss2_val = loss2.item()
                loss_latent_val = loss_latent.item()
                loss_attn_val = loss_attn.item()

                losses.append(loss.item())
                losses1.append(loss1_val)
                losses2.append(loss2_val)
                epoch_losses.append(loss.item())
                epoch_losses1.append(loss1_val)
                epoch_losses2.append(loss2_val)

                if model_name not in per_model_losses:
                    per_model_losses[model_name] = {"loss": [], "loss1": [], "loss2": []}
                per_model_losses[model_name]["loss"].append(loss.item())
                per_model_losses[model_name]["loss1"].append(loss1_val)
                per_model_losses[model_name]["loss2"].append(loss2_val)

                _oom_here = False
                try:
                    scaler.scale(loss).backward()
                except RuntimeError as _bwd_exc:
                    # Covers both "CUDA out of memory" and "MPS backend out
                    # of memory" — both RuntimeError messages contain this
                    # substring.
                    _is_oom = "out of memory" in str(_bwd_exc).lower()
                    if not _is_oom:
                        raise
                    # OOM during backward: free state, skip this batch, notify.
                    _oom_here = True
                    self.optimizer.zero_grad(set_to_none=True)
                    empty_cache(self.device)
                    if self._attention_loss is not None:
                        self._attention_loss.remove_hooks()
                    _oom_msg = (
                        f"{self.device.type.upper()} OOM during backward pass "
                        f"(rank={self.rank}, epoch={epoch_i}, batch={i}): {_bwd_exc}"
                    )
                    tqdm.write(f"[OOM] {_oom_msg} — skipping batch")
                    self.reporter.report_error(
                        f"{self.device.type}_oom", _oom_msg, epoch=epoch_i, batch=i
                    )

                # An OOM on ANY rank must skip the batch on EVERY rank. A rank
                # that proceeded alone would block forever in DDP's gradient
                # all-reduce waiting for the peer that already moved on, so the
                # skip decision has to be agreed collectively before acting.
                if any_rank_true(_oom_here, self.device):
                    if not _oom_here:
                        self.optimizer.zero_grad(set_to_none=True)
                        if self._attention_loss is not None:
                            self._attention_loss.remove_hooks()
                    pbar.update(1)
                    continue

                # Unscale gradients before any code reads or modifies .grad.
                # scaler.scale(loss).backward() multiplies .grad by the scaler
                # factor (~65536); unscale_ divides them back to true magnitude.
                # Must happen before flat_minima.apply() and record_gradient, and
                # before scaler.step() (which calls unscale_ internally if not done).
                scaler.unscale_(self.optimizer)

                # ---- Phase 6: Flat-minima regularization (post-backward) ----
                # Applied after unscale_ so grad_norm reads true gradient magnitudes.
                if self._flat_minima is not None:
                    self._flat_minima.apply(self._unet_module, lambda_flat)

                # ---- Phase 5: Record gradient for adaptive weighting ----
                if adaptive_enabled and self.attack_manager.adaptive:
                    grad_vecs = [
                        p.grad.detach().flatten()
                        for p in self._unet_module.parameters()
                        if p.grad is not None
                    ]
                    if grad_vecs:
                        grad_vec = torch.cat(grad_vecs)
                        # H-mask: distinct key so masked/unmasked SD3.5
                        # gradient signals aren't EMA-blended together under
                        # one name if adaptive_ensemble is ever enabled here.
                        # record_gradient() already no-ops on an unrecognized
                        # name, so this needs no AttackModelManager changes.
                        record_model_name = (
                            f"{model_name}_masked" if use_masked_sd3 else model_name
                        )
                        self.attack_manager.record_gradient(record_model_name, grad_vec)

                # NaN/Inf check before stepping — abort and save before the
                # poisoned update is applied to the model weights. A NaN on any
                # ONE rank must abort ALL of them: returning from a single rank
                # would strand its peers in the next all-reduce.
                _nan_here = bool(torch.isnan(loss) or torch.isinf(loss))
                if any_rank_true(_nan_here, self.device):
                    if self._attention_loss is not None:
                        self._attention_loss.remove_hooks()
                    nan_path = path_of_models + f"iter_{cur_iter}_early.pth"
                    if is_main_process():
                        torch.save(self.model.state_dict(), nan_path)
                    _origin = "this rank" if _nan_here else "another rank"
                    self.reporter.report_error(
                        "nan_loss",
                        f"NaN/Inf loss detected on {_origin} at epoch={epoch_i} "
                        f"batch={i} (global iter={cur_iter}). "
                        f"Emergency checkpoint saved: {nan_path}",
                        epoch=epoch_i,
                        batch=i,
                    )
                    tqdm.write(
                        f"[NaN] Loss is NaN/Inf on {_origin} at epoch={epoch_i} "
                        f"batch={i} — aborting. Checkpoint: {nan_path}"
                    )
                    return

                scaler.step(self.optimizer)
                scaler.update()

                # S1: log scaler state periodically to surface silent step-skipping
                if batch_iter_count % 100 == 0 and is_main_process():
                    tqdm.write(
                        f"[Scaler] scale={scaler.get_scale():.0f} "
                        f"batch={batch_iter_count}"
                    )

                batch_iter_count += 1

                # ---- Phase 5: Periodically update adaptive weights ----
                if (
                    adaptive_enabled
                    and self.attack_manager.adaptive
                    and batch_iter_count % update_period == 0
                ):
                    self.attack_manager.update_weights()

                total_iter += batch_size

                pbar.set_description_str(
                    f"AVG Loss: {np.mean(losses):.5f} "
                    f"Loss1: {np.mean(losses1):.5f} "
                    f"Loss2: {np.mean(losses2):.5f}"
                    + (f" CLIP: {loss_clip_val:.5f}" if loss_clip_val else "")
                    + (f" Spec: {loss_spectral_val:.5f}" if loss_spectral_val else "")
                    + (f" Lat: {loss_latent_val:.5f}" if loss_latent_val else "")
                    + (f" Attn: {loss_attn_val:.5f}" if loss_attn_val else "")
                )
                pbar.update(1)

                losses = []
                losses1 = []
                losses2 = []

            # ---- Per-epoch summary ----
            _local_epoch_loss = (
                float(np.mean(epoch_losses)) if epoch_losses else float("inf")
            )
            # Average across ranks so every rank agrees on the epoch loss — and
            # therefore agrees on whether this is a new best. Ranks comparing
            # their own local losses would disagree about when to checkpoint.
            epoch_avg_loss = all_reduce_mean(_local_epoch_loss, self.device)
            per_model_avg = {
                mn: float(np.mean(per_model_losses[mn]["loss"]))
                for mn in per_model_losses
                if per_model_losses[mn]["loss"]
            }
            if is_main_process():
                parts = [f"Epoch {epoch_i}  avg={epoch_avg_loss:.5f}"]
                if self.is_distributed:
                    parts.append(f"(rank0 local={_local_epoch_loss:.5f})")
                for mn in sorted(per_model_losses):
                    m = per_model_losses[mn]
                    parts.append(
                        f"[{mn}] loss={np.mean(m['loss']):.4f} "
                        f"loss1={np.mean(m['loss1']):.4f} "
                        f"loss2={np.mean(m['loss2']):.4f} "
                        f"(n={len(m['loss'])})"
                    )
                tqdm.write("  ".join(parts))
            self.reporter.report_epoch(epoch_i, epoch_avg_loss, per_model_avg)

            # ---- Periodic local checkpoint (rank 0 only) ----
            if (epoch_i + 1) % self.reporter.checkpoint_every == 0:
                ckpt_path = path_of_models + f"_epoch{epoch_i}.pth"
                if is_main_process():
                    torch.save(self.model.state_dict(), ckpt_path)
                    self.reporter.report_checkpoint(
                        epoch_i, epoch_avg_loss, ckpt_path, is_best=False
                    )
                    tqdm.write(f"[Checkpoint] Saved periodic checkpoint: {ckpt_path}")

            # ---- Best-model checkpoint + Hub upload ----
            # Driven by the rank-averaged loss, so this branch is entered on
            # every rank in lockstep; only rank 0 performs the write.
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                if is_main_process():
                    torch.save(self.model.state_dict(), best_model_path)
                    self.reporter.report_checkpoint(
                        epoch_i, best_loss, best_model_path, is_best=True
                    )
                    tqdm.write(
                        f"[Checkpoint] New best model (loss={best_loss:.5f}): "
                        f"{best_model_path}"
                    )
                if is_main_process() and hub_enabled and hub_repo_id:
                    try:
                        self._unet_module.push_to_hub(
                            repo_id=hub_repo_id,
                            private=hub_private,
                            token=hub_token,
                            commit_message=(
                                f"Epoch {epoch_i}: best checkpoint (loss={best_loss:.5f})"
                            ),
                            model_card_kwargs=self._model_card_kwargs(
                                "best", epoch_i, best_loss
                            ),
                        )
                        tqdm.write(f"[Hub] Uploaded best checkpoint to {hub_repo_id}")
                    except Exception as exc:
                        tqdm.write(f"[Hub] Best-model upload failed: {exc}")

            # ---- Periodic Hub upload (rank 0 only) ----
            if (
                is_main_process()
                and hub_enabled
                and hub_repo_id
                and (epoch_i + 1) % hub_upload_every_n == 0
            ):
                try:
                    self._unet_module.push_to_hub(
                        repo_id=hub_repo_id,
                        private=hub_private,
                        token=hub_token,
                        commit_message=(
                            f"Epoch {epoch_i}: periodic checkpoint (loss={epoch_avg_loss:.5f})"
                        ),
                        model_card_kwargs=self._model_card_kwargs(
                            "periodic", epoch_i, epoch_avg_loss
                        ),
                    )
                    tqdm.write(f"[Hub] Uploaded periodic checkpoint to {hub_repo_id}")
                except Exception as exc:
                    tqdm.write(f"[Hub] Periodic upload failed: {exc}")

        # ---- Final save (rank 0 only; all ranks hold identical weights) ----
        final_path = path_of_models + "_final.pth"
        if is_main_process():
            torch.save(self.model.state_dict(), final_path)
            if hub_enabled and hub_repo_id:
                try:
                    self._unet_module.push_to_hub(
                        repo_id=hub_repo_id,
                        private=hub_private,
                        token=hub_token,
                        commit_message=f"Training complete: final model (loss={epoch_avg_loss:.5f})",
                        model_card_kwargs=self._model_card_kwargs(
                            "final", epoch_i, epoch_avg_loss
                        ),
                    )
                    tqdm.write(f"[Hub] Uploaded final model to {hub_repo_id}")
                except Exception as exc:
                    tqdm.write(f"[Hub] Final upload failed: {exc}")
        self.reporter.report_complete(iter_num, epoch_avg_loss, final_path)
        if is_main_process():
            tqdm.write(
                f"[Training complete] {iter_num} epochs | final loss={epoch_avg_loss:.5f} | "
                f"best loss={best_loss:.5f} | model: {final_path}"
            )

        return img_adv, final_path

    def edit_image(
        self,
        prompt,
        img,
        img_mask,
        num_inf=30,
        SEED=5,
        generator=None,
    ):
        """Edit image using the diffusion model."""
        strength = 1.0
        guidance_scale = 7.5
        self.generator.manual_seed(SEED)

        # Use first available attack model for editing
        if self.attack_model is not None:
            model = self.attack_model.model
        else:
            # Pick the SD model from the manager if available
            for name, m in self.attack_manager.models.items():
                if name == "sd":
                    model = m.model
                    break
            else:
                # Fallback: use first model
                _, m = next(iter(self.attack_manager.models.items()))
                model = m.model

        edited_image = model(
            prompt=prompt,
            image=img,
            mask_image=img_mask,
            eta=1,
            num_inference_steps=num_inf,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=self.generator,
        ).images

        return edited_image
