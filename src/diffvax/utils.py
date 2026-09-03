"""Utility functions for DiffVax."""

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import random
import json
import os
from pathlib import Path
from typing import Optional
from transformers import set_seed
from huggingface_hub import snapshot_download
import shutil

totensor = T.ToTensor()
topil = T.ToPILImage()


def recover_image(image, init_image, mask, background=False):
    """Compose image with mask: either mask region from image or from init_image."""
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)


def prepare_mask_and_masked_image(image, mask):
    """Prepare image and mask tensors for inpainting."""
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image, image


def prepare_image_return_3d(image):
    """Prepare single image for model input."""
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image


def resolve_device() -> torch.device:
    """Best available compute device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device) -> torch.dtype:
    """Preferred compute dtype for a given device.

    fp16 has full kernel coverage on CUDA and is what the GradScaler path
    expects. MPS has historically incomplete/unreliable fp16 kernel coverage
    (attention, some reductions and FFT ops) — bf16 is the supported
    reduced-precision type on Apple Silicon and, sharing fp32's exponent
    range, needs no loss scaling. CPU always uses fp32.
    """
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def make_generator(
    device: torch.device, seed: Optional[int] = None
) -> torch.Generator:
    """Construct a torch.Generator appropriate for the given device.

    MPS is special-cased to a CPU generator: diffusers documents that
    torch.Generator(device="mps") does not reproduce seeded results
    consistently (a long-standing PyTorch/MPS limitation), and recommends
    seeding on CPU even when the pipeline itself runs on MPS. CUDA and CPU
    use their own device's generator as normal.
    """
    gen_device = "cpu" if device.type == "mps" else device
    generator = torch.Generator(device=gen_device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def load_perturbation_net(
    checkpoint: str,
    num_classes: int = 3,
    nb_filter: Optional[list] = None,
    device: Optional[torch.device] = None,
):
    """Load a trained NestedUNet perturbation network for inference/eval.

    ``checkpoint`` may be:
      - a local .pth file (raw state_dict, as saved by the training loop's
        torch.save() checkpoints) — nb_filter must be passed explicitly if
        the checkpoint used a non-default architecture (e.g. the H6 larger
        variant), since a raw state_dict carries no architecture metadata.
      - a local directory produced by NestedUNet.save_pretrained()
      - a Hugging Face Hub repo id (e.g. "username/diffvax-run") — recovers
        nb_filter automatically from the repo's config.json. Private repos
        are picked up via the HF_TOKEN env var / cached `hf auth login`,
        same as any gated diffusers pipeline download.

    Returned network is frozen (requires_grad=False) and in eval() mode.
    """
    from diffvax.model import NestedUNet

    device = device or resolve_device()
    if os.path.isfile(checkpoint):
        net = NestedUNet(num_classes=num_classes, nb_filter=nb_filter).to(device).eval()
        net.load_state_dict(
            torch.load(checkpoint, weights_only=True, map_location=device)
        )
        print(f"Loaded perturbation net from local checkpoint: {checkpoint}")
    else:
        # Not a local file (also transparently handles a local
        # save_pretrained() directory) -> treat as a Hub repo id.
        net = NestedUNet.from_pretrained(checkpoint).to(device).eval()
        print(f"Loaded perturbation net from Hugging Face Hub: {checkpoint}")

    for param in net.parameters():
        param.requires_grad = False
    return net


def immunize_image_pil(
    perturbation_net,
    image_pil: Image.Image,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    mask_pil: Optional[Image.Image] = None,
) -> Image.Image:
    """Apply a trained perturbation network to a PIL image.

    Full-image by default (no mask gating) — the perturbation network itself
    never sees a mask either way. Pass ``mask_pil`` to confine the applied
    perturbation to the subject region instead, mirroring the training
    loop's ``perturbation_mask_gating`` (diffvax_immunization.py): dataset
    mask convention is 1=background, 0=subject, so the perturbation is kept
    where mask==0 and zeroed where mask==1. Only pass this for checkpoints
    actually trained with that flag set — for full-image-trained checkpoints
    it would zero out perturbation content the network relied on.
    Mirrors the training loop's own float32-in/compute-dtype-out pattern:
    the NestedUNet is always fp32 regardless of the active surrogate's
    compute dtype, so the input is cast to fp32 for the forward pass and the
    output cast back.
    """
    device = device or resolve_device()
    dtype = dtype or resolve_dtype(device)

    image_np = np.array(image_pil.convert("RGB"))
    image_t = torch.from_numpy(image_np[None].transpose(0, 3, 1, 2))
    image_t = (image_t.to(dtype=torch.float32) / 127.5 - 1.0).to(device=device, dtype=dtype)

    with torch.no_grad():
        unet_out = perturbation_net(image_t.float()).to(dtype)

    if mask_pil is not None:
        h, w = image_t.shape[-2:]
        if mask_pil.size != (w, h):
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        mask_np = np.array(mask_pil.convert("L"))
        mask_t = torch.from_numpy((mask_np >= 128).astype(np.float32))[None, None]
        mask_t = mask_t.to(device=device, dtype=dtype)
        unet_out = unet_out * (1.0 - mask_t)

    img_adv = torch.clamp(image_t + unet_out, -1, 1)

    return topil(((img_adv / 2 + 0.5).clamp(0, 1)[0]).to(torch.float32).cpu())


def empty_cache(device: Optional[torch.device] = None) -> None:
    """Free cached allocator memory for whichever accelerator backend is active.

    A no-op on CPU. Centralizes the cuda/mps split so call sites don't need
    their own per-backend branching.
    """
    if device is None:
        device = resolve_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def set_seed_lib(seed):
    """Set random seed for reproducibility across CPU, CUDA, and MPS."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    random.seed(seed)
    set_seed(seed)


def load_image(image_name, data_dir, is_mask=False, images_subdir="images", masks_subdir="masks", size=(512, 512), mask_type=None):
    """Load image or mask from data directory."""
    data_path = Path(data_dir)
    if is_mask:
        if mask_type:
            mask_filename = f"{mask_type}_{image_name}.png"
        else:
            mask_filename = f"mask_{image_name}.png"
        image = (
            Image.open(data_path / masks_subdir / mask_filename)
            .convert("RGB")
            .resize(size)
        )
    else:
        image = (
            Image.open(data_path / images_subdir / f"{image_name}.png")
            .convert("RGB")
            .resize(size)
        )
    return image


def load_image_from_path(image_path, size=(512, 512)):
    """Load image from file path."""
    image = Image.open(image_path).convert("RGB").resize(size)
    return image


def save_image(img, img_path):
    """Save image to file."""
    img.save(img_path, "PNG")


def get_train_val_image_prompt_list(data_dir):
    """Load train/val image-prompt pairs."""
    base = Path(data_dir)

    if not base.exists():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to download datasets from the Hub. "
                "Install with: pip install huggingface_hub"
            ) from e

        local_root = snapshot_download(repo_id="ozdentarikcan/DiffVaxDataset", repo_type="dataset")
        base = Path(local_root)

    train_meta = base / "train" / "metadata.jsonl"
    val_meta = base / "validation" / "metadata.jsonl"
    if train_meta.exists() and val_meta.exists():

        def read_meta(meta_path: Path):
            out = []
            with meta_path.open("r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)

                    img_filename = Path(row["file_name"]).name

                    entry = {
                        "image": img_filename,
                        "prompts": row["prompts"],
                        "flux_prompts": row.get("flux_prompts", row["prompts"]),
                    }
                    if "mask_types_available" in row:
                        entry["mask_types_available"] = row["mask_types_available"]
                    out.append(entry)
            return out

        return read_meta(train_meta), read_meta(val_meta)

    raise FileNotFoundError(
        f"Could not find metadata files:\n"
        f"  {train_meta} and {val_meta}\n"
        f"Given data_dir: {data_dir}"
    )

def ensure_dataset_in_data_dir(
    repo_id: str,
    data_dir: str = "data",
):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    marker = data_dir / ".hf_ready"
    if marker.exists():
        return data_dir

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
    )
    snapshot_path = Path(snapshot_path)

    # Copy contents of snapshot into data/
    for item in snapshot_path.iterdir():
        target = data_dir / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    marker.touch()
    return data_dir
