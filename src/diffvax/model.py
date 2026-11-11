# Based on: https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py

import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin

# Jinja2 template rendered by push_to_hub()/save_pretrained() into README.md.
# {{ card_data }} is the YAML frontmatter auto-built from this class's
# library_name/license/tags/pipeline_tag kwargs below — leave it as-is.
# Every other {{ }} variable is filled per-checkpoint via model_card_kwargs
# at each push_to_hub() call site in DiffVaxImmunization (see
# _model_card_kwargs()); defaults below only apply if a training script calls
# push_to_hub() directly without going through that helper.
_MODEL_CARD_TEMPLATE = """---
{{ card_data }}
---

# {{ model_name | default("DiffVax NestedUNet", true) }}

A [DiffVax](https://github.com/sbartlett97/DiffVax) perturbation-generator
checkpoint: a small trained NestedUNet (UNet++) that produces an additive,
imperceptible perturbation disrupting diffusion-based image editing in a
single forward pass — optimization-free, unlike PhotoGuard/DiffusionGuard.

## Training configuration

- **Attack surrogate(s):** {{ surrogates | default("[not recorded]", true) }}
- **Resolution:** {{ resolution_info | default("[not recorded]", true) }}
- **Loss terms enabled:** {{ loss_terms | default("[not recorded]", true) }}
- **Core hyperparameters:** {{ hyperparams | default("[not recorded]", true) }}

## Training result

- **Checkpoint type:** {{ checkpoint_type | default("[not recorded]", true) }}
- **Epoch:** {{ epoch | default("[not recorded]", true) }}
- **Loss:** {{ loss_value | default("[not recorded]", true) }}

## Intended use & limitations

Research artifact for adversarial image-immunization research (defensive
security / academic use). Not all surrogate/target combinations this
checkpoint may have trained against have been validated end-to-end on GPU —
protection-rate claims for SD3/FLUX/closed-source editing targets should be
independently verified before relying on them; see the training repository's
README ("Project Status" section) for the current verification status at the
time this checkpoint was produced.

## Citation

```
@inproceedings{ozden2026diffvax,
  title={DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing},
  author={Ozden, Tarik Can and Kara, Ozgur and Akcin, Oguzhan and Zaman, Kerem and Srivastava, Shashank and Chinchali, Sandeep P and Rehg, James M},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
```
"""


def _group_norm(num_channels: int) -> nn.GroupNorm:
    """GroupNorm with up to 8 groups — batch-size-independent, safe at bs=1."""
    num_groups = min(8, num_channels)
    # Ensure num_channels is divisible by num_groups (reduce groups if needed)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, norm="group"):
        super().__init__()
        # norm="batch" is the ORIGINAL DiffVax layer, kept only so the published
        # checkpoints (trained with BatchNorm2d) still load; new training uses
        # GroupNorm, which is safe at batch size 1.
        make = nn.BatchNorm2d if norm == "batch" else _group_norm
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = make(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = make(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="diffvax",
    repo_url="https://github.com/sbartlett97/DiffVax",
    license="mit",
    tags=["image-to-image", "adversarial-robustness", "diffusion", "image-immunization"],
    pipeline_tag="image-to-image",
    model_card_template=_MODEL_CARD_TEMPLATE,
):
    """Nested U-Net (UNet++) perturbation generator.

    Inherits ``PyTorchModelHubMixin`` to enable ``save_pretrained()`` /
    ``push_to_hub()`` / ``from_pretrained()`` for HuggingFace Hub integration.
    Constructor arguments are serialised to ``config.json`` so the model
    can be reconstructed exactly from a Hub checkpoint.

    Args:
        num_classes:      Number of output channels (default 3 for RGB).
        input_channels:   Number of input channels (default 3).
        deep_supervision: If True, return intermediate outputs from each
                          decoder level for deep-supervision training.
        epsilon:          Optional hard L-inf output bound in pixel units
                          (e.g. 16 for 16/255). None (default) = unbounded.
        nb_filter:        Filter counts at each of the 5 encoder levels.
                          Default ``[32, 64, 128, 256, 512]`` (~1.8 M params).
                          Use ``[64, 128, 256, 512, 1024]`` for the H6 larger
                          variant (~7 M params) for 1088 px training.
    """

    _DEFAULT_NB_FILTER = [32, 64, 128, 256, 512]

    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 3,
        deep_supervision: bool = False,
        nb_filter: list | None = None,
        epsilon: float | None = None,
        norm: str = "group",
        bound: str = "tanh",
        **kwargs,
    ):
        super().__init__()
        self.norm = norm
        # How the epsilon bound is applied:
        #   "tanh": eps*tanh(x/eps). Smooth, but its gradient vanishes once |x|
        #           exceeds eps, and a fresh NestedUNet outputs |x|~0.4 vs
        #           eps=0.125 (16/255): 55% of outputs start saturated, and training
        #           drove a generator fully saturated with a flat objective.
        #   "ste":  hard clip in the forward pass, identity gradient (straight-
        #           through), like the projection step of PGD.
        self.bound = bound

        # Hard L-inf budget in PIXEL units (0-255 scale), e.g. 16 for
        # "16/255". None = unbounded (legacy checkpoints, incl. the published
        # diffvax_trained.pth, must keep producing identical output).
        self.epsilon = epsilon
        # Image tensors live in [-1, 1], so a pixel budget e maps to 2*e/255.
        self._eps_t = None if epsilon is None else 2.0 * float(epsilon) / 255.0

        if nb_filter is None:
            nb_filter = list(self._DEFAULT_NB_FILTER)

        self.nb_filter = nb_filter
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], norm=norm)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], norm=norm)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], norm=norm)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], norm=norm)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], norm=norm)

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], norm=norm)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], norm=norm)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], norm=norm)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], norm=norm)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], norm=norm)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1], norm=norm)
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2], norm=norm)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0], norm=norm)
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1], norm=norm)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0], norm=norm)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [self._bound(o) for o in (output1, output2, output3, output4)]

        else:
            output = self.final(x0_4)
            return self._bound(output)

    def _bound(self, x):
        """Soft-clip to the hard budget: eps * tanh(x / eps).

        Near-identity for |x| << eps (so warm-starting and early learning are
        unaffected) and saturating at exactly +/-eps. (eps * tanh(x) would
        instead shrink everything by ~eps and slow learning.)
        """
        if self._eps_t is None:
            return x
        if self.bound == "ste":
            # exactly the clipped value forward (x - x.detach() == 0), identity gradient
            return x.clamp(-self._eps_t, self._eps_t).detach() + (x - x.detach())
        return self._eps_t * torch.tanh(x / self._eps_t)

    def init_small_output(self, scale: float = 0.05) -> None:
        """Shrink the final layer so a fresh network starts inside the epsilon
        bound (linear regime) instead of saturated. Call only for training from
        scratch; loading a checkpoint overwrites these weights anyway."""
        with torch.no_grad():
            for name in ("final", "final1", "final2", "final3", "final4"):
                layer = getattr(self, name, None)
                if layer is not None:
                    layer.weight.mul_(scale)
                    layer.bias.zero_()
