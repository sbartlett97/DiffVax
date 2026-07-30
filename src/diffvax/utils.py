"""Utility functions for DiffVax."""

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import random
import json
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
