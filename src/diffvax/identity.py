"""Face-identity similarity for the evaluation (does the edit still depict the person?).

Uses insightface's detector + ArcFace recogniser. NOTE: insightface's bundled
model packs (e.g. buffalo_l) are licensed for NON-COMMERCIAL research use only.
Fine for internal evaluation; a shipped product needs a commercially licensed
face model.
"""
from typing import Optional

import numpy as np
from PIL import Image


class IdentityScorer:
    def __init__(self, model: str = "buffalo_l", det_size: int = 640, gpu: bool = True):
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model, providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0 if gpu else -1, det_size=(det_size, det_size))
        self._cache = {}

    def embed(self, img: Image.Image) -> Optional[np.ndarray]:
        key = id(img)
        if key in self._cache:
            return self._cache[key]
        bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
        faces = self.app.get(bgr)
        emb = None
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            emb = f.normed_embedding
        self._cache[key] = emb
        return emb

    def __call__(self, ref: Image.Image, edit: Image.Image) -> float:
        a, b = self.embed(ref), self.embed(edit)
        if a is None or b is None:
            return float("nan")
        return float(np.dot(a, b))
