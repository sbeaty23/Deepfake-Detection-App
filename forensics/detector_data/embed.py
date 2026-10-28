"""Optional embedding extraction. Writes .npy aligned with manifest row order."""
import numpy as np
import pandas as pd
from PIL import Image


def dino_embeddings(df: pd.DataFrame, roots: dict, model="facebook/dinov2-base", bs=64):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    proc, net = AutoImageProcessor.from_pretrained(model), AutoModel.from_pretrained(model).eval()
    out = []
    paths = [f"{roots[s]}/{p}" for s, p in zip(df["source_dataset"], df["path"])]
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        with torch.no_grad():
            h = net(**proc(images=ims, return_tensors="pt")).last_hidden_state[:, 0]
        out.append(h.numpy())
    return np.concatenate(out)


def arcface_embeddings(df: pd.DataFrame, roots: dict):
    """Returns (n,512) with NaN rows where no face was found."""
    import cv2
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l"); app.prepare(ctx_id=-1)
    out = np.full((len(df), 512), np.nan, dtype="float32")
    for i, (s, p) in enumerate(zip(df["source_dataset"], df["path"])):
        faces = app.get(cv2.imread(f"{roots[s]}/{p}"))
        if faces:
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            out[i] = f.normed_embedding
    return out