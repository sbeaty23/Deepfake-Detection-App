"""Walk source directories, hash and describe every image, merge sidecar metadata."""
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import pandas as pd
import yaml
from PIL import Image, ImageOps

from .schema import ensure_columns, validate

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _describe(path_str: str) -> dict | None:
    p = Path(path_str)
    try:
        raw = p.read_bytes()
        with Image.open(p) as im0:
            fmt = im0.format
            qt = getattr(im0, "quantization", None)
            im = ImageOps.exif_transpose(im0).convert("RGB")
        # pixel hash is invariant to metadata and container re-encodes with identical pixels
        pix = hashlib.sha256(f"{im.size}".encode() + im.tobytes()).hexdigest()
        q = float(sum(qt[0]) / len(qt[0])) if qt and 0 in qt else None
        return dict(
            image_id=pix, file_sha256=hashlib.sha256(raw).hexdigest(),
            width=im.width, height=im.height, format=fmt, jpeg_q_proxy=q,
            phash=str(imagehash.phash(im)),
        )
    except Exception as e:  # corrupt/unreadable files are logged, not fatal
        print(f"[skip] {path_str}: {e}")
        return None


def ingest_source(src: dict, workers: int = 8) -> pd.DataFrame:
    root = Path(src["root"])
    files = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in EXTS)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_describe, files, chunksize=64))
    recs = []
    for f, r in zip(files, rows):
        if r is None:
            continue
        r["path"] = str(Path(f).relative_to(root))
        recs.append(r)
    df = pd.DataFrame(recs)
    # constant per-source fields from config (label, generator, license, ...)
    for k, v in src.get("fields", {}).items():
        df[k] = v
    df["source_dataset"] = src["name"]
    df["_root"] = str(root)
    # optional sidecar CSV keyed by relative `path` (seed, prompt, owner, parent_path, ...)
    if src.get("sidecar"):
        side = pd.read_csv(src["sidecar"])
        df = df.merge(side, on="path", how="left", suffixes=("", "_side"))
        for c in [c for c in df.columns if c.endswith("_side")]:
            base = c[:-5]
            df[base] = df[c].where(df[c].notna(), df.get(base))
            df = df.drop(columns=c)
    return df


def build_manifest(config_path: str, out: str, workers: int = 8) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path))
    parts = [ingest_source(s, workers) for s in cfg["sources"]]
    df = pd.concat(parts, ignore_index=True)
    # resolve parent_path (relative to the same source) -> parent_image_id
    if "parent_path" in df.columns and df["parent_path"].notna().any():
        key = df.set_index(["source_dataset", "path"])["image_id"]
        pairs = list(zip(df["source_dataset"], df["parent_path"]))
        df["parent_image_id"] = [key.get(p) if isinstance(p[1], str) else None for p in pairs]
    df = ensure_columns(df).drop(columns=["_root"], errors="ignore")
    validate(df)
    df.to_parquet(out, index=False)
    print(f"manifest: {len(df)} rows -> {out}")
    return df