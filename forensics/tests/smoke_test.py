"""Builds a tiny fake corpus (incl. duplicates, a re-encode, a parent/child pair) and runs the pipeline."""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

rng = np.random.default_rng(0)
tmp = Path(tempfile.mkdtemp())

def img(seed):
    r = np.random.default_rng(seed)
    base = r.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((128, 128), Image.BICUBIC)

def make(name, n, off, fmt="png"):
    d = tmp / name; d.mkdir()
    for i in range(n):
        img(off + i).save(d / f"{i}.{fmt}")
    return d

real = make("real", 40, 0)
Path(tmp / "real" / "dupe.jpg").write_bytes(b"")
(tmp / "real" / "dupe.jpg").unlink()
img(0).save(tmp / "real" / "0_resaved.jpg", quality=90)       # near-dup of real/0
ood = make("ood", 10, 5000)
g1 = make("gan", 30, 1000); g2 = make("ldm", 30, 2000); g3 = make("dit", 30, 3000)
g4 = make("sd21", 20, 4000)
img(1).filter(ImageFilter.GaussianBlur(1)).save(g1 / "edit_of_real1.png")  # edit of real/1

import yaml
def fld(**k): return k
cfg = {"sources": [
  {"name": "real", "root": str(real), "fields": fld(label="real")},
  {"name": "ood", "root": str(ood), "fields": fld(label="real")},
  {"name": "gan", "root": str(g1), "fields": fld(label="synthetic", generator_id="sg2", generator_family="gan_style", release_date="2019-12-01")},
  {"name": "ldm", "root": str(g2), "fields": fld(label="synthetic", generator_id="sdxl", generator_family="ldm_unet", release_date="2023-07-01")},
  {"name": "sd21", "root": str(g4), "fields": fld(label="synthetic", generator_id="sd-2.1", generator_family="ldm_unet", release_date="2022-12-01")},
  {"name": "dit", "root": str(g3), "fields": fld(label="synthetic", generator_id="flux", generator_family="dit_flow", release_date="2024-08-01")},
]}
yaml.safe_dump(cfg, open(tmp / "s.yaml", "w"))
yaml.safe_dump({"seed": 1, "heldout_families": ["dit_flow"], "heldout_generators": ["sd-2.1"],
                "real_ood_sources": ["ood"], "val_fraction": .1, "test_in_family_fraction": .1,
                "real_heldout_fraction": .2}, open(tmp / "sp.yaml", "w"))
m = str(tmp / "m.parquet")
run = lambda *a: subprocess.run([sys.executable, "-m", "detector_data.cli", *a], check=True)
run("ingest", "--sources", str(tmp / "s.yaml"), "--out", m, "--workers", "2")
run("group", "--manifest", m)
run("split", "--manifest", m, "--config", str(tmp / "sp.yaml"))
import pandas as pd
df = pd.read_parquet(m)
a = df[df.path == "0.png"].query("source_dataset=='real'").group_id.iloc[0]
b = df[df.path == "0_resaved.jpg"].group_id.iloc[0]
assert a == b, "near-duplicate re-encode should share a group"
print("smoke test passed")