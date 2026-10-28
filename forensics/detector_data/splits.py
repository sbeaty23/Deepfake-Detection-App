"""Generator-held-out splits at group level, plus a leakage audit."""
import hashlib

import pandas as pd
import yaml

from .schema import FAKE_LABELS

# Higher priority wins when one group touches several partitions (prevents leakage).
PRIORITY = ["test_heldout_family", "test_heldout_gen", "test_temporal", "test_real_ood", "pool"]


def _u(group_id: str, seed: int) -> float:
    """Deterministic uniform [0,1) from group id: reproducible without stored state."""
    return int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _row_partition(r, cfg) -> str:
    if r["label"] in FAKE_LABELS:
        if r["generator_family"] in cfg.get("heldout_families", []):
            return "test_heldout_family"
        if r["generator_id"] in cfg.get("heldout_generators", []):
            return "test_heldout_gen"
        cut = cfg.get("temporal_cutoff")
        if cut and pd.notna(r["release_date"]) and str(r["release_date"]) >= str(cut):
            return "test_temporal"
    elif r["source_dataset"] in cfg.get("real_ood_sources", []):
        return "test_real_ood"
    return "pool"


def assign_splits(df: pd.DataFrame, cfg_path: str) -> pd.DataFrame:
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg.get("seed", 0)
    df = df.copy()
    df["_part"] = [_row_partition(r, cfg) for _, r in df.iterrows()]
    # group partition = highest-priority partition among members
    rank = {p: i for i, p in enumerate(PRIORITY)}
    gpart = df.groupby("group_id")["_part"].agg(lambda s: min(s, key=rank.get))
    df["_gpart"] = df["group_id"].map(gpart)
    all_real = df.groupby("group_id")["label"].agg(lambda s: (s == "real").all())

    val_f, in_f = cfg.get("val_fraction", 0.1), cfg.get("test_in_family_fraction", 0.1)
    real_held = cfg.get("real_heldout_fraction", 0.15)

    def final(r):
        gp = r["_gpart"]
        if gp == "pool":
            u = _u(r["group_id"], seed)
            if all_real[r["group_id"]] and u < real_held:
                return "test_real_heldout"          # reals paired against held-out fakes
            u = _u("v" + r["group_id"], seed)
            return "val" if u < val_f else "test_in_family" if u < val_f + in_f else "train"
        if r["label"] == "real" and gp in ("test_heldout_family", "test_heldout_gen", "test_temporal"):
            return "test_real_heldout"              # real parent dragged into a held-out group
        return gp

    df["split"] = df.apply(final, axis=1)
    return df.drop(columns=["_part", "_gpart"])


def audit(df: pd.DataFrame) -> None:
    """Hard checks. Raises on leakage."""
    held_fam = df[df["split"] == "test_heldout_family"]
    train = df[df["split"] == "train"]
    # 1. a group lives in exactly one split, except real rows pulled to test_real_heldout
    non_real = df[df["label"] != "real"]
    multi = non_real.groupby("group_id")["split"].nunique()
    assert (multi <= 1).all(), f"{(multi > 1).sum()} fake groups span several splits"
    # 2. no pixel-identical image across train and any test split
    test_ids = set(df[df["split"].str.startswith("test")]["image_id"])
    leak = set(train["image_id"]) & test_ids
    assert not leak, f"{len(leak)} identical images in train and test"
    # 3. held-out generators/families absent from train
    assert not (set(held_fam["generator_family"].dropna()) & set(train["generator_family"].dropna()))
    gen_leak = set(df[df["split"].isin(["test_heldout_gen", "test_temporal"])]["generator_id"].dropna()) \
        & set(train["generator_id"].dropna())
    assert not gen_leak, f"generators in train and held-out: {gen_leak}"
    # 4. license gate (flip on for commercial work)
    print("audit OK\n")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("\nper-generator by split:")
    print(non_real.groupby(["generator_id", "split"]).size().unstack(fill_value=0))