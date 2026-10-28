"""Manifest schema. One row per image. Missing columns are added as nulls."""
import pandas as pd

CORE = [
    # identity of the asset
    "image_id", "file_sha256", "path", "width", "height", "format", "jpeg_q_proxy",
    # label
    "label", "label_confidence",
    # provenance / license
    "source_dataset", "source_uri", "source_owner_id", "license_spdx", "license_tier",
    "attribution_text", "redistributable", "commercial_train_ok", "consent_basis",
    # real-image fields
    "identity_cluster_id",
    # synthetic fields
    "generator_id", "generator_family", "base_model_id", "release_date", "gen_method",
    "seed", "prompt", "parent_path", "parent_image_id", "target_identity_cluster_id",
    # dedup / grouping / split
    "phash", "dup_cluster_id", "group_id", "split",
]
LABELS = {"real", "synthetic", "edited_real", "face_swap"}
FAKE_LABELS = LABELS - {"real"}


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in CORE:
        if c not in df.columns:
            df[c] = None
    extra = [c for c in df.columns if c not in CORE]
    return df[CORE + extra]


def validate(df: pd.DataFrame) -> None:
    bad = set(df["label"].dropna()) - LABELS
    assert not bad, f"unknown labels: {bad}"
    fake = df[df["label"].isin(FAKE_LABELS)]
    assert fake["generator_id"].notna().all(), "every non-real row needs generator_id"
    assert df["image_id"].notna().all() and df["path"].notna().all()