# detector_data

Manifest, dedup, grouping and generator-held-out splits for a deepfake photo detector.

    pip install -r requirements.txt

    # 1. edit configs/sources.yaml (paths, generator metadata, licenses)
    python -m detector_data.cli ingest --sources configs/sources.yaml --out manifest.parquet

    # 2. (optional) embeddings for semantic dedup / identity grouping
    #    python -c "from detector_data.embed import *; ..." -> dino.npy, arcface.npy (row-aligned)
    python -m detector_data.cli group --manifest manifest.parquet \
        --embeddings dino.npy:0.92 arcface.npy:0.45     # calibrate thresholds on labelled pairs

    # 3. edit configs/splits.yaml (held-out families / generators / cutoff), then
    python -m detector_data.cli split --manifest manifest.parquet --config configs/splits.yaml
    python -m detector_data.cli audit --manifest manifest.parquet   # run in CI before every training run

Smoke test: `PYTHONPATH=. python tests/smoke_test.py`

Sidecar CSV columns understood: path, seed, prompt, source_owner_id, parent_path,
license_spdx, release_date, identity_cluster_id, ... (any manifest column).