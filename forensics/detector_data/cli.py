import argparse

import numpy as np
import pandas as pd

from .dedup import embedding_edges, exact_edges, phash_edges
from .groups import build_groups
from .ingest import build_manifest
from .splits import assign_splits, audit


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("ingest"); a.add_argument("--sources", required=True); a.add_argument("--out", default="manifest.parquet"); a.add_argument("--workers", type=int, default=8)
    b = sp.add_parser("group"); b.add_argument("--manifest", default="manifest.parquet"); b.add_argument("--phash-dist", type=int, default=6)
    b.add_argument("--embeddings", nargs="*", default=[], help="path.npy:threshold (rows aligned to manifest)")
    c = sp.add_parser("split"); c.add_argument("--manifest", default="manifest.parquet"); c.add_argument("--config", required=True)
    d = sp.add_parser("audit"); d.add_argument("--manifest", default="manifest.parquet")
    args = ap.parse_args()

    if args.cmd == "ingest":
        build_manifest(args.sources, args.out, args.workers)
    elif args.cmd == "group":
        df = pd.read_parquet(args.manifest)
        edges = exact_edges(df) + phash_edges(df, args.phash_dist)
        for spec in args.embeddings:
            path, thr = spec.rsplit(":", 1)
            emb = np.load(path)
            ok = ~np.isnan(emb).any(axis=1)             # rows without a face are skipped
            idx = np.flatnonzero(ok)
            edges += [(int(idx[i]), int(idx[j])) for i, j in embedding_edges(emb[ok], float(thr))]
        df = build_groups(df, edges)
        df.to_parquet(args.manifest, index=False)
        print(f"{len(df)} rows -> {df['group_id'].nunique()} groups")
    elif args.cmd == "split":
        df = assign_splits(pd.read_parquet(args.manifest), args.config)
        df.to_parquet(args.manifest, index=False)
        audit(df)
    elif args.cmd == "audit":
        audit(pd.read_parquet(args.manifest))


if __name__ == "__main__":
    main()