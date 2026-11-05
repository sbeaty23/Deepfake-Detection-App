"""Duplicate edges: exact pixel hash, perceptual hash (banded LSH), optional embeddings."""
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


def exact_edges(df: pd.DataFrame):
    edges = []
    for _, idx in df.groupby("image_id").indices.items():
        for a, b in zip(idx[:-1], idx[1:]):
            edges.append((int(a), int(b)))
    return edges


def phash_edges(df: pd.DataFrame, max_dist: int = 6, max_bucket: int = 5000):
    """Pigeonhole LSH: 64-bit hash split into (max_dist+1) bands. Any pair within
    max_dist differs in at most max_dist bands, so at least one band matches exactly."""
    hashes = np.array([int(h, 16) if isinstance(h, str) else 0 for h in df["phash"]], dtype=np.uint64)
    n_bands = max_dist + 1
    bounds = np.linspace(0, 64, n_bands + 1).astype(int)
    seen, edges = set(), []
    for b in range(n_bands):
        lo, hi = bounds[b], bounds[b + 1]
        mask = np.uint64((1 << (hi - lo)) - 1)
        buckets = defaultdict(list)
        for i, h in enumerate(hashes):
            buckets[int((h >> np.uint64(lo)) & mask)].append(i)
        for members in buckets.values():
            if len(members) < 2 or len(members) > max_bucket:
                continue  # oversized bucket = degenerate hash (e.g. flat images); skip
            for i, j in combinations(members, 2):
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                if bin(int(hashes[i] ^ hashes[j])).count("1") <= max_dist:
                    edges.append((i, j))
    return edges


def embedding_edges(emb: np.ndarray, thr: float, k: int = 10):
    """Cosine-similarity kNN edges. emb: (n, d) rows aligned with the manifest.
    Use DINOv2 for near-dup images, ArcFace for same-identity (thr ~0.4-0.5; calibrate!)."""
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
    try:
        import faiss
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb.astype("float32"))
        sims, nbrs = index.search(emb.astype("float32"), k + 1)
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
        d, nbrs = nn.kneighbors(emb)
        sims = 1 - d
    edges = []
    for i in range(len(emb)):
        for s, j in zip(sims[i], nbrs[i]):
            if j != i and s >= thr:
                edges.append((i, int(j)))
    return edges