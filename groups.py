"""Connected-component grouping. A group_id is the unit that is never split across partitions."""
import hashlib

import pandas as pd


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# Rows sharing ALL non-null values in a column set are linked. Edit to match your data.
DEFAULT_KEY_SETS = [
    ["generator_id", "seed", "prompt"],        # same generation lineage
    ["source_dataset", "source_owner_id"],     # same photographer / Flickr owner
    ["identity_cluster_id"],                   # same real identity
    ["target_identity_cluster_id"],            # swap targets
]


def build_groups(df: pd.DataFrame, edges, key_sets=DEFAULT_KEY_SETS) -> pd.DataFrame:
    uf = UnionFind(len(df))
    for a, b in edges:
        uf.union(a, b)
    # key-based links
    for cols in key_sets:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        sub = df[cols].dropna()
        for _, idx in sub.groupby(cols).indices.items():
            rows = sub.index[idx]
            for r in rows[1:]:
                uf.union(int(rows[0]), int(r))
    # edited/swapped image is linked to its real parent
    first = df.drop_duplicates("image_id").set_index("image_id").index
    pos = {iid: i for i, iid in zip(df.index, df["image_id"])}
    for i, pid in zip(df.index, df["parent_image_id"]):
        if isinstance(pid, str) and pid in pos:
            uf.union(int(i), int(pos[pid]))
    roots = [uf.find(i) for i in range(len(df))]
    df = df.copy()
    df["dup_cluster_id"] = roots
    # stable id: hash of smallest image_id in the component
    rep = df.groupby("dup_cluster_id")["image_id"].transform("min")
    df["group_id"] = [hashlib.sha1(r.encode()).hexdigest()[:12] for r in rep]
    df["dup_cluster_id"] = df["dup_cluster_id"].astype(str)
    return df