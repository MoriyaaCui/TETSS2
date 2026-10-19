#!/usr/bin/env python3
"""Build grouped train/validation sets before augmenting training positives.

Output coordinates are 0-based half-open windows and 1-based centers.
Overlapping potential windows, linked local negatives, shared TE-copy negative
sources and identical (including reverse-complement) sequences stay together.
This is not a homology-aware split or an automatic held-out-cell-line audit.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.model_selection import GroupShuffleSplit

RADIUS = 100
SIZE = 201
NEG_TYPES = ["N1_Random_TE", "N2_Same_Name", "N3_Shifted"]


def reverse_complement(seq):
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def sequence_key(seq):
    return min(seq, reverse_complement(seq))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Intervals:
    """Merged, 0-based half-open intervals indexed by chromosome."""
    def __init__(self, rows):
        self.data = {}
        for chrom, start, end in rows:
            if end > start:
                self.data.setdefault(str(chrom), []).append((max(0, int(start)), int(end)))
        for chrom, values in self.data.items():
            merged = []
            for start, end in sorted(values):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            self.data[chrom] = ([s for s, _ in merged], [e for _, e in merged])

    def overlaps(self, chrom, start, end):
        if chrom not in self.data:
            return False
        starts, ends = self.data[chrom]
        i = bisect.bisect_left(starts, end) - 1
        return i >= 0 and ends[i] > start


def read_loci(path):
    df = pd.read_csv(path, sep="\t")
    required = {"Peak_Chrom", "Peak_Start", "Peak_End", "Peak_Summit", "Peak_Strand", "TE_Name", "Peak_Class", "RPM", "coordinate_system"}
    if required - set(df):
        raise ValueError(f"{path.name}: missing columns {sorted(required - set(df))}; first run filter_rampage_for_modeling.py")
    if not df.coordinate_system.eq("one-based-inclusive").all():
        raise ValueError(f"{path.name}: expected normalized 1-based inclusive coordinates")
    for c in ["Peak_Start", "Peak_End", "Peak_Summit"]:
        numeric = pd.to_numeric(df[c], errors="raise")
        if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric % 1 != 0).any():
            raise ValueError(f"{path.name}: {c} must contain finite integers")
        df[c] = numeric.astype("int64")
    if not df.Peak_Strand.isin(["+", "-"]).all() or (df.Peak_Start < 1).any() or (df.Peak_End < df.Peak_Start).any():
        raise ValueError(f"{path.name}: invalid intervals/strands")
    return df


def normalize_te_name(value):
    return str(value).split("|")[-1].strip()


def extract(fa, chrom, center, strand):
    start, end = int(center) - 101, int(center) + 100
    if chrom not in fa or start < 0 or end > len(fa[chrom]):
        return None
    seq = str(fa[chrom][start:end]).upper()
    if len(seq) != SIZE or set(seq) - set("ACGT"):
        return None
    return reverse_complement(seq) if strand == "-" else seq


def make_record(chrom, center, strand, label, sample_type, seq, anchor, te_name, source_id, rpm=0.0, assignment=None):
    return dict(chrom=str(chrom), start=int(center) - 101, end=int(center) + 100,
                center_pos=int(center), anchor_pos=int(anchor), strand=strand,
                label=int(label), sample_type=sample_type, augmented=0,
                shift_from_anchor=int(center) - int(anchor), TE_Name=te_name,
                locus_id=f"{chrom}:{anchor}:{strand}", source_id=source_id,
                original_RPM=float(rpm), Assignment_Fraction=assignment, sequence=seq)


def assign_groups(df, max_shift):
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    # Reserve the entire possible augmentation footprint before splitting.
    lo = df.start.to_numpy() - np.where(df.label.eq(1), max_shift, 0)
    hi = df.end.to_numpy() + np.where(df.label.eq(1), max_shift, 0)
    for _, sub in df.groupby("chrom", sort=True):
        ids = sorted(sub.index, key=lambda i: (lo[i], hi[i], i))
        current, stop = None, -1
        for i in ids:
            if current is not None and lo[i] < stop:
                union(current, i)
                stop = max(stop, hi[i])
            else:
                current, stop = i, hi[i]
    for values in [df.sequence.map(sequence_key), df.source_id]:
        seen = {}
        for i, value in enumerate(values):
            if value in seen:
                union(i, seen[value])
            else:
                seen[value] = i
    # Keep local negatives with the positive anchor that generated them.
    positive_ids = {row.locus_id: i for i, row in df[df.label.eq(1)].iterrows()}
    for i, row in df[df.sample_type.eq("N3_Shifted")].iterrows():
        if row.locus_id in positive_ids:
            union(i, positive_ids[row.locus_id])
    roots = [find(i) for i in range(len(df))]
    labels = {root: f"G{k:08d}" for k, root in enumerate(sorted(set(roots)))}
    return [labels[r] for r in roots]


def choose_split(df, val_frac, seed):
    if df.split_group.nunique() < 2:
        raise ValueError("All records belong to one connected group; a separated split is impossible")
    strata = (df.label.astype(str) + ":" + df.sample_type).to_numpy()
    names = sorted(set(strata))
    total = np.array([(strata == name).sum() for name in names])
    best = None
    # Seeded proposals optimize counts only, never model performance.
    splitter = GroupShuffleSplit(n_splits=256, test_size=val_frac, random_state=seed)
    for train, val in splitter.split(df, groups=df.split_group):
        if df.iloc[train].label.nunique() != 2 or df.iloc[val].label.nunique() != 2:
            continue
        counts = np.array([(strata[val] == name).sum() for name in names])
        if (counts == 0).any() or (counts == total).any():
            continue
        score = float(np.mean(np.abs(counts / total - val_frac)))
        if best is None or score < best[0]:
            best = (score, train, val)
    if best is None:
        raise ValueError("No grouped split retains every source/category in both sets; inspect group sizes or add data")
    split = np.full(len(df), "train", dtype=object)
    split[best[2]] = "val"
    return split


def augment_train(base, fa, rng, shifts, sigma, max_shift, forbidden_keys):
    records = base.to_dict("records")
    skipped = 0
    for row in base[base.label.eq(1)].to_dict("records"):
        offsets = np.rint(rng.normal(0, sigma, shifts)).astype(int)
        offsets = np.unique(np.clip(offsets, -max_shift, max_shift))
        for shift in offsets:
            if shift == 0:
                continue
            center = row["anchor_pos"] + int(shift)
            seq = extract(fa, row["chrom"], center, row["strand"])
            if seq is None or sequence_key(seq) in forbidden_keys:
                skipped += 1
                continue
            new = dict(row, center_pos=center, start=center-101, end=center+100,
                       shift_from_anchor=int(shift), augmented=1, sequence=seq)
            records.append(new)
    return pd.DataFrame(records), skipped


def assert_separation(train, val):
    common_groups = set(train.split_group) & set(val.split_group)
    common_seq = set(train.sequence.map(sequence_key)) & set(val.sequence.map(sequence_key))
    val_intervals = Intervals(val[["chrom", "start", "end"]].itertuples(index=False, name=None))
    overlaps = sum(val_intervals.overlaps(r.chrom, r.start, r.end) for r in train.itertuples())
    if common_groups or common_seq or overlaps:
        raise ValueError(f"Split contamination: {len(common_groups)} groups, {len(common_seq)} sequences, {overlaps} overlapping train windows")
    return {"shared_groups": 0, "shared_sequences_including_reverse_complements": 0,
            "overlapping_train_windows": 0, "validation_augmented_rows": int(val.augmented.sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--positives", type=Path, required=True, help="rampage_filtered.tsv")
    p.add_argument("--all-rampage", type=Path, required=True, help="all_rampage_loci.tsv, including failed QC and excluded samples")
    p.add_argument("--repeatmasker", type=Path, required=True, help="Headered TSV: chrom,start,end,strand,repName,repClass")
    p.add_argument("--repeatmasker-coordinates", choices=["zero-based-half-open", "one-based-inclusive"], required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--exclude-bed", type=Path, help="0-based half-open full evaluation windows; excluded from both development sets")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shifts-per-positive", type=int, default=10)
    p.add_argument("--shift-sigma", type=float, default=10.0)
    p.add_argument("--max-shift", type=int, default=50)
    p.add_argument("--blacklist-buffer", type=int, default=500)
    p.add_argument("--te-classes", nargs="+", default=["LINE", "SINE", "LTR", "DNA"])
    p.add_argument("--attempts-per-negative", type=int, default=100)
    a = p.parse_args()
    if not 0 < a.val_frac < 1 or min(a.shifts_per_positive, a.max_shift, a.blacklist_buffer) < 0 or a.shift_sigma < 0 or a.attempts_per_negative < 1:
        p.error("Invalid split/augmentation/sampling settings")
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        p.error("Output directory must be empty; do not overwrite an earlier dataset")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    fa = Fasta(str(a.fasta), as_raw=True)
    positives = read_loci(a.positives)
    all_loci = read_loci(a.all_rampage)
    # The blacklist must cover every eligible peak as well as the other inputs.
    peak_keys = lambda d: set(d[["Peak_Chrom", "Peak_Start", "Peak_End", "Peak_Summit", "Peak_Strand"]].itertuples(index=False, name=None))
    if not peak_keys(positives) <= peak_keys(all_loci):
        raise ValueError("--all-rampage does not include all eligible positive loci")
    missing_chrom = (set(positives.Peak_Chrom) | set(all_loci.Peak_Chrom)) - set(fa.keys())
    if missing_chrom:
        raise ValueError(f"RAMPAGE chromosomes missing from FASTA: {sorted(missing_chrom)}; provide a matching reference")
    blacklist = Intervals((r.Peak_Chrom, r.Peak_Start-1-a.blacklist_buffer, r.Peak_End+a.blacklist_buffer) for r in all_loci.itertuples())
    reserved_rows = []
    if a.exclude_bed:
        with open(a.exclude_bed) as f:
            for line in f:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                fields = line.split()
                if len(fields) < 3:
                    raise ValueError("--exclude-bed must contain at least three columns")
                chrom, start, end = fields[0], int(fields[1]), int(fields[2])
                if chrom not in fa or start < 0 or end <= start or end > len(fa[chrom]):
                    raise ValueError(f"Invalid reserved BED interval: {line.strip()}")
                reserved_rows.append((chrom, start, end))
        if not reserved_rows:
            raise ValueError("--exclude-bed contains no intervals")
    reserved = Intervals(reserved_rows)
    # Duplicate evidence at one same-strand summit contributes one positive.
    positives = positives.assign(_priority=positives.Peak_Class.map({"High_confidence": 0, "RAMPAGE_only": 1}))
    positives = positives.sort_values(["Peak_Chrom", "Peak_Summit", "Peak_Strand", "_priority", "RPM"], ascending=[True, True, True, True, False], kind="stable")
    evidence_rows = len(positives)
    positives = positives.drop_duplicates(["Peak_Chrom", "Peak_Summit", "Peak_Strand"])
    records, usable, pos_keys = [], [], set()
    excluded_reserved = excluded_invalid = 0
    for row in positives.to_dict("records"):
        chrom, center, strand = row["Peak_Chrom"], int(row["Peak_Summit"]), row["Peak_Strand"]
        if reserved.overlaps(chrom, center-101-a.max_shift, center+100+a.max_shift):
            excluded_reserved += 1
            continue
        seq = extract(fa, chrom, center, strand)
        if seq is None:
            excluded_invalid += 1
            continue
        kind = "P1_High_Conf" if row["Peak_Class"] == "High_confidence" else "P2_Rampage_Only"
        record = make_record(chrom, center, strand, 1, kind, seq, center,
                             normalize_te_name(row["TE_Name"]), f"POS:{chrom}:{center}:{strand}", row["RPM"], row.get("Assignment_Fraction"))
        records.append(record)
        usable.append(row)
        pos_keys.add(sequence_key(seq))
    if len(records) < 4:
        raise ValueError("Too few usable positive loci")
    rmsk = pd.read_csv(a.repeatmasker, sep="\t")
    required = {"chrom", "start", "end", "strand", "repName", "repClass"}
    if required - set(rmsk):
        raise ValueError(f"RepeatMasker input missing {sorted(required - set(rmsk))}; UCSC raw tables need an explicit column mapping")
    for c in ["start", "end"]:
        numeric = pd.to_numeric(rmsk[c], errors="raise")
        if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric % 1 != 0).any():
            raise ValueError(f"RepeatMasker {c} must contain finite integers")
        rmsk[c] = numeric.astype("int64")
    if a.repeatmasker_coordinates == "one-based-inclusive":
        rmsk["start"] -= 1
    if (rmsk.start < 0).any() or (rmsk.end <= rmsk.start).any() or not rmsk.strand.isin(["+", "-"]).all():
        raise ValueError("RepeatMasker has invalid coordinates/strands; UCSC C strand must be converted to -")
    rmsk["repName"] = rmsk.repName.map(normalize_te_name)
    rmsk = rmsk[rmsk.repClass.isin(a.te_classes)].copy()
    missing_rmsk = sorted(set(rmsk.chrom) - set(fa.keys()))
    rmsk = rmsk[rmsk.chrom.isin(fa.keys())]
    # A full 201-bp negative window must fit inside the annotated TE for N1/N2.
    candidates = rmsk[rmsk.end-rmsk.start >= SIZE].to_dict("records")
    name_groups = {}
    for row in candidates:
        # N2 uses other copies without any known peak/buffer in the whole TE,
        # not a distant window within the same long signal-bearing insertion.
        if not blacklist.overlaps(row["chrom"], row["start"], row["end"]):
            name_groups.setdefault(row["repName"], []).append(row)
    same_name_positives = [r for r in usable if normalize_te_name(r["TE_Name"]) in name_groups]
    used_windows, negative_keys = set(), set()
    n_pos = len(records)
    targets = [n_pos // 3 + int(i < n_pos % 3) for i in range(3)]
    counts = {}
    for kind, target in zip(NEG_TYPES, targets):
        found = 0
        if (kind == NEG_TYPES[0] and not candidates) or (kind == NEG_TYPES[1] and not same_name_positives):
            raise ValueError(f"No eligible source for {kind}; check TE names and lengths")
        for _ in range(max(1, target) * a.attempts_per_negative):
            if found >= target:
                break
            if kind == NEG_TYPES[2]:
                src = usable[int(rng.integers(len(usable)))]
                chrom, strand, anchor = src["Peak_Chrom"], src["Peak_Strand"], int(src["Peak_Summit"])
                distance = int(rng.integers(a.blacklist_buffer+RADIUS+1, a.blacklist_buffer+RADIUS+501))
                center = anchor + int(rng.choice([-1, 1])) * distance
                name = normalize_te_name(src["TE_Name"])
                source_id = f"LOCAL:{chrom}:{anchor}:{strand}"
            else:
                pool = candidates
                if kind == NEG_TYPES[1]:
                    src = same_name_positives[int(rng.integers(len(same_name_positives)))]
                    pool = name_groups[normalize_te_name(src["TE_Name"])]
                te = pool[int(rng.integers(len(pool)))]
                chrom, strand = te["chrom"], te["strand"]
                center = int(rng.integers(te["start"]+101, te["end"]-100+1))
                anchor, name = center, te["repName"]
                source_id = f"TE:{chrom}:{te['start']}:{te['end']}:{strand}"
            start, end = center-101, center+100
            key = (chrom, start, end)  # Do not duplicate the opposite strand either.
            if key in used_windows or blacklist.overlaps(chrom, start, end) or reserved.overlaps(chrom, start, end):
                continue
            seq = extract(fa, chrom, center, strand)
            if seq is None:
                continue
            seqkey = sequence_key(seq)
            if seqkey in pos_keys or seqkey in negative_keys:
                continue
            records.append(make_record(chrom, center, strand, 0, kind, seq, anchor, name, source_id))
            used_windows.add(key)
            negative_keys.add(seqkey)
            found += 1
        counts[kind] = found
        if found < target:
            raise ValueError(f"{kind}: generated {found}/{target}; add eligible sources or increase --attempts-per-negative. No silent class replacement was performed")
    base = pd.DataFrame(records).reset_index(drop=True)
    base["record_id"] = [f"R{i:09d}" for i in range(len(base))]
    base["split_group"] = assign_groups(base, a.max_shift)
    base["split"] = choose_split(base, a.val_frac, a.seed)
    train, val = base[base.split.eq("train")].copy(), base[base.split.eq("val")].copy()
    forbidden_keys = set(val.sequence.map(sequence_key)) | set(base[base.label.eq(0)].sequence.map(sequence_key))
    augmented, skipped = augment_train(train, fa, rng, a.shifts_per_positive, a.shift_sigma, a.max_shift, forbidden_keys)
    augmented["record_id"] = [f"A{i:09d}" for i in range(len(augmented))]
    audit = assert_separation(augmented, val)
    audit.update(positive_evidence_rows=evidence_rows, unique_positive_summits=len(positives), usable_positive_loci=n_pos,
                 excluded_positive_reserved=excluded_reserved, excluded_positive_invalid_sequence=excluded_invalid,
                 missing_repeatmasker_chromosomes=missing_rmsk, negative_counts=counts,
                 split_groups=int(base.split_group.nunique()), largest_group=int(base.split_group.value_counts().max()),
                 realized_validation_fraction=float(len(val)/len(base)), skipped_augmented_sequences=skipped,
                 heldout_coordinate_exclusion_applied=bool(a.exclude_bed), homology_exclusion_applied=False)
    files = {"base_dataset.tsv": base, "train_summit_centered.tsv": train,
             "train_position_augmented.tsv": augmented, "validation.tsv": val}
    for name, df in files.items():
        df.to_csv(a.out_dir / name, sep="\t", index=False)
    base[["record_id", "locus_id", "source_id", "chrom", "start", "end", "label", "sample_type", "split_group", "split"]].to_csv(a.out_dir / "split_manifest.tsv", sep="\t", index=False)
    summary = pd.concat([df.assign(dataset=name) for name, df in files.items()]).groupby(["dataset", "label", "sample_type", "augmented"]).size().reset_index(name="count")
    summary.to_csv(a.out_dir / "dataset_summary.tsv", sep="\t", index=False)
    config = {k: (v.name if isinstance(v, Path) else v) for k, v in vars(a).items()}
    config["input_sha256"] = {k: sha256(getattr(a, k)) for k in ["fasta", "positives", "all_rampage", "repeatmasker"] + (["exclude_bed"] if a.exclude_bed else [])}
    config["output_sha256"] = {name: sha256(a.out_dir / name) for name in files}
    config.update(split_method="overlap_and_sequence_grouped_before_augmentation", validation_augmentation=False,
                  train_sampling="class_balanced_equal_positive_locus_weight", coordinate_system="start/end: zero-based-half-open; center_pos/anchor_pos: one-based")
    (a.out_dir / "dataset_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (a.out_dir / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    fa.close()
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
