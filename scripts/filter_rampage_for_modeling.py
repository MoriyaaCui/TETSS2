#!/usr/bin/env python3
"""Filter RAMPAGE records; normalize peak coordinates to 1-based inclusive.

TE_Name is an existing annotation, not a new overlap calculation. Entropy is
reported as a flag and does not determine eligibility. See --help for inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--peak-coordinates", choices=["one-based-inclusive", "zero-based-half-open"], required=True)
    p.add_argument("--summit-coordinates", choices=["one-based", "zero-based"], required=True)
    p.add_argument("--rpm-threshold", type=float, default=0.5)
    p.add_argument("--assignment-threshold", type=float, default=0.5)
    p.add_argument("--entropy-threshold", type=float, default=2.5)
    p.add_argument("--sample-column", help="Column carrying exact sample identifiers or cell-line labels")
    p.add_argument("--exclude-samples", nargs="+", default=[], help="Exact values in --sample-column; no fuzzy matching")
    a = p.parse_args()
    if a.exclude_samples and not a.sample_column:
        p.error("--exclude-samples requires --sample-column")
    if a.rpm_threshold < 0 or not 0 <= a.assignment_threshold <= 1:
        p.error("RPM threshold must be nonnegative; assignment threshold must be between 0 and 1")
    df = pd.read_csv(a.input, sep="\t")
    required = {"Peak_Chrom", "Peak_Start", "Peak_End", "Peak_Summit", "Peak_Strand", "TE_Name", "RPM", "Assignment_Fraction", "Peak_Class"}
    missing = required - set(df)
    if missing:
        p.error(f"Missing input columns: {sorted(missing)}")
    if a.sample_column and a.sample_column not in df:
        p.error(f"Sample column not found: {a.sample_column}")
    # Refuse to reinterpret a previous normalized output as raw input.
    if "coordinate_system" in df:
        p.error("Input is already normalized; use the raw source table")
    for column in ["Peak_Start", "Peak_End", "Peak_Summit"]:
        values = pd.to_numeric(df[column], errors="raise")
        if values.isna().any() or not np.isfinite(values).all() or (values % 1 != 0).any():
            p.error(f"{column} must contain finite integer coordinates")
        df[column] = values.astype("int64")
    for column in ["RPM", "Assignment_Fraction"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if df[column].isna().any() or not np.isfinite(df[column]).all():
            p.error(f"{column} contains missing/nonfinite values")
    if (df.RPM < 0).any() or not df.Assignment_Fraction.between(0, 1).all():
        p.error("RPM must be nonnegative and Assignment_Fraction must be in [0, 1]")
    df["Peak_Chrom"] = df.Peak_Chrom.astype("string").str.strip()
    df["Peak_Strand"] = df.Peak_Strand.astype("string").str.strip()
    df["Peak_Class"] = df.Peak_Class.astype("string").str.strip()
    if not df.Peak_Strand.isin(["+", "-"]).all() or df.Peak_Chrom.isna().any() or df.Peak_Chrom.eq("").any():
        p.error("Every record requires a chromosome and a + or - strand")
    if not df.Peak_Class.isin(["High_confidence", "RAMPAGE_only"]).all():
        p.error("Peak_Class must be High_confidence or RAMPAGE_only")
    df["source_row_id"] = np.arange(len(df))
    if a.peak_coordinates == "zero-based-half-open":
        df["Peak_Start"] += 1
    if a.summit_coordinates == "zero-based":
        df["Peak_Summit"] += 1
    if (df.Peak_Start < 1).any() or (df.Peak_End < df.Peak_Start).any() or (df.Peak_Summit < 1).any():
        p.error("Invalid peak/summit coordinates after normalization")
    df["coordinate_system"] = "one-based-inclusive"
    df["Summit_Pos"] = df.Peak_Summit
    df["Peak_Length"] = df.Peak_End - df.Peak_Start + 1
    df["log1p_RPM"] = np.log1p(df.RPM)
    df["Summit_Inside_Peak"] = df.Peak_Summit.between(df.Peak_Start, df.Peak_End)
    names = df.TE_Name.astype("string").fillna("").str.strip()
    df["Has_TE"] = ~names.str.lower().isin(["", ".", "na", "nan", "none", "null", "-"])
    df["Assignment_OK"] = df.Assignment_Fraction >= a.assignment_threshold
    df["RPM_OK"] = df.RPM >= a.rpm_threshold
    entropy = pd.to_numeric(df["Entropy"], errors="coerce") if "Entropy" in df else pd.Series(np.nan, index=df.index)
    df["Entropy_High"] = entropy >= a.entropy_threshold
    df["Peak_Type"] = np.where(df.Peak_Length <= 10, "narrow", "broad")
    df["Excluded_Sample"] = False
    exclusion_counts = {}
    if a.exclude_samples:
        labels = df[a.sample_column].astype("string").str.strip()
        exclusion_counts = {x: int(labels.eq(x).sum()) for x in a.exclude_samples}
        absent = [x for x, count in exclusion_counts.items() if count == 0]
        if absent:
            p.error(f"Excluded labels not found: {absent}; verify exact values and the selected column")
        df["Excluded_Sample"] = labels.isin(a.exclude_samples)
    eligible = df.Summit_Inside_Peak & df.Has_TE & df.Assignment_OK & df.RPM_OK & ~df.Excluded_Sample
    filtered = df[eligible].copy()
    if filtered.empty:
        p.error("No records passed filtering")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if (a.out_dir / "filter_config.json").exists():
        p.error("Output already contains a previous run; choose a new --out-dir")
    subsets = {
        "rampage_filtered": filtered,
        "rampage_filtered_high_confidence": filtered[filtered.Peak_Class == "High_confidence"],
        "rampage_filtered_rampage_only": filtered[filtered.Peak_Class == "RAMPAGE_only"],
        "rampage_filtered_narrow": filtered[filtered.Peak_Type == "narrow"],
        "rampage_filtered_broad": filtered[filtered.Peak_Type == "broad"],
        "rampage_filtered_high_confidence_narrow": filtered[(filtered.Peak_Class == "High_confidence") & (filtered.Peak_Type == "narrow")],
        "rampage_filtered_high_confidence_broad": filtered[(filtered.Peak_Class == "High_confidence") & (filtered.Peak_Type == "broad")],
        "rampage_filtered_entropy_high": filtered[filtered.Entropy_High],
    }
    for name, sub in subsets.items():
        sub.to_csv(a.out_dir / f"{name}.tsv", sep="\t", index=False)
    df.assign(Eligible=eligible).to_csv(a.out_dir / "filter_audit.tsv", sep="\t", index=False)
    # Include excluded samples and failed QC in the negative-sampling blacklist.
    df.to_csv(a.out_dir / "all_rampage_loci.tsv", sep="\t", index=False)
    rows = [{"dataset": "all_input", "count": len(df)}]
    for name, sub in subsets.items():
        rows.append({"dataset": name, "count": len(sub), "mean_rpm": sub.RPM.mean(), "mean_peak_length": sub.Peak_Length.mean()})
    pd.DataFrame(rows).to_csv(a.out_dir / "filter_summary.csv", index=False)
    config = {k: (v.name if isinstance(v, Path) else v) for k, v in vars(a).items()}
    config.update(input_sha256=sha256(a.input), input_rows=len(df), retained_rows=len(filtered), exclusion_counts=exclusion_counts, output_coordinates="one-based-inclusive")
    (a.out_dir / "filter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Retained {len(filtered)} of {len(df)} records in {a.out_dir}")


if __name__ == "__main__":
    main()
