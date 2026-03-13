#!/usr/bin/env python3
"""
Relabel tempo_sections.csv using distribution-based thresholds instead of
arbitrary fixed values. Labels are assigned based on quantiles of the speed
score across the entire dataset.

Supports --song-level mode: combine adjacent categories into slow/medium/fast,
assign overall labels per song, and only split into segments when there is a
huge tempo change.
"""

import argparse
import sys

import pandas as pd
import numpy as np

SPEED_CATEGORIES = ["slow", "medium", "fast", "very_fast"]
SPEED_CATEGORY_TO_NUM = {c: i for i, c in enumerate(SPEED_CATEGORIES)}

# 3-category mapping: very_fast -> fast
THREE_CATEGORIES = ["slow", "medium", "fast"]
THREE_CATEGORY_TO_NUM = {c: i for i, c in enumerate(THREE_CATEGORIES)}


def to_three_categories(cat: str) -> str:
    """Map 4-category to 3-category (very_fast -> fast)."""
    return "fast" if cat == "very_fast" else cat


def compute_speed_score(row: pd.Series) -> float:
    """Compute the same speed score used in tempo.py."""
    onsets_per_sec = row["onsets_per_sec"]
    med = row["median_note_length_sec"]
    ioi = row["median_ioi_sec"]
    if pd.isna(med):
        med = 0.5
    if pd.isna(ioi):
        ioi = 0.0
    return onsets_per_sec / (1 + (med + ioi) / 2)


def assign_category_from_quantiles(
    score: float,
    q25: float,
    q50: float,
    q75: float,
) -> tuple[str, int]:
    """Assign speed category based on quantile thresholds."""
    if score <= q25:
        cat = "slow"
    elif score <= q50:
        cat = "medium"
    elif score <= q75:
        cat = "fast"
    else:
        cat = "very_fast"
    return cat, SPEED_CATEGORY_TO_NUM[cat]


def cat_distance(a: str, b: str) -> int:
    """Distance between two 3-category labels (0=same, 1=adjacent, 2=opposite)."""
    na, nb = THREE_CATEGORY_TO_NUM[a], THREE_CATEGORY_TO_NUM[b]
    return abs(na - nb)


def compute_song_level_labels(
    df: pd.DataFrame,
    huge_change_min_pct: float = 0.25,
) -> pd.DataFrame:
    """
    Combine sections into song-level labels (slow/medium/fast). Merge adjacent
    categories. Only split into segments when there is a huge tempo change
    (e.g., slow vs fast spanning a substantial portion of the song).
    """
    df = df.copy()
    df["cat3"] = df["speed_category"].apply(to_three_categories)

    rows = []
    for piece_idx, grp in df.groupby("piece_index"):
        grp = grp.sort_values("start_sec").reset_index(drop=True)
        meta = grp.iloc[0]
        total_length = meta["total_length_sec"],
        total_length = total_length[0] if isinstance(total_length, tuple) else total_length

        # Merge consecutive sections with same or adjacent categories
        segments = []
        seg_start = grp.iloc[0]["start_sec"]
        seg_end = grp.iloc[0]["end_sec"]
        seg_cat_dur = {grp.iloc[0]["cat3"]: seg_end - seg_start}

        for i in range(1, len(grp)):
            row = grp.iloc[i]
            start, end = row["start_sec"], row["end_sec"]
            cat = row["cat3"]
            dur = end - start

            # Merge if same category or adjacent (slow+medium or medium+fast)
            cats_in_seg = set(seg_cat_dur.keys()) | {cat}
            can_merge = cat in seg_cat_dur or any(
                cat_distance(cat, c) == 1 for c in seg_cat_dur
            )
            if can_merge:
                seg_end = end
                seg_cat_dur[cat] = seg_cat_dur.get(cat, 0) + dur
            else:
                seg_cat = max(seg_cat_dur, key=seg_cat_dur.get)
                segments.append((seg_start, seg_end, seg_cat, seg_end - seg_start))
                seg_start, seg_end = start, end
                seg_cat_dur = {cat: dur}

        seg_cat = max(seg_cat_dur, key=seg_cat_dur.get)
        segments.append((seg_start, seg_end, seg_cat, seg_end - seg_start))

        # Dominant category by duration
        cat_durations = {}
        for _, _, cat, dur in segments:
            cat_durations[cat] = cat_durations.get(cat, 0) + dur
        dominant = max(cat_durations, key=cat_durations.get)
        dominant_pct = cat_durations[dominant] / total_length if total_length > 0 else 1.0

        # Check for huge change: opposite extremes (slow vs fast) and substantial
        has_huge_change = False
        for _, _, cat, dur in segments:
            if cat_distance(cat, dominant) == 2 and (dur / total_length) >= huge_change_min_pct:
                has_huge_change = True
                break

        if not has_huge_change:
            # Single label for entire song
            rows.append({
                "piece_index": piece_idx,
                "canonical_composer": meta["canonical_composer"],
                "canonical_title": meta["canonical_title"],
                "year": meta["year"],
                "split": meta["split"],
                "midi_filename": meta["midi_filename"],
                "total_length_sec": total_length,
                "segment_index": 1,
                "start_sec": 0.0,
                "end_sec": total_length,
                "tempo_label": dominant,
                "tempo_category_num": THREE_CATEGORY_TO_NUM[dominant],
            })
        else:
            # Multiple segments with different labels
            for seg_idx, (s_start, s_end, s_cat, _) in enumerate(segments, 1):
                rows.append({
                    "piece_index": piece_idx,
                    "canonical_composer": meta["canonical_composer"],
                    "canonical_title": meta["canonical_title"],
                    "year": meta["year"],
                    "split": meta["split"],
                    "midi_filename": meta["midi_filename"],
                    "total_length_sec": total_length,
                    "segment_index": seg_idx,
                    "start_sec": s_start,
                    "end_sec": s_end,
                    "tempo_label": s_cat,
                    "tempo_category_num": THREE_CATEGORY_TO_NUM[s_cat],
                })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relabel tempo sections using dataset distribution quantiles."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="tempo_sections.csv",
        help="Input CSV (default: tempo_sections.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV (default: overwrite input)",
    )
    parser.add_argument(
        "--quantiles",
        type=str,
        default="25,50,75",
        help="Quantile boundaries for 4 categories, e.g. 25,50,75 (default: quartiles)",
    )
    parser.add_argument(
        "--song-level",
        action="store_true",
        help="Combine to 3 categories (slow/medium/fast), assign per-song labels, "
             "split only on huge tempo changes",
    )
    parser.add_argument(
        "--huge-change-pct",
        type=float,
        default=0.25,
        help="Min fraction of song in opposite tempo to trigger split (default: 0.25)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    if args.song_level:
        # Song-level mode: input must already have speed_category
        if "speed_category" not in df.columns:
            print("Error: --song-level requires input with speed_category column. "
                  "Run without --song-level first.", file=sys.stderr)
            sys.exit(1)
        out_df = compute_song_level_labels(df, huge_change_min_pct=args.huge_change_pct)
        out_path = args.output or "tempo_songs.csv"
        out_df.to_csv(out_path, index=False)
        n_songs = out_df["piece_index"].nunique()
        n_segments = len(out_df)
        n_split = n_segments - n_songs
        counts = out_df["tempo_label"].value_counts().sort_index()
        print(f"Song-level labels (n={n_songs} songs, {n_segments} rows, {n_split} songs split):")
        print("  By segment:")
        for cat in THREE_CATEGORIES:
            n = counts.get(cat, 0)
            pct = 100 * n / len(out_df)
            print(f"    {cat}: {n} ({pct:.1f}%)")
        # Per-song: dominant label by total duration across segments
        def dominant_label(grp: pd.DataFrame) -> str:
            grp = grp.copy()
            grp["dur"] = grp["end_sec"] - grp["start_sec"]
            dur_by_cat = grp.groupby("tempo_label")["dur"].sum()
            return dur_by_cat.idxmax()

        song_labels = out_df.groupby("piece_index").apply(dominant_label)
        song_counts = song_labels.value_counts().sort_index()
        print("  By song (dominant label):")
        for cat in THREE_CATEGORIES:
            n = song_counts.get(cat, 0)
            pct = 100 * n / n_songs
            print(f"    {cat}: {n} ({pct:.1f}%)")
        print(f"\nWrote to {out_path}")
        return

    # Section-level mode: distribution-based quantile labeling
    df["_score"] = df.apply(compute_speed_score, axis=1)
    quantile_vals = [int(x.strip()) for x in args.quantiles.split(",")]
    if len(quantile_vals) != 3:
        print("Error: --quantiles must have exactly 3 values for 4 categories", file=sys.stderr)
        sys.exit(1)
    q25, q50, q75 = np.percentile(df["_score"], quantile_vals)

    print(f"Dataset distribution quantiles (n={len(df)}):")
    print(f"  25th: {q25:.4f}  50th: {q50:.4f}  75th: {q75:.4f}")

    categories = []
    category_nums = []
    for score in df["_score"]:
        cat, num = assign_category_from_quantiles(score, q25, q50, q75)
        categories.append(cat)
        category_nums.append(num)

    df["speed_category"] = categories
    df["speed_category_num"] = category_nums
    df = df.drop(columns=["_score"])

    out_path = args.output or args.input
    df.to_csv(out_path, index=False)

    counts = df["speed_category"].value_counts().sort_index()
    print(f"\nLabel distribution:")
    for cat in SPEED_CATEGORIES:
        n = counts.get(cat, 0)
        pct = 100 * n / len(df)
        print(f"  {cat}: {n} ({pct:.1f}%)")
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
