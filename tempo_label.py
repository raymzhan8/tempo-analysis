#!/usr/bin/env python3
"""
Label each song in per_song_distribution(s).json as slow, moderate, or fast
based on beat-level tempo estimated from filtered IOI percentiles.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

IOI_CANDIDATE_KEYS = [
    "ioi",
    "iois",
    "ioi_sec",
    "ioi_seconds",
    "inter_onset_intervals",
    "inter_onset_intervals_sec",
]

MIN_IOI_SEC = 0.08
MIN_VALID_IOIS = 10


def locate_ioi_array(song: dict) -> tuple[str | None, np.ndarray | None]:
    """
    Locate the IOI array for a song by checking candidate keys.
    Returns (key_used, array) or (None, None) if not found.
    """
    for key in IOI_CANDIDATE_KEYS:
        if key not in song:
            continue
        val = song[key]
        if val is None:
            continue
        try:
            arr = np.asarray(val, dtype=float)
            if arr.size > 0:
                return key, arr
        except (TypeError, ValueError):
            continue
    return None, None


def extract_iois_from_histogram(hist: dict) -> np.ndarray:
    """
    Reconstruct approximate IOI values from a histogram.
    Each bin contributes bin_center values repeated by its count.
    """
    bin_sec = float(hist.get("bin_sec", 0.01))
    max_sec = float(hist.get("max_sec", 2.5))
    counts = hist.get("counts", [])
    if not counts:
        return np.array([])

    centers = []
    for i, c in enumerate(counts):
        center = (i + 0.5) * bin_sec
        if center >= max_sec:
            break
        if center >= MIN_IOI_SEC and c > 0:
            centers.extend([center] * int(c))
    return np.array(centers, dtype=float) if centers else np.array([])


def get_song_id(song: dict, index: int) -> str:
    """Derive a unique song identifier from the song dict."""
    for key in ("song_id", "id", "midi_filename", "filename", "index"):
        if key in song and song[key] is not None:
            return str(song[key])
    return str(index)


def filter_iois(arr: np.ndarray) -> np.ndarray:
    """Keep only finite numeric values >= MIN_IOI_SEC."""
    mask = np.isfinite(arr) & (arr >= MIN_IOI_SEC)
    return arr[mask]


def process_song(song: dict, index: int) -> dict:
    """
    Process a single song and return a dict with all required fields.
    """
    song_id = get_song_id(song, index)
    result = {
        "song_id": song_id,
        "n_ioi_original": None,
        "n_ioi_filtered": None,
        "p50_ioi": None,
        "p80_ioi": None,
        "p90_ioi": None,
        "tempo_bpm": None,
        "tempo_label": None,
        "skipped": False,
        "skip_reason": None,
    }

    iois = None

    key, arr = locate_ioi_array(song)
    if key is not None and arr is not None:
        result["n_ioi_original"] = len(arr)
        iois = filter_iois(arr)

    if iois is None or len(iois) == 0:
        hist = song.get("ioi_histogram")
        if isinstance(hist, dict):
            iois = extract_iois_from_histogram(hist)
            result["n_ioi_original"] = int(sum(hist.get("counts", [])))

    if iois is None or len(iois) == 0:
        result["skipped"] = True
        result["skip_reason"] = "no_ioi_data"
        return result

    result["n_ioi_filtered"] = len(iois)

    if len(iois) < MIN_VALID_IOIS:
        result["skipped"] = True
        result["skip_reason"] = f"fewer_than_{MIN_VALID_IOIS}_valid_iois"
        return result

    p50 = float(np.percentile(iois, 50))
    p80 = float(np.percentile(iois, 80))
    p90 = float(np.percentile(iois, 90))

    result["p50_ioi"] = round(p50, 4)
    result["p80_ioi"] = round(p80, 4)
    result["p90_ioi"] = round(p90, 4)
    result["tempo_bpm"] = round(60.0 / p90, 2)
    result["skipped"] = False
    result["skip_reason"] = None

    return result


def load_songs(path: Path) -> list[tuple[int, dict]]:
    """
    Load songs from JSON. Supports:
    - List of per-song dicts
    - Dict mapping song ids to per-song dicts
    - Dict with 'songs' key containing a list
    """
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return list(enumerate(data))
    if isinstance(data, dict):
        if "songs" in data:
            return list(enumerate(data["songs"]))
        return list(enumerate(data.values()))
    raise ValueError(f"Unexpected JSON structure: {type(data)}")


def compute_tercile_thresholds(tempo_bpm_values: list[float]) -> tuple[float, float]:
    """
    Compute slow/moderate/fast thresholds as dataset terciles.
    Returns (low_threshold, high_threshold) such that:
    - slow: tempo_bpm < low_threshold
    - moderate: low_threshold <= tempo_bpm < high_threshold
    - fast: tempo_bpm >= high_threshold
    """
    if not tempo_bpm_values:
        return 0.0, 0.0
    arr = np.array(tempo_bpm_values)
    low = float(np.percentile(arr, 33.333))
    high = float(np.percentile(arr, 66.667))
    return low, high


def assign_tempo_label(tempo_bpm: float, low: float, high: float) -> str:
    """Assign slow, moderate, or fast based on tercile thresholds."""
    if tempo_bpm < low:
        return "slow"
    if tempo_bpm < high:
        return "moderate"
    return "fast"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Label songs as slow, moderate, or fast based on IOI percentiles."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to per_song_distribution(s).json (default: auto-detect)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory for tempo_labels.json and tempo_labels.csv",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent

    if args.input:
        input_path = Path(args.input)
    else:
        candidates = [
            parent_dir / "public_html" / "per_song_distributions.json",
            parent_dir / "public_html" / "per_song_distribution.json",
            script_dir / "per_song_distributions.json",
            script_dir / "per_song_distribution.json",
        ]
        input_path = None
        for p in candidates:
            if p.exists():
                input_path = p
                break
        if input_path is None:
            print("Error: No input file found. Specify path or place per_song_distribution(s).json in public_html/", file=sys.stderr)
            return 1

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else script_dir

    try:
        songs_with_index = load_songs(input_path)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_path}: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading {input_path}: {e}", file=sys.stderr)
        return 1

    results = []
    for idx, song in tqdm(songs_with_index, desc="Labeling tempo", unit="song"):
        try:
            r = process_song(song, idx)
            results.append(r)
        except Exception as e:
            song_id = get_song_id(song, idx)
            results.append({
                "song_id": song_id,
                "n_ioi_original": None,
                "n_ioi_filtered": None,
                "p50_ioi": None,
                "p80_ioi": None,
                "p90_ioi": None,
                "tempo_bpm": None,
                "tempo_label": None,
                "skipped": True,
                "skip_reason": f"error: {e}",
            })

    tempo_bpm_values = [r["tempo_bpm"] for r in results if r["tempo_bpm"] is not None]
    low_thresh, high_thresh = compute_tercile_thresholds(tempo_bpm_values)

    for r in results:
        if r["tempo_bpm"] is not None:
            r["tempo_label"] = assign_tempo_label(r["tempo_bpm"], low_thresh, high_thresh)

    output_json = output_dir / "tempo_labels.json"
    output_csv = output_dir / "tempo_labels.csv"

    try:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
    except OSError as e:
        print(f"Error writing {output_json}: {e}", file=sys.stderr)
        return 1

    try:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
    except OSError as e:
        print(f"Error writing {output_csv}: {e}", file=sys.stderr)
        return 1

    n_total = len(results)
    n_skipped = sum(1 for r in results if r["skipped"])
    n_labeled = n_total - n_skipped

    print("Summary")
    print("-------")
    print(f"Input: {input_path}")
    print(f"Total songs: {n_total}")
    print(f"Labeled: {n_labeled}")
    print(f"Skipped: {n_skipped}")
    print()
    print("Tempo thresholds (terciles)")
    print("----------------------------")
    print(f"Slow:    tempo_bpm < {low_thresh:.2f}")
    print(f"Moderate: {low_thresh:.2f} <= tempo_bpm < {high_thresh:.2f}")
    print(f"Fast:    tempo_bpm >= {high_thresh:.2f}")
    print()
    if tempo_bpm_values:
        print("Tempo BPM statistics")
        print("--------------------")
        arr = np.array(tempo_bpm_values)
        print(f"Min:    {float(np.min(arr)):.2f}")
        print(f"P33:    {float(np.percentile(arr, 33.333)):.2f}")
        print(f"Median: {float(np.median(arr)):.2f}")
        print(f"P67:    {float(np.percentile(arr, 66.667)):.2f}")
        print(f"Max:    {float(np.max(arr)):.2f}")
        print(f"Mean:   {float(np.mean(arr)):.2f}")
    print()
    print(f"Output: {output_json}")
    print(f"        {output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
