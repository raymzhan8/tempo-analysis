#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IOI_CANDIDATE_KEYS = [
    "ioi",
    "iois",
    "ioi_sec",
    "ioi_seconds",
    "inter_onset_intervals",
    "inter_onset_intervals_sec",
]

NOTE_LENGTH_CANDIDATE_KEYS = [
    "note_length",
    "note_lengths",
    "note_length_sec",
    "note_length_seconds",
    "durations",
    "duration_sec",
]

MIN_IOI_SEC = 0.08
IOI_HIST_BIN_SEC = 0.01
IOI_HIST_MAX_SEC = 2.5
NOTE_LEN_HIST_BIN_SEC = 0.01
NOTE_LEN_HIST_MAX_SEC = 2.5


def load_json(path: Path) -> dict | list:
    with open(path, "r") as f:
        return json.load(f)


def find_ioi_array(song: dict) -> np.ndarray | None:
    for key in IOI_CANDIDATE_KEYS:
        if key not in song:
            continue
        val = song[key]
        if val is None:
            continue
        try:
            arr = np.asarray(val, dtype=float)
            if arr.size > 0:
                return arr
        except (TypeError, ValueError):
            continue
    hist = song.get("ioi_histogram")
    if isinstance(hist, dict):
        return extract_array_from_histogram(hist, min_val=MIN_IOI_SEC)
    return None


def find_note_length_array(song: dict) -> np.ndarray | None:
    for key in NOTE_LENGTH_CANDIDATE_KEYS:
        if key not in song:
            continue
        val = song[key]
        if val is None:
            continue
        try:
            arr = np.asarray(val, dtype=float)
            if arr.size > 0:
                return arr
        except (TypeError, ValueError):
            continue
    hist = song.get("note_length_histogram")
    if isinstance(hist, dict):
        return extract_array_from_histogram(hist, min_val=0.0)
    return None


def extract_array_from_histogram(hist: dict, min_val: float = 0.0) -> np.ndarray:
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
        if center > min_val and c > 0:
            centers.extend([center] * int(c))
    return np.array(centers, dtype=float) if centers else np.array([])


def clean_numeric_array(arr: np.ndarray, min_val: float | None = None) -> np.ndarray:
    mask = np.isfinite(arr) & (arr > 0)
    if min_val is not None:
        mask = mask & (arr >= min_val)
    return arr[mask]


def build_distribution_histogram(values: np.ndarray, bin_sec: float, max_sec: float) -> dict:
    if len(values) == 0:
        return {"bin_sec": bin_sec, "max_sec": max_sec, "counts": [], "n": 0}
    n_bins = int(max_sec / bin_sec)
    counts = [0] * n_bins
    for v in values:
        if 0 <= v < max_sec:
            idx = min(int(v / bin_sec), n_bins - 1)
            counts[idx] += 1
    return {"bin_sec": bin_sec, "max_sec": max_sec, "counts": counts, "n": int(len(values))}


def to_list_of_dicts(data: dict | list) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "songs" in data:
        return data["songs"]
    if isinstance(data, dict):
        return list(data.values())
    return []


def extract_tempo_entries(data: dict | list) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "songs" in data:
        return data["songs"]
    if isinstance(data, dict):
        return list(data.values())
    return []


def get_song_id_from_tempo(entry: dict) -> str | None:
    for key in ("song_id", "id", "midi_filename", "filename"):
        if key in entry and entry[key] is not None:
            return str(entry[key])
    return None


def get_song_id_from_dist(entry: dict, index: int) -> str | None:
    for key in ("song_id", "id", "midi_filename", "filename"):
        if key in entry and entry[key] is not None:
            return str(entry[key])
    if "index" in entry:
        return str(entry["index"])
    return str(index)


def match_song_records(
    tempo_entries: list[dict],
    dist_entries: list[dict],
) -> tuple[list[dict], list[str], list[str]]:
    tempo_by_id = {}
    for e in tempo_entries:
        sid = get_song_id_from_tempo(e)
        if sid:
            tempo_by_id[sid] = e

    matched = []
    unmatched_tempo = []
    unmatched_dist = []

    for idx, dist in enumerate(dist_entries):
        dist_id = get_song_id_from_dist(dist, idx)
        if not dist_id:
            unmatched_dist.append(f"index {idx}")
            continue
        tempo = tempo_by_id.get(dist_id)
        if tempo is None:
            unmatched_dist.append(dist_id)
            continue
        matched.append({"tempo": tempo, "dist": dist, "song_id": dist_id})
        del tempo_by_id[dist_id]

    unmatched_tempo = [sid for sid in tempo_by_id]

    return matched, unmatched_tempo, unmatched_dist


def aggregate_by_label(matched: list[dict]) -> dict:
    by_label = {"slow": {"ioi": [], "note_length": [], "tempo_bpm": []},
                "moderate": {"ioi": [], "note_length": [], "tempo_bpm": []},
                "fast": {"ioi": [], "note_length": [], "tempo_bpm": []}}

    for m in matched:
        tempo = m["tempo"]
        dist = m["dist"]
        label = tempo.get("tempo_label")
        if label not in by_label:
            continue
        if tempo.get("skipped"):
            continue

        iois = find_ioi_array(dist)
        if iois is not None:
            iois = clean_numeric_array(iois, min_val=MIN_IOI_SEC)
            by_label[label]["ioi"].extend(iois.tolist())

        note_lengths = find_note_length_array(dist)
        if note_lengths is not None:
            note_lengths = clean_numeric_array(note_lengths)
            by_label[label]["note_length"].extend(note_lengths.tolist())

        bpm = tempo.get("tempo_bpm")
        if bpm is not None and np.isfinite(bpm):
            by_label[label]["tempo_bpm"].append(float(bpm))

    return by_label


def save_outputs(
    output_dir: Path,
    dataset_summary: dict,
    tempo_class_csv_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_tempo_analysis.json"
    csv_path = output_dir / "tempo_class_comparison.csv"

    with open(json_path, "w") as f:
        json.dump(dataset_summary, f, indent=2)

    df = pd.DataFrame(tempo_class_csv_rows)
    df.to_csv(csv_path, index=False)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent

    tempo_candidates = [
        script_dir / "tempo_labels.json",
        parent_dir / "public_html" / "tempo_labels.json",
    ]
    dist_candidates = [
        parent_dir / "public_html" / "per_song_distributions.json",
        parent_dir / "public_html" / "per_song_distribution.json",
        script_dir / "per_song_distributions.json",
        script_dir / "per_song_distribution.json",
    ]
    output_candidates = [
        parent_dir / "public_html",
        script_dir,
    ]

    tempo_path = next((p for p in tempo_candidates if p.exists()), None)
    dist_path = next((p for p in dist_candidates if p.exists()), None)
    output_dir = next((p for p in output_candidates if p.exists()), script_dir)

    if tempo_path is None:
        print("Error: tempo_labels.json not found.", file=sys.stderr)
        return 1
    if dist_path is None:
        print("Error: per_song_distribution(s).json not found.", file=sys.stderr)
        return 1

    try:
        tempo_data = load_json(tempo_path)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading {tempo_path}: {e}", file=sys.stderr)
        return 1

    try:
        dist_data = load_json(dist_path)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading {dist_path}: {e}", file=sys.stderr)
        return 1

    tempo_entries = extract_tempo_entries(tempo_data)
    dist_entries = to_list_of_dicts(dist_data)

    matched, unmatched_tempo, unmatched_dist = match_song_records(tempo_entries, dist_entries)

    if unmatched_tempo:
        print(f"Tempo entries without matching distribution: {len(unmatched_tempo)}")
    if unmatched_dist:
        print(f"Distribution entries without matching tempo label: {len(unmatched_dist)}")

    by_label = aggregate_by_label(matched)

    tempo_class_csv_rows = []
    for m in matched:
        tempo = m["tempo"]
        dist = m["dist"]
        label = tempo.get("tempo_label")
        if tempo.get("skipped") or label not in ("slow", "moderate", "fast"):
            continue

        iois = find_ioi_array(dist)
        note_lengths = find_note_length_array(dist)
        if iois is not None:
            iois = clean_numeric_array(iois, min_val=MIN_IOI_SEC)
        else:
            iois = np.array([])
        if note_lengths is not None:
            note_lengths = clean_numeric_array(note_lengths)
        else:
            note_lengths = np.array([])

        row = {
            "song_id": m["song_id"],
            "tempo_bpm": tempo.get("tempo_bpm"),
            "tempo_label": label,
            "n_ioi": len(iois),
            "n_note_length": len(note_lengths),
            "ioi_p50": round(float(np.percentile(iois, 50)), 4) if len(iois) > 0 else None,
            "ioi_p80": round(float(np.percentile(iois, 80)), 4) if len(iois) > 0 else None,
            "ioi_p90": round(float(np.percentile(iois, 90)), 4) if len(iois) > 0 else None,
            "note_length_p50": round(float(np.percentile(note_lengths, 50)), 4) if len(note_lengths) > 0 else None,
            "note_length_p80": round(float(np.percentile(note_lengths, 80)), 4) if len(note_lengths) > 0 else None,
            "note_length_p90": round(float(np.percentile(note_lengths, 90)), 4) if len(note_lengths) > 0 else None,
        }
        tempo_class_csv_rows.append(row)

    label_counts = {"slow": 0, "moderate": 0, "fast": 0}
    for m in matched:
        label = m["tempo"].get("tempo_label")
        if label in label_counts and not m["tempo"].get("skipped"):
            label_counts[label] += 1

    per_label_tempo = {}
    per_label_ioi_dist = {}
    per_label_note_length_dist = {}

    for label in ("slow", "moderate", "fast"):
        iois = np.array(by_label[label]["ioi"], dtype=float)
        note_lengths = np.array(by_label[label]["note_length"], dtype=float)
        bpm = np.array(by_label[label]["tempo_bpm"], dtype=float)

        per_label_tempo[label] = {
            "n_songs": int(len(bpm)),
            "mean_bpm": round(float(np.mean(bpm)), 2) if len(bpm) > 0 else None,
            "median_bpm": round(float(np.median(bpm)), 2) if len(bpm) > 0 else None,
            "min_bpm": round(float(np.min(bpm)), 2) if len(bpm) > 0 else None,
            "max_bpm": round(float(np.max(bpm)), 2) if len(bpm) > 0 else None,
        }
        per_label_ioi_dist[label] = build_distribution_histogram(iois, IOI_HIST_BIN_SEC, IOI_HIST_MAX_SEC)
        per_label_note_length_dist[label] = build_distribution_histogram(
            note_lengths, NOTE_LEN_HIST_BIN_SEC, NOTE_LEN_HIST_MAX_SEC
        )

    overall_tempo_bpm = []
    for m in matched:
        if not m["tempo"].get("skipped"):
            b = m["tempo"].get("tempo_bpm")
            if b is not None and np.isfinite(b):
                overall_tempo_bpm.append(float(b))
    overall_tempo_bpm = np.array(overall_tempo_bpm)

    dataset_summary = {
        "overall": {
            "n_songs_matched": len(matched),
            "n_songs_labeled": sum(1 for m in matched if not m["tempo"].get("skipped")),
            "n_unmatched_tempo": len(unmatched_tempo),
            "n_unmatched_dist": len(unmatched_dist),
            "tempo_bpm": {
                "mean": round(float(np.mean(overall_tempo_bpm)), 2) if len(overall_tempo_bpm) > 0 else None,
                "median": round(float(np.median(overall_tempo_bpm)), 2) if len(overall_tempo_bpm) > 0 else None,
                "min": round(float(np.min(overall_tempo_bpm)), 2) if len(overall_tempo_bpm) > 0 else None,
                "max": round(float(np.max(overall_tempo_bpm)), 2) if len(overall_tempo_bpm) > 0 else None,
            },
        },
        "tempo_label_counts": label_counts,
        "per_label_tempo": per_label_tempo,
        "per_label_ioi_distributions": per_label_ioi_dist,
        "per_label_note_length_distributions": per_label_note_length_dist,
    }

    save_outputs(output_dir, dataset_summary, tempo_class_csv_rows)

    print(f"Matched songs: {len(matched)}")
    print(f"Label counts: {label_counts}")
    print(f"Output: {output_dir / 'dataset_tempo_analysis.json'}")
    print(f"        {output_dir / 'tempo_class_comparison.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
