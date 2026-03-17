#!/usr/bin/env python3
"""
Read the MAESTRO dataset and extract summary features for the interactive report.
Outputs JSON with aggregated stats and per-recording data for exploration.

Also extracts per-song IOI (inter-onset interval) and note length distributions
from MIDI files for dataset understanding.
"""

import argparse
import json
import statistics
from pathlib import Path

import mido
import pandas as pd
from tqdm import tqdm

# Path to MAESTRO dataset (relative to this script)
MAESTRO_CSV = Path(__file__).resolve().parent / "maestro-v3.0.0" / "maestro-v3.0.0.csv"
MAESTRO_SPLITS_CSV = Path(__file__).resolve().parent / "maestro_splits.csv"
MAESTRO_DIR = Path(__file__).resolve().parent / "maestro-v3.0.0"
OUTPUT_JSON = Path(__file__).resolve().parent.parent / "public_html" / "maestro_data.json"
OUTPUT_DISTRIBUTIONS_JSON = Path(__file__).resolve().parent.parent / "public_html" / "per_song_distributions.json"

# Distribution extraction params (10ms bins, same as AMT)
ONSET_QUANTIZE_SEC = 0.05
IOI_BIN_SEC = 0.01
DURATION_BIN_SEC = 0.01
MAX_IOI_SEC = 2.5
MAX_DURATION_SEC = 2.5
# 60/IOI (events per minute, BPM-like). Filter IOI < this to avoid huge values
MIN_IOI_FOR_INV = 0.01
INV_IOI_BIN = 5.0
MAX_INV_IOI = 1200.0
# Tempo-over-time: window sizes (60s/30s step, 30s/15s step)
TEMPO_WINDOW_SEC = 60.0
TEMPO_WINDOW_SEC_30 = 30.0


def get_all_notes(midi: mido.MidiFile) -> list[dict]:
    """Extract all notes with start_sec and end_sec from MIDI."""
    current_time = 0.0
    active = {}
    all_notes = []
    for msg in midi:
        current_time += msg.time
        if msg.type == "note_on":
            key = (msg.note, msg.channel)
            if msg.velocity > 0:
                active[key] = current_time
            else:
                if key in active:
                    all_notes.append({"start_sec": active[key], "end_sec": current_time})
                    del active[key]
    return all_notes


def get_chord_onset_times(notes: list[dict], quantize_sec: float = ONSET_QUANTIZE_SEC) -> list[float]:
    """Return chord onset times; notes within quantize_sec are grouped as one chord."""
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: n["start_sec"])
    onset_times = [sorted_notes[0]["start_sec"]]
    for n in sorted_notes[1:]:
        if n["start_sec"] - onset_times[-1] > quantize_sec:
            onset_times.append(n["start_sec"])
    return onset_times


def build_histogram(values: list[float], bin_sec: float, max_sec: float) -> dict:
    """Build histogram with given bin size. Returns counts (bin_edges reconstructible from bin_sec, max_sec)."""
    n_bins = int(max_sec / bin_sec)
    counts = [0] * n_bins
    for v in values:
        if 0 <= v < max_sec:
            idx = min(int(v / bin_sec), n_bins - 1)
            counts[idx] += 1
    return {"bin_sec": bin_sec, "max_sec": max_sec, "counts": counts}


def compute_tempo_over_time(
    onset_times: list[float],
    window_sec: float = TEMPO_WINDOW_SEC,
) -> dict:
    """
    Compute tempo(t) = 60 / median(IOI_window) for sliding windows.
    IOI between onset i and i+1 is assigned to window by its midpoint.
    Returns {window_sec, times, tempo} where times are window centers.
    """
    if len(onset_times) < 2:
        return {"window_sec": window_sec, "times": [], "tempo": []}
    iois = [onset_times[i + 1] - onset_times[i] for i in range(len(onset_times) - 1)]
    midpoints = [(onset_times[i] + onset_times[i + 1]) / 2 for i in range(len(onset_times) - 1)]
    duration = onset_times[-1] - onset_times[0]
    if duration <= 0:
        return {"window_sec": window_sec, "times": [], "tempo": []}

    times = []
    tempo = []
    tempo_p80 = []
    tempo_p90 = []
    t = onset_times[0] + window_sec / 2
    while t < onset_times[-1] - window_sec / 2:
        window_start = t - window_sec / 2
        window_end = t + window_sec / 2
        window_iois = [
            iois[i] for i in range(len(iois))
            if window_start <= midpoints[i] < window_end and iois[i] >= MIN_IOI_FOR_INV
        ]
        if window_iois:
            med = statistics.median(window_iois)
            tempo.append(round(60 / med, 2))
            if len(window_iois) >= 2:
                q = statistics.quantiles(window_iois, n=10)
                p80 = q[7] if len(q) >= 8 else med
                p90 = q[8] if len(q) >= 9 else med
            else:
                p80 = med
                p90 = med
            tempo_p80.append(round(60 / p80, 2))
            tempo_p90.append(round(60 / p90, 2))
        else:
            tempo.append(None)
            tempo_p80.append(None)
            tempo_p90.append(None)
        times.append(round(t, 2))
        t += window_sec / 2
    return {"window_sec": window_sec, "times": times, "tempo": tempo, "tempo_p80": tempo_p80, "tempo_p90": tempo_p90}


def extract_song_distributions(
    midi_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> dict | None:
    """
    Extract IOI and note length distributions for a MIDI file or segment.
    When start_sec/end_sec are provided, filter notes to that time range.
    Returns dict with ioi_histogram, note_length_histogram, and summary stats.
    """
    if not midi_path.exists():
        return None
    try:
        midi = mido.MidiFile(midi_path)
    except Exception:
        return None
    notes = get_all_notes(midi)
    if not notes:
        return None

    # Clip to segment when start_sec/end_sec provided
    if start_sec is not None and end_sec is not None:
        notes = [n for n in notes if start_sec <= n["start_sec"] < end_sec]
    if len(notes) < 2:
        return None

    # Note lengths (duration of each note)
    note_lengths = [n["end_sec"] - n["start_sec"] for n in notes]

    # IOIs: inter-onset intervals between consecutive chord onsets
    onset_times = get_chord_onset_times(notes)
    iois = [onset_times[i + 1] - onset_times[i] for i in range(len(onset_times) - 1)]

    if not iois:
        iois = [0.0]  # avoid empty stats

    # 60/IOI (events per minute, BPM-like). Filter very small IOIs to avoid infinity
    inv_iois = [60 / x for x in iois if x >= MIN_IOI_FOR_INV]
    if not inv_iois:
        inv_iois = [0.0]

    tempo_over_time = compute_tempo_over_time(onset_times, window_sec=TEMPO_WINDOW_SEC)
    tempo_over_time_30 = compute_tempo_over_time(onset_times, window_sec=TEMPO_WINDOW_SEC_30)

    return {
        "num_notes": len(notes),
        "num_onsets": len(onset_times),
        "num_iois": len(iois),
        "ioi_histogram": build_histogram(iois, IOI_BIN_SEC, MAX_IOI_SEC),
        "inverse_ioi_histogram": build_histogram(inv_iois, INV_IOI_BIN, MAX_INV_IOI),
        "note_length_histogram": build_histogram(note_lengths, DURATION_BIN_SEC, MAX_DURATION_SEC),
        "ioi_stats": {
            "mean": round(statistics.mean(iois), 4),
            "median": round(statistics.median(iois), 4),
            "p90": round(statistics.quantiles(iois, n=10)[8], 4) if len(iois) >= 2 else round(statistics.median(iois), 4),
            "min": round(min(iois), 4),
            "max": round(max(iois), 4),
        },
        "inverse_ioi_stats": {
            "mean": round(statistics.mean(inv_iois), 4),
            "median": round(statistics.median(inv_iois), 4),
            "min": round(min(inv_iois), 4),
            "max": round(max(inv_iois), 4),
        },
        "note_length_stats": {
            "mean": round(statistics.mean(note_lengths), 4),
            "median": round(statistics.median(note_lengths), 4),
            "p90": round(statistics.quantiles(note_lengths, n=10)[8], 4) if len(note_lengths) >= 2 else round(statistics.median(note_lengths), 4),
            "min": round(min(note_lengths), 4),
            "max": round(max(note_lengths), 4),
        },
        "tempo_over_time": tempo_over_time,
        "tempo_over_time_30": tempo_over_time_30,
    }


def extract_per_song_distributions(
    df: pd.DataFrame,
    max_songs: int = 10,
    use_splits: bool = False,
) -> list[dict]:
    """Extract IOI and note length distributions for the first max_songs files. Use max_songs=0 for all."""
    n = len(df) if max_songs <= 0 else min(max_songs, len(df))
    has_segments = use_splits and "start_sec" in df.columns and "end_sec" in df.columns
    results = []
    for idx in tqdm(range(n), desc="Extracting distributions", unit="song"):
        row = df.iloc[idx]
        midi_path = MAESTRO_DIR / row["midi_filename"]
        start_sec = float(row["start_sec"]) if has_segments and pd.notna(row.get("start_sec")) else None
        end_sec = float(row["end_sec"]) if has_segments and pd.notna(row.get("end_sec")) else None
        dist = extract_song_distributions(midi_path, start_sec=start_sec, end_sec=end_sec)
        midi_filename = str(row["midi_filename"])
        if has_segments and start_sec is not None and end_sec is not None:
            song_id = f"{midi_filename}|{start_sec}|{end_sec}"
        else:
            song_id = midi_filename
        if dist is None:
            results.append({
                "index": idx,
                "song_id": song_id,
                "composer": str(row["canonical_composer"]),
                "title": str(row["canonical_title"]),
                "midi_filename": midi_filename,
                "split": str(row["split"]) if has_segments else None,
                "year": int(row["year"]) if has_segments and pd.notna(row.get("year")) else None,
                "duration_sec": round(float(row["segment_duration"]), 2) if has_segments else None,
                "audio_filename": str(row["audio_filename"]) if has_segments else None,
                "error": "Failed to load or empty MIDI",
            })
            continue
        entry = {
            "index": idx,
            "song_id": song_id,
            "composer": str(row["canonical_composer"]),
            "title": str(row["canonical_title"]),
            "midi_filename": midi_filename,
            **dist,
        }
        if has_segments:
            entry["split"] = str(row["split"])
            entry["year"] = int(row["year"]) if pd.notna(row.get("year")) else None
            entry["duration_sec"] = round(float(row["segment_duration"]), 2)
            entry["audio_filename"] = str(row["audio_filename"])
        results.append(entry)
    return results


def load_maestro(csv_path: Path | None = None) -> pd.DataFrame:
    """Load and validate the MAESTRO dataset."""
    path = csv_path or MAESTRO_CSV
    df = pd.read_csv(path)
    # Ensure duration is numeric
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df.dropna(subset=["duration"])
    # When using splits, validate start_sec/end_sec and compute segment_duration
    if "start_sec" in df.columns and "end_sec" in df.columns:
        df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
        df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
        df["segment_duration"] = df["end_sec"] - df["start_sec"]
    return df


def extract_features(df: pd.DataFrame, use_splits: bool = False) -> dict:
    """Extract summary features and per-recording data from MAESTRO."""
    # Use segment_duration when splits (segment-level); else full file duration
    dur_col = "segment_duration" if (use_splits and "segment_duration" in df.columns) else "duration"
    duration_vals = df[dur_col]

    # Per-recording rows for the table (lightweight)
    has_segments = use_splits and "start_sec" in df.columns and "end_sec" in df.columns
    records = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Extracting features", unit="rec")):
        midi_filename = str(row["midi_filename"])
        start_sec = float(row["start_sec"]) if has_segments and pd.notna(row.get("start_sec")) else None
        end_sec = float(row["end_sec"]) if has_segments and pd.notna(row.get("end_sec")) else None
        song_id = f"{midi_filename}|{start_sec}|{end_sec}" if (start_sec is not None and end_sec is not None) else midi_filename
        records.append({
            "index": i,
            "song_id": song_id,
            "composer": str(row["canonical_composer"]),
            "title": str(row["canonical_title"]),
            "split": str(row["split"]),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "duration_sec": round(float(row[dur_col]), 2),
            "midi_filename": midi_filename,
            "audio_filename": str(row["audio_filename"]),
        })

    # Summary stats
    summary = {
        "total_recordings": len(df),
        "total_duration_hours": round(duration_vals.sum() / 3600, 2),
        "duration_sec": {
            "min": round(float(duration_vals.min()), 2),
            "max": round(float(duration_vals.max()), 2),
            "mean": round(float(duration_vals.mean()), 2),
            "median": round(float(duration_vals.median()), 2),
        },
        "by_split": df["split"].value_counts().to_dict(),
        "by_year": df["year"].value_counts().sort_index().to_dict(),
        "composer_counts": df["canonical_composer"].value_counts().head(50).to_dict(),
        "unique_composers": int(df["canonical_composer"].nunique()),
        "unique_titles": int(df["canonical_title"].nunique()),
    }

    # Histogram bins for duration distribution
    duration_bins = [0, 60, 120, 180, 300, 600, 900, 1200, 1800, float("inf")]
    duration_labels = ["0-1min", "1-2min", "2-3min", "3-5min", "5-10min", "10-15min", "15-20min", "20-30min", "30min+"]
    df_copy = df.copy()
    df_copy["duration_bin"] = pd.cut(duration_vals, bins=duration_bins, labels=duration_labels)
    summary["duration_distribution"] = df_copy["duration_bin"].value_counts().reindex(duration_labels, fill_value=0).to_dict()

    return {
        "summary": summary,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract MAESTRO features and per-song distributions")
    parser.add_argument("--distributions-only", action="store_true", help="Only extract per-song IOI/note-length distributions")
    parser.add_argument("--max-songs", type=int, default=10, help="Number of songs for distribution extraction (default: 10). Use 0 for all songs.")
    parser.add_argument("--splits", action="store_true", help="Use maestro_splits.csv (segment-level) instead of maestro-v3.0.0.csv")
    args = parser.parse_args()

    csv_path = MAESTRO_SPLITS_CSV if args.splits else MAESTRO_CSV
    if args.splits and not csv_path.exists():
        raise SystemExit(f"Error: {csv_path} not found. Run without --splits to use maestro-v3.0.0.csv")
    df = load_maestro(csv_path=csv_path)

    # Always extract features (maestro_data.json) so Recordings table reflects splits when used
    features = extract_features(df, use_splits=args.splits)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(features, f, indent=2)
    label = "segments" if args.splits else "recordings"
    print(f"Wrote {OUTPUT_JSON} ({len(features['records'])} {label})")

    # Per-song IOI and note length distributions
    distributions = extract_per_song_distributions(df, max_songs=args.max_songs, use_splits=args.splits)
    OUTPUT_DISTRIBUTIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DISTRIBUTIONS_JSON, "w") as f:
        json.dump({"songs": distributions, "params": {"bin_sec": IOI_BIN_SEC, "max_ioi_sec": MAX_IOI_SEC, "max_duration_sec": MAX_DURATION_SEC}}, f, indent=2)
    label = "segments" if args.splits else "songs"
    print(f"Wrote {OUTPUT_DISTRIBUTIONS_JSON} ({len(distributions)} {label})")


if __name__ == "__main__":
    main()
