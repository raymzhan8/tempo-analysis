from pathlib import Path
from typing import Any

import argparse
import math
import statistics
import sys
import pandas as pd
import mido

SPEED_CATEGORY_TO_NUM = {"slow": 0, "medium": 1, "fast": 2, "very_fast": 3}
ONSET_QUANTIZE_SEC = 0.1

# Model-aligned binning (10ms bins, same as AMT)
IOI_BIN_SEC = 0.01
DURATION_BIN_SEC = 0.01
# Configurable max bins for histogram compactness (seconds)
DEFAULT_MAX_IOI_SEC = 5.0
DEFAULT_MAX_DURATION_SEC = 5.0



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


def compute_median_ioi(onset_times: list[float]) -> float | None:
    """Return median inter-onset interval (sec) between consecutive chord onsets."""
    if len(onset_times) < 2:
        return None
    iois = [onset_times[i + 1] - onset_times[i] for i in range(len(onset_times) - 1)]
    return statistics.median(iois)


def get_all_notes(midi: mido.MidiFile) -> list[dict]:
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



def analyze_tempo_by_metrics(
    midi: mido.MidiFile,
    window_sec: float,
    *,
    thresholds: tuple[float, float, float] | None = None,
) -> dict:

    all_notes = get_all_notes(midi)
    total_length = midi.length
    if total_length <= 0 or window_sec <= 0:
        return {
            "total_length_sec": total_length,
            "window_sec": window_sec,
            "sections": [],
            "window": [],
        }

    window = []
    t = 0.0
    while t < total_length:
        t_end = min(t + window_sec, total_length)
        in_w = [n for n in all_notes if t <= n["start_sec"] < t_end]
        # print(num_onsets, t, t_end)
        # onset_times = [n["start_sec"] for n in in_w]
        onset_times = get_chord_onset_times(in_w)
        num_onsets = len(onset_times)
        median_ioi = compute_median_ioi(onset_times)

        if in_w:
            durations = [n["end_sec"] - n["start_sec"] for n in in_w]
            median_len = statistics.median(durations)
        else:
            median_len = None

        onsets_per_sec = num_onsets / (t_end - t)
        med = median_len if median_len is not None else 0.5
        score = onsets_per_sec / (1 + med)

        window.append({
            "start_sec": t,
            "end_sec": t_end,
            "num_onsets": num_onsets,
            "median_note_length_sec": median_len,
            "median_ioi_sec": median_ioi,
            "score": score,
        })
        t = t_end

    for w in window:
        onsets_per_sec = w["num_onsets"] / (w["end_sec"] - w["start_sec"])
        med = w["median_note_length_sec"]
        if med is None:
            med = 0.5
        # can tweak 
        # w["score"] = onsets_per_sec - 0.5 * med
        ioi = w.get("median_ioi_sec")
        if ioi is None:
            ioi = 0
        w["score"] = onsets_per_sec / (1 + (med + ioi)/2)

    # t_low, t_mid, t_high = thresholds if thresholds is not None else (1, 3, 6)
    t_low, t_mid, t_high = thresholds if thresholds is not None else (2.5, 4.0, 6.0)
    # scores = sorted(w["score"] for w in window)
    # n = len(scores)
    # q1 = scores[n // 4] if n >= 4 else scores[0]
    # q2 = scores[n // 2] if n >= 2 else scores[0]
    # q3 = scores[(3 * n) // 4] if n >= 4 else scores[-1]
    for w in window:
        s = w["score"]
        if s <= t_low:
            w["speed_category"] = "slow"
        elif s <= t_mid:
            w["speed_category"] = "medium"
        elif s <= t_high:
            w["speed_category"] = "fast"
        else:
            w["speed_category"] = "very_fast"
        w["speed_category_num"] = SPEED_CATEGORY_TO_NUM[w["speed_category"]]
        del w["score"]


    # sectionalizing to if a big section has a notable tempo change
    sections = []
    for w in window:
        w_num = w["speed_category_num"]
        merge = (
            sections
            and abs(w_num - sections[-1]["speed_category_num"]) < 1
        )
        if merge:
            sec = sections[-1]
            sec["end_sec"] = w["end_sec"]
            sec["num_onsets"] += w["num_onsets"]
            n_prev = sec["num_onsets"] - w["num_onsets"]
            n_cur = w["num_onsets"]
            m_prev = sec["median_note_length_sec"]
            m_cur = w["median_note_length_sec"]
            if n_prev + n_cur > 0 and m_prev is not None and m_cur is not None:
                sec["median_note_length_sec"] = (n_prev * m_prev + n_cur * m_cur) / (n_prev + n_cur)
            elif m_cur is not None:
                sec["median_note_length_sec"] = m_cur
            # Merge median_ioi: weight by num_onsets
            ioi_prev, ioi_cur = sec.get("median_ioi_sec"), w.get("median_ioi_sec")
            if ioi_prev is not None and ioi_cur is not None and n_prev + n_cur > 0:
                sec["median_ioi_sec"] = (n_prev * ioi_prev + n_cur * ioi_cur) / (n_prev + n_cur)
            elif ioi_cur is not None:
                sec["median_ioi_sec"] = ioi_cur
        else:
            sections.append({
                "start_sec": w["start_sec"],
                "end_sec": w["end_sec"],
                "num_onsets": w["num_onsets"],
                "median_note_length_sec": w["median_note_length_sec"],
                "median_ioi_sec": w["median_ioi_sec"],
                "speed_category": w["speed_category"],
                "speed_category_num": w["speed_category_num"],
            })

    for sec in sections:
        dur = sec["end_sec"] - sec["start_sec"]
        sec["onsets_per_sec"] = sec["num_onsets"] / dur if dur > 0 else 0.0
        med = sec.get("median_note_length_sec") or 0.5
        sec["score"] = sec["onsets_per_sec"] / (1 + med)

    return {
        "total_length_sec": total_length,
        "window_sec": window_sec,
        "sections": sections,
        "window": window,
    }


def process_midi_to_rows(midi_path: str, idx: int, row_meta: dict, window_sec: float) -> list:
    midi = mido.MidiFile(midi_path)
    result = analyze_tempo_by_metrics(midi, window_sec=window_sec)
    rows = []
    for sec_idx, sec in enumerate(result["sections"]):
        rows.append({
            "piece_index": idx,
            "canonical_composer": row_meta.get("canonical_composer", ""),
            "canonical_title": row_meta.get("canonical_title", ""),
            "year": row_meta.get("year", ""),
            "split": row_meta.get("split", ""),
            "midi_filename": row_meta.get("midi_filename", Path(midi_path).name),
            "total_length_sec": result["total_length_sec"],
            "section_index": sec_idx + 1,
            "start_sec": sec["start_sec"],
            "end_sec": sec["end_sec"],
            "num_onsets": sec["num_onsets"],
            "onsets_per_sec": sec["onsets_per_sec"],
            "median_note_length_sec": sec["median_note_length_sec"],
            "median_ioi_sec": sec.get("median_ioi_sec"),
            "speed_category": sec["speed_category"],
            "speed_category_num": sec["speed_category_num"],
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze tempo sections in MIDI files.")
    parser.add_argument(
        "index",
        nargs="?",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "-o", "--output",
        default="tempo_sections.csv",
    )
    args = parser.parse_args()

    window_sec = args.window_sec
    rows = []

    if args.index is not None:
        df = pd.read_csv("./maestro-v3.0.0/maestro-v3.0.0.csv")
        n = len(df)
        if args.index < 0 or args.index >= n:
            print(f"Error: index {args.index} out of range [0, {n-1}]", file=sys.stderr)
            sys.exit(1)
        row = df.iloc[args.index]
        midi_path = f"./maestro-v3.0.0/{row['midi_filename']}"
        row_meta = {
            "canonical_composer": row["canonical_composer"],
            "canonical_title": row["canonical_title"],
            "year": row["year"],
            "split": row["split"],
            "midi_filename": row["midi_filename"],
        }
        rows = process_midi_to_rows(midi_path, args.index, row_meta, window_sec)
        n_pieces = 1
        print(f"Processed index {args.index}: {row['midi_filename']}")
    else:
        # Full MAESTRO dataset
        df = pd.read_csv("./maestro-v3.0.0/maestro-v3.0.0.csv")
        n = len(df)
        for idx in range(n):
            row = df.iloc[idx]
            midi_path = f"./maestro-v3.0.0/{row['midi_filename']}"
            row_meta = {
                "canonical_composer": row["canonical_composer"],
                "canonical_title": row["canonical_title"],
                "year": row["year"],
                "split": row["split"],
                "midi_filename": row["midi_filename"],
            }
            rows.extend(process_midi_to_rows(midi_path, idx, row_meta, window_sec))
        n_pieces = n

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} section rows ({n_pieces} pieces) to {args.output}")



# OLD FUNCTION

# def notes_in_interval(
#     midi: mido.MidiFile,
#     start_sec: float,
#     end_sec: float,
# ) -> tuple[float | None, int]:
#     current_time = 0
#     active = {}
#     all_notes = []

#     for msg in midi:
#         current_time += msg.time

#         if msg.type == "note_on":
#             key = (msg.note, msg.channel)
#             if msg.velocity > 0:
#                 active[key] = current_time
#             else:
#                 if key in active:
#                     all_notes.append({
#                         "start_sec": active[key],
#                         "end_sec": current_time,
#                         "note": msg.note,
#                         "channel": msg.channel,
#                     })
#                     del active[key]

#     in_interval = [
#         n for n in all_notes
#         if start_sec <= n["start_sec"] < end_sec
#     ]
#     num_onsets = len(in_interval)
#     if not in_interval:
#         return (None, num_onsets)
#     durations = [n["end_sec"] - n["start_sec"] for n in in_interval]
#     avg_length = sum(durations) / len(durations)
#     return (avg_length, num_onsets)

