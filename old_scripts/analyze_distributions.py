#!/usr/bin/env python3
"""
Analyze distributions of note lengths and inter-onset intervals (IOIs) across
the MAESTRO dataset. Answers:
- Is the distribution bell-curve or bimodal?
Outputs HTML report with histograms.
"""

import base64
import io
from pathlib import Path
import argparse
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import pandas as pd

from tempo import (
    get_all_notes,
    get_chord_onset_times,
    analyze_tempo_by_metrics,
)


def collect_note_lengths_and_iois_from_segments(
    midi_path: str,
    segment_map: list[tuple[float, float, str]],
) -> tuple[list[float], list[float], list[tuple[str, float]], list[tuple[str, float]]]:
    """
    For a single MIDI file, assign note lengths and IOIs to tempo segments.
    segment_map: list of (start_sec, end_sec, tempo_label) from tempo_songs.csv
    Returns: all_note_lengths, all_iois, note_lengths_by_speed, iois_by_speed
    """
    midi = mido.MidiFile(midi_path)
    all_notes = get_all_notes(midi)

    def get_speed(t: float) -> str:
        for start, end, cat in segment_map:
            if start <= t < end:
                return cat
        return "unknown"

    all_note_lengths = [n["end_sec"] - n["start_sec"] for n in all_notes]
    note_lengths_by_speed = [(get_speed(n["start_sec"]), n["end_sec"] - n["start_sec"]) for n in all_notes]

    onset_times = get_chord_onset_times(all_notes)
    all_iois = [onset_times[i + 1] - onset_times[i] for i in range(len(onset_times) - 1)]
    iois_by_speed = [(get_speed(onset_times[i]), onset_times[i + 1] - onset_times[i]) for i in range(len(onset_times) - 1)]

    return all_note_lengths, all_iois, note_lengths_by_speed, iois_by_speed


def collect_note_lengths_and_iois(
    midi_path: str,
    window_sec: float = 60.0,
) -> tuple[list[float], list[float], list[tuple[str, float]], list[tuple[str, float]]]:
    """
    For a single MIDI file, return:
    - all_note_lengths, all_iois
    - note_lengths_by_speed: (speed_category, duration)
    - iois_by_speed: (speed_category, ioi)
    """
    midi = mido.MidiFile(midi_path)
    all_notes = get_all_notes(midi)
    result = analyze_tempo_by_metrics(midi, window_sec=window_sec)

    section_map = [(w["start_sec"], w["end_sec"], w["speed_category"]) for w in result["window"]]

    def get_speed(t: float) -> str:
        for start, end, cat in section_map:
            if start <= t < end:
                return cat
        return "unknown"

    all_note_lengths = [n["end_sec"] - n["start_sec"] for n in all_notes]
    note_lengths_by_speed = [(get_speed(n["start_sec"]), n["end_sec"] - n["start_sec"]) for n in all_notes]

    onset_times = get_chord_onset_times(all_notes)
    all_iois = [onset_times[i + 1] - onset_times[i] for i in range(len(onset_times) - 1)]
    iois_by_speed = [(get_speed(onset_times[i]), onset_times[i + 1] - onset_times[i]) for i in range(len(onset_times) - 1)]

    return all_note_lengths, all_iois, note_lengths_by_speed, iois_by_speed


def distribution_stats(values: list[float], name: str) -> dict:
    if not values:
        return {}
    n = len(values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    try:
        stdev = statistics.stdev(values)
    except statistics.StatisticsError:
        stdev = 0.0
    skew_approx = (mean - median) / stdev if stdev > 0 else 0
    sorted_v = sorted(values)
    p5 = sorted_v[int(0.05 * n)] if n >= 20 else sorted_v[0]
    p25 = sorted_v[n // 4] if n >= 4 else sorted_v[0]
    p75 = sorted_v[(3 * n) // 4] if n >= 4 else sorted_v[-1]
    p95 = sorted_v[int(0.95 * n)] if n >= 20 else sorted_v[-1]
    return {
        "name": name,
        "n": n,
        "mean": mean,
        "median": median,
        "stdev": stdev,
        "skew_approx": skew_approx,
        "p5": p5,
        "p25": p25,
        "p75": p75,
        "p95": p95,
    }


def bimodality_coefficient(values: list[float]) -> float:
    """BC > 5/9 (~0.56) suggests bimodality."""
    if len(values) < 4:
        return 0.0
    n = len(values)
    mean = statistics.mean(values)
    m2 = sum((x - mean) ** 2 for x in values) / n
    m3 = sum((x - mean) ** 3 for x in values) / n
    m4 = sum((x - mean) ** 4 for x in values) / n
    if m2 <= 0:
        return 0.0
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3
    if kurt <= 0:
        return 0.0
    return (skew**2 + 1) / kurt


def fig_to_base64(fig: plt.Figure, dpi: int = 120) -> str:
    """Encode matplotlib figure as base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_histogram(
    values: list[float],
    title: str,
    xlabel: str = "seconds",
    xmax: float | None = None,
    bins: int = 80,
) -> str:
    """Create histogram and return base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, color="#4a90d9", alpha=0.8, edgecolor="white", linewidth=0.3)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_overlay_histogram(
    slow_values: list[float],
    fast_values: list[float],
    title: str,
    xlabel: str = "seconds",
    xmax: float | None = None,
    bins: int = 80,
) -> str:
    """Create histogram overlaying slow vs fast/very_fast and return base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if slow_values:
        ax.hist(slow_values, bins=bins, color="#4a90d9", alpha=0.6, label="Slow", edgecolor="white", linewidth=0.3)
    if fast_values:
        ax.hist(fast_values, bins=bins, color="#e74c3c", alpha=0.6, label="Fast / Very fast", edgecolor="white", linewidth=0.3)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_overlay_histogram_3way(
    slow_values: list[float],
    medium_values: list[float],
    fast_values: list[float],
    title: str,
    xlabel: str = "seconds",
    xmax: float | None = None,
    bins: int = 80,
) -> str:
    """Create histogram overlaying slow, medium, fast and return base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if slow_values:
        ax.hist(slow_values, bins=bins, color="#4a90d9", alpha=0.6, label="Slow", edgecolor="white", linewidth=0.3)
    if medium_values:
        ax.hist(medium_values, bins=bins, color="#27ae60", alpha=0.6, label="Medium", edgecolor="white", linewidth=0.3)
    if fast_values:
        ax.hist(fast_values, bins=bins, color="#e74c3c", alpha=0.6, label="Fast", edgecolor="white", linewidth=0.3)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_overlay_histogram_4way(
    slow_values: list[float],
    medium_values: list[float],
    fast_values: list[float],
    very_fast_values: list[float],
    title: str,
    xlabel: str = "seconds",
    xmax: float | None = None,
    bins: int = 80,
) -> str:
    """Create histogram overlaying slow, medium, fast, very_fast."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if slow_values:
        ax.hist(slow_values, bins=bins, color="#4a90d9", alpha=0.5, label="Slow", edgecolor="white", linewidth=0.3)
    if medium_values:
        ax.hist(medium_values, bins=bins, color="#27ae60", alpha=0.5, label="Medium", edgecolor="white", linewidth=0.3)
    if fast_values:
        ax.hist(fast_values, bins=bins, color="#e74c3c", alpha=0.5, label="Fast", edgecolor="white", linewidth=0.3)
    if very_fast_values:
        ax.hist(very_fast_values, bins=bins, color="#9b59b6", alpha=0.5, label="Very fast", edgecolor="white", linewidth=0.3)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def run_analysis(
    data_dir: str = "./maestro-v3.0.0",
    max_pieces: int | None = None,
    window_sec: float = 60.0,
    output_html: str | None = None,
    tempo_songs_csv: str | None = None,
    tempo_sections_csv: str | None = None,
) -> None:
    use_tempo_songs = tempo_songs_csv is not None
    if use_tempo_songs:
        songs_path = Path(tempo_songs_csv)
        if not songs_path.exists():
            print(f"Error: {songs_path} not found.", file=sys.stderr)
            sys.exit(1)
        songs_df = pd.read_csv(songs_path)
        # Build piece_index -> [(start_sec, end_sec, tempo_label), ...]
        segments_by_piece: dict[int, list[tuple[float, float, str]]] = {}
        for _, row in songs_df.iterrows():
            pid = int(row["piece_index"])
            if pid not in segments_by_piece:
                segments_by_piece[pid] = []
            segments_by_piece[pid].append((
                float(row["start_sec"]),
                float(row["end_sec"]),
                str(row["tempo_label"]),
            ))
        # Get unique pieces with midi_filename
        piece_meta = songs_df.groupby("piece_index").first().reset_index()
        n = len(piece_meta)
        if max_pieces is not None:
            piece_meta = piece_meta.iloc[:max_pieces]
            n = len(piece_meta)
    else:
        csv_path = Path(data_dir) / "maestro-v3.0.0.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found.", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(csv_path)
        if max_pieces is not None:
            df = df.iloc[:max_pieces]
        n = len(df)

    all_note_lengths = []
    all_iois = []
    slow_note_lengths = []
    slow_iois = []
    medium_note_lengths = []
    medium_iois = []
    fast_note_lengths = []
    fast_iois = []
    very_fast_note_lengths = []
    very_fast_iois = []

    def process_piece(idx: int, midi_path: Path, segment_map: list | None = None) -> None:
        nonlocal all_note_lengths, all_iois
        nonlocal slow_note_lengths, slow_iois, medium_note_lengths, medium_iois
        nonlocal fast_note_lengths, fast_iois, very_fast_note_lengths, very_fast_iois
        if not midi_path.exists():
            return
        try:
            if segment_map is not None:
                nl, iois, nl_by_speed, ioi_by_speed = collect_note_lengths_and_iois_from_segments(
                    str(midi_path), segment_map
                )
            else:
                nl, iois, nl_by_speed, ioi_by_speed = collect_note_lengths_and_iois(
                    str(midi_path), window_sec=window_sec
                )
        except Exception as e:
            print(f"Skip {midi_path.name}: {e}", file=sys.stderr)
            return
        all_note_lengths.extend(nl)
        all_iois.extend(iois)
        for cat, v in nl_by_speed:
            if cat == "slow":
                slow_note_lengths.append(v)
            elif cat == "medium":
                medium_note_lengths.append(v)
            elif cat == "fast":
                fast_note_lengths.append(v)
            elif cat == "very_fast":
                very_fast_note_lengths.append(v)
        for cat, v in ioi_by_speed:
            if cat == "slow":
                slow_iois.append(v)
            elif cat == "medium":
                medium_iois.append(v)
            elif cat == "fast":
                fast_iois.append(v)
            elif cat == "very_fast":
                very_fast_iois.append(v)

    if use_tempo_songs:
        for i in range(len(piece_meta)):
            row = piece_meta.iloc[i]
            pid = int(row["piece_index"])
            midi_path = Path(data_dir) / row["midi_filename"]
            segment_map = segments_by_piece.get(pid, [])
            process_piece(i, midi_path, segment_map)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{n} pieces...", file=sys.stderr)
    else:
        for idx in range(n):
            row = df.iloc[idx]
            midi_path = Path(data_dir) / row["midi_filename"]
            process_piece(idx, midi_path, None)
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{n} pieces...", file=sys.stderr)

    def trim_extremes(vals: list[float], p_low: float = 0.001, p_high: float = 0.999) -> list[float]:
        if len(vals) < 100:
            return vals
        s = sorted(vals)
        n = len(s)
        lo, hi = s[int(p_low * n)], s[int(p_high * n)]
        return [x for x in vals if lo <= x <= hi]

    all_nl_trim = trim_extremes(all_note_lengths)
    all_ioi_trim = trim_extremes(all_iois)
    slow_nl_trim = trim_extremes(slow_note_lengths) if slow_note_lengths else []
    slow_ioi_trim = trim_extremes(slow_iois) if slow_iois else []
    medium_nl_trim = trim_extremes(medium_note_lengths) if medium_note_lengths else []
    medium_ioi_trim = trim_extremes(medium_iois) if medium_iois else []
    fast_nl_trim = trim_extremes(fast_note_lengths) if fast_note_lengths else []
    fast_ioi_trim = trim_extremes(fast_iois) if fast_iois else []
    fast_or_very_fast_nl = fast_note_lengths + very_fast_note_lengths
    fast_or_very_fast_ioi = fast_iois + very_fast_iois
    fast_or_very_fast_nl_trim = trim_extremes(fast_or_very_fast_nl) if fast_or_very_fast_nl else []
    fast_or_very_fast_ioi_trim = trim_extremes(fast_or_very_fast_ioi) if fast_or_very_fast_ioi else []

    # Build stats for each population
    nl_stats_base = [
        ("All pieces", all_note_lengths),
        ("All (trimmed 0.1%-99.9%)", all_nl_trim),
        ("Slow sections only", slow_note_lengths),
        ("Slow (trimmed)", slow_nl_trim),
    ]
    if medium_note_lengths:
        nl_stats_base.extend([
            ("Medium sections only", medium_note_lengths),
            ("Medium (trimmed)", medium_nl_trim),
        ])
    nl_stats_base.extend([
        ("Fast sections only", fast_note_lengths),
        ("Fast (trimmed)", fast_nl_trim),
    ])
    if very_fast_note_lengths:
        nl_stats_base.extend([
            ("Very fast sections only", very_fast_note_lengths),
            ("Fast + very fast combined", fast_or_very_fast_nl),
        ])
    nl_stats = []
    for label, vals in nl_stats_base:
        if vals:
            s = distribution_stats(vals, label)
            s["bc"] = bimodality_coefficient(vals)
            nl_stats.append((label, s))

    ioi_stats_base = [
        ("All pieces", all_iois),
        ("All (trimmed)", all_ioi_trim),
        ("Slow sections only", slow_iois),
        ("Slow (trimmed)", slow_ioi_trim),
    ]
    if medium_iois:
        ioi_stats_base.extend([
            ("Medium sections only", medium_iois),
            ("Medium (trimmed)", medium_ioi_trim),
        ])
    ioi_stats_base.extend([
        ("Fast sections only", fast_iois),
        ("Fast (trimmed)", fast_ioi_trim),
    ])
    if very_fast_iois:
        ioi_stats_base.extend([
            ("Very fast sections only", very_fast_iois),
            ("Fast + very fast combined", fast_or_very_fast_ioi),
        ])
    ioi_stats = []
    for label, vals in ioi_stats_base:
        if vals:
            s = distribution_stats(vals, label)
            s["bc"] = bimodality_coefficient(vals)
            ioi_stats.append((label, s))

    bc_nl = bimodality_coefficient(all_nl_trim)
    bc_ioi = bimodality_coefficient(all_ioi_trim)

    if output_html:
        # Generate histograms
        img_nl_all = plot_histogram(all_nl_trim, "Note lengths (all, trimmed)", xmax=1.0)
        fast_nl_for_hist = fast_nl_trim or fast_note_lengths
        img_nl_fast = plot_histogram(fast_nl_for_hist, "Note lengths (fast sections)", xmax=0.6) if fast_nl_for_hist else ""
        img_ioi_all = plot_histogram(all_ioi_trim, "Inter-onset intervals (all, trimmed)", xmax=0.8)
        fast_ioi_for_hist = fast_ioi_trim or fast_iois
        img_ioi_fast = plot_histogram(fast_ioi_for_hist, "Inter-onset intervals (fast sections)", xmax=0.5) if fast_ioi_for_hist else ""
        # Overlay: 3-way (tempo_songs) or 2-way (legacy)
        if medium_note_lengths:
            img_nl_slow_vs_fast = plot_overlay_histogram_3way(
                slow_nl_trim or slow_note_lengths,
                medium_nl_trim or medium_note_lengths,
                fast_nl_trim or fast_note_lengths,
                "Note lengths: Slow vs Medium vs Fast (tempo_songs)",
                xmax=0.8,
            )
            img_ioi_slow_vs_fast = plot_overlay_histogram_3way(
                slow_ioi_trim or slow_iois,
                medium_ioi_trim or medium_iois,
                fast_ioi_trim or fast_iois,
                "Inter-onset intervals: Slow vs Medium vs Fast (tempo_songs)",
                xmax=0.6,
            )
        else:
            img_nl_slow_vs_fast = plot_overlay_histogram(
                slow_nl_trim or slow_note_lengths,
                fast_or_very_fast_nl_trim or fast_or_very_fast_nl,
                "Note lengths: Slow vs Fast/Very fast",
                xmax=0.8,
            )
            img_ioi_slow_vs_fast = plot_overlay_histogram(
                slow_ioi_trim or slow_iois,
                fast_or_very_fast_ioi_trim or fast_or_very_fast_ioi,
                "Inter-onset intervals: Slow vs Fast/Very fast",
                xmax=0.6,
            )

        data_source = "tempo_songs" if use_tempo_songs else "MAESTRO"
        section_dist_html = ""
        if tempo_sections_csv:
            sections_path = Path(tempo_sections_csv)
            if sections_path.exists():
                sections_df = pd.read_csv(sections_path)
                if "speed_category" in sections_df.columns and "start_sec" in sections_df.columns:
                    # Build segments from tempo_sections_distribution (same structure as tempo_songs)
                    sec_segments_by_piece: dict[int, list[tuple[float, float, str]]] = {}
                    for _, row in sections_df.iterrows():
                        pid = int(row["piece_index"])
                        if pid not in sec_segments_by_piece:
                            sec_segments_by_piece[pid] = []
                        sec_segments_by_piece[pid].append((
                            float(row["start_sec"]),
                            float(row["end_sec"]),
                            str(row["speed_category"]),
                        ))
                    sec_piece_meta = sections_df.groupby("piece_index").first().reset_index()
                    # Collect note lengths and IOIs by tempo_sections labels
                    sec_slow_nl, sec_slow_ioi = [], []
                    sec_medium_nl, sec_medium_ioi = [], []
                    sec_fast_nl, sec_fast_ioi = [], []
                    sec_very_fast_nl, sec_very_fast_ioi = [], []
                    sec_all_nl, sec_all_ioi = [], []
                    for i in range(len(sec_piece_meta)):
                        row = sec_piece_meta.iloc[i]
                        pid = int(row["piece_index"])
                        midi_path = Path(data_dir) / row["midi_filename"]
                        seg_map = sec_segments_by_piece.get(pid, [])
                        if not midi_path.exists() or not seg_map:
                            continue
                        try:
                            nl, iois, nl_by_speed, ioi_by_speed = collect_note_lengths_and_iois_from_segments(
                                str(midi_path), seg_map
                            )
                        except Exception:
                            continue
                        sec_all_nl.extend(nl)
                        sec_all_ioi.extend(iois)
                        for cat, v in nl_by_speed:
                            if cat == "slow":
                                sec_slow_nl.append(v)
                            elif cat == "medium":
                                sec_medium_nl.append(v)
                            elif cat == "fast":
                                sec_fast_nl.append(v)
                            elif cat == "very_fast":
                                sec_very_fast_nl.append(v)
                        for cat, v in ioi_by_speed:
                            if cat == "slow":
                                sec_slow_ioi.append(v)
                            elif cat == "medium":
                                sec_medium_ioi.append(v)
                            elif cat == "fast":
                                sec_fast_ioi.append(v)
                            elif cat == "very_fast":
                                sec_very_fast_ioi.append(v)
                    # Trim and build stats for tempo sections
                    def trim(vals: list[float], p_low: float = 0.001, p_high: float = 0.999) -> list[float]:
                        if len(vals) < 100:
                            return vals
                        s = sorted(vals)
                        ln = len(s)
                        lo, hi = s[int(p_low * ln)], s[int(p_high * ln)]
                        return [x for x in vals if lo <= x <= hi]
                    sec_nl_trim = trim(sec_all_nl)
                    sec_ioi_trim = trim(sec_all_ioi)
                    sec_slow_nl_t = trim(sec_slow_nl) if sec_slow_nl else []
                    sec_slow_ioi_t = trim(sec_slow_ioi) if sec_slow_ioi else []
                    sec_medium_nl_t = trim(sec_medium_nl) if sec_medium_nl else []
                    sec_medium_ioi_t = trim(sec_medium_ioi) if sec_medium_ioi else []
                    sec_fast_nl_t = trim(sec_fast_nl) if sec_fast_nl else []
                    sec_fast_ioi_t = trim(sec_fast_ioi) if sec_fast_ioi else []
                    sec_vf_nl_t = trim(sec_very_fast_nl) if sec_very_fast_nl else []
                    sec_vf_ioi_t = trim(sec_very_fast_ioi) if sec_very_fast_ioi else []
                    # Stats
                    sec_nl_stats = []
                    for lbl, vals in [
                        ("All sections", sec_all_nl),
                        ("All (trimmed)", sec_nl_trim),
                        ("Slow", sec_slow_nl),
                        ("Slow (trimmed)", sec_slow_nl_t),
                        ("Medium", sec_medium_nl),
                        ("Medium (trimmed)", sec_medium_nl_t),
                        ("Fast", sec_fast_nl),
                        ("Fast (trimmed)", sec_fast_nl_t),
                        ("Very fast", sec_very_fast_nl),
                        ("Very fast (trimmed)", sec_vf_nl_t),
                    ]:
                        if vals:
                            s = distribution_stats(vals, lbl)
                            s["bc"] = bimodality_coefficient(vals)
                            sec_nl_stats.append((lbl, s))
                    sec_ioi_stats = []
                    for lbl, vals in [
                        ("All sections", sec_all_ioi),
                        ("All (trimmed)", sec_ioi_trim),
                        ("Slow", sec_slow_ioi),
                        ("Slow (trimmed)", sec_slow_ioi_t),
                        ("Medium", sec_medium_ioi),
                        ("Medium (trimmed)", sec_medium_ioi_t),
                        ("Fast", sec_fast_ioi),
                        ("Fast (trimmed)", sec_fast_ioi_t),
                        ("Very fast", sec_very_fast_ioi),
                        ("Very fast (trimmed)", sec_vf_ioi_t),
                    ]:
                        if vals:
                            s = distribution_stats(vals, lbl)
                            s["bc"] = bimodality_coefficient(vals)
                            sec_ioi_stats.append((lbl, s))
                    # Histograms
                    sec_img_nl_all = plot_histogram(sec_nl_trim, "Note lengths (all, trimmed)", xmax=1.0) if sec_nl_trim else ""
                    sec_img_nl_overlay = plot_overlay_histogram_4way(
                        sec_slow_nl_t or sec_slow_nl,
                        sec_medium_nl_t or sec_medium_nl,
                        sec_fast_nl_t or sec_fast_nl,
                        sec_vf_nl_t or sec_very_fast_nl,
                        "Note lengths: Slow vs Medium vs Fast vs Very fast (tempo_sections)",
                        xmax=0.8,
                    )
                    sec_img_nl_fast = ""
                    if sec_fast_nl or sec_very_fast_nl:
                        fast_vals = (sec_fast_nl_t or sec_fast_nl) + (sec_vf_nl_t or sec_very_fast_nl)
                        sec_img_nl_fast = plot_histogram(fast_vals, "Note lengths (fast + very fast)", xmax=0.6)
                    sec_img_ioi_all = plot_histogram(sec_ioi_trim, "Inter-onset intervals (all, trimmed)", xmax=0.8) if sec_ioi_trim else ""
                    sec_img_ioi_overlay = plot_overlay_histogram_4way(
                        sec_slow_ioi_t or sec_slow_ioi,
                        sec_medium_ioi_t or sec_medium_ioi,
                        sec_fast_ioi_t or sec_fast_ioi,
                        sec_vf_ioi_t or sec_very_fast_ioi,
                        "Inter-onset intervals: Slow vs Medium vs Fast vs Very fast (tempo_sections)",
                        xmax=0.6,
                    )
                    sec_img_ioi_fast = ""
                    if sec_fast_ioi or sec_very_fast_ioi:
                        fast_vals = (sec_fast_ioi_t or sec_fast_ioi) + (sec_vf_ioi_t or sec_very_fast_ioi)
                        sec_img_ioi_fast = plot_histogram(fast_vals, "IOIs (fast + very fast)", xmax=0.5)
                    sec_bc_nl = bimodality_coefficient(sec_nl_trim) if sec_nl_trim else 0
                    sec_bc_ioi = bimodality_coefficient(sec_ioi_trim) if sec_ioi_trim else 0
                    section_dist_html = _build_section_dist_html(
                        sec_nl_stats=sec_nl_stats,
                        sec_ioi_stats=sec_ioi_stats,
                        sec_bc_nl=sec_bc_nl,
                        sec_bc_ioi=sec_bc_ioi,
                        sec_img_nl_all=sec_img_nl_all,
                        sec_img_nl_overlay=sec_img_nl_overlay,
                        sec_img_nl_fast=sec_img_nl_fast,
                        sec_img_ioi_all=sec_img_ioi_all,
                        sec_img_ioi_overlay=sec_img_ioi_overlay,
                        sec_img_ioi_fast=sec_img_ioi_fast,
                    )
        html = _build_html(
            nl_stats=nl_stats,
            ioi_stats=ioi_stats,
            bc_nl=bc_nl,
            bc_ioi=bc_ioi,
            img_nl_all=img_nl_all,
            img_nl_fast=img_nl_fast or img_nl_all,
            img_ioi_all=img_ioi_all,
            img_ioi_fast=img_ioi_fast or img_ioi_all,
            img_nl_slow_vs_fast=img_nl_slow_vs_fast,
            img_ioi_slow_vs_fast=img_ioi_slow_vs_fast,
            n_pieces=n,
            data_source=data_source,
            section_dist_html=section_dist_html,
        )
        Path(output_html).write_text(html, encoding="utf-8")
        print(f"Wrote HTML report to {output_html}", file=sys.stderr)
    else:
        # Console output
        for label, s in nl_stats:
            print(f"\n{label}: n={s['n']}")
            print(f"  mean={s['mean']:.4f}s  median={s['median']:.4f}s  stdev={s['stdev']:.4f}s")
            print(f"  skew_approx={s['skew_approx']:.3f}  bimodality_coef={s['bc']:.3f}")
        for label, s in ioi_stats:
            print(f"\n{label}: n={s['n']}")
            print(f"  mean={s['mean']:.4f}s  median={s['median']:.4f}s  stdev={s['stdev']:.4f}s")
            print(f"  skew_approx={s['skew_approx']:.3f}  bimodality_coef={s['bc']:.3f}")
        print(f"\nNote lengths: BC={bc_nl:.3f} -> {'BIMODAL' if bc_nl > 0.56 else 'Unimodal'}")
        print(f"IOIs:         BC={bc_ioi:.3f} -> {'BIMODAL' if bc_ioi > 0.56 else 'Unimodal'}")


def _stat_row(label: str, s: dict) -> str:
    bc_class = "bimodal" if s["bc"] > 0.56 else "unimodal"
    return f"""
        <tr>
            <td>{label}</td>
            <td>{s['n']:,}</td>
            <td>{s['mean']:.4f}s</td>
            <td>{s['median']:.4f}s</td>
            <td>{s['stdev']:.4f}s</td>
            <td>{s['skew_approx']:.3f}</td>
            <td class="{bc_class}">{s['bc']:.3f}</td>
            <td>{s['p5']:.4f} / {s['p25']:.4f} / {s['p75']:.4f} / {s['p95']:.4f}</td>
        </tr>"""


def _build_section_dist_html(
    sec_nl_stats: list,
    sec_ioi_stats: list,
    sec_bc_nl: float,
    sec_bc_ioi: float,
    sec_img_nl_all: str,
    sec_img_nl_overlay: str,
    sec_img_nl_fast: str,
    sec_img_ioi_all: str,
    sec_img_ioi_overlay: str,
    sec_img_ioi_fast: str,
) -> str:
    """Build the Tempo Sections Distribution section (same structure as main section)."""
    nl_rows = "".join(_stat_row(label, s) for label, s in sec_nl_stats)
    ioi_rows = "".join(_stat_row(label, s) for label, s in sec_ioi_stats)
    return f"""
    <h2>Tempo Sections Distribution</h2>
    <p class="subtitle">Note lengths and IOIs by section-level labels (tempo_sections_distribution.csv, quantile-based)</p>

    <h3>Interpretation: Bell Curve vs Bimodal</h3>
    <div class="interpretation">
        <p><strong>Note lengths:</strong> BC = {sec_bc_nl:.3f} → {'BIMODAL (BC &gt; 0.56)' if sec_bc_nl > 0.56 else 'Unimodal (bell-like)'}</p>
        <p><strong>IOIs:</strong> BC = {sec_bc_ioi:.3f} → {'BIMODAL (BC &gt; 0.56)' if sec_bc_ioi > 0.56 else 'Unimodal (bell-like)'}</p>
    </div>

    <h3>Note Lengths (duration in seconds)</h3>
    {"<div class=\"chart\"><h4>All sections (trimmed)</h4><img src=\"data:image/png;base64," + sec_img_nl_all + "\" alt=\"Note lengths\"></div>" if sec_img_nl_all else ""}
    <div class="chart">
        <h4>Slow vs Medium vs Fast vs Very fast</h4>
        <img src="data:image/png;base64,{sec_img_nl_overlay}" alt="Note lengths by section tempo">
    </div>
    {"<div class=\"chart\"><h4>Fast + very fast</h4><img src=\"data:image/png;base64," + sec_img_nl_fast + "\" alt=\"Note lengths fast\"></div>" if sec_img_nl_fast else ""}
    <table>
        <thead><tr><th>Population</th><th>n</th><th>Mean</th><th>Median</th><th>Stdev</th><th>Skew</th><th>BC</th><th>p5 / p25 / p75 / p95</th></tr></thead>
        <tbody>{nl_rows}</tbody>
    </table>

    <h3>Inter-Onset Intervals (IOI in seconds)</h3>
    {"<div class=\"chart\"><h4>All sections (trimmed)</h4><img src=\"data:image/png;base64," + sec_img_ioi_all + "\" alt=\"IOI\"></div>" if sec_img_ioi_all else ""}
    <div class="chart">
        <h4>Slow vs Medium vs Fast vs Very fast</h4>
        <img src="data:image/png;base64,{sec_img_ioi_overlay}" alt="IOI by section tempo">
    </div>
    {"<div class=\"chart\"><h4>Fast + very fast</h4><img src=\"data:image/png;base64," + sec_img_ioi_fast + "\" alt=\"IOI fast\"></div>" if sec_img_ioi_fast else ""}
    <table>
        <thead><tr><th>Population</th><th>n</th><th>Mean</th><th>Median</th><th>Stdev</th><th>Skew</th><th>BC</th><th>p5 / p25 / p75 / p95</th></tr></thead>
        <tbody>{ioi_rows}</tbody>
    </table>
"""


def _build_html(
    nl_stats: list,
    ioi_stats: list,
    bc_nl: float,
    bc_ioi: float,
    img_nl_all: str,
    img_nl_fast: str,
    img_ioi_all: str,
    img_ioi_fast: str,
    img_nl_slow_vs_fast: str,
    img_ioi_slow_vs_fast: str,
    n_pieces: int,
    data_source: str = "MAESTRO",
    section_dist_html: str = "",
) -> str:
    nl_rows = "".join(_stat_row(label, s) for label, s in nl_stats)
    ioi_rows = "".join(_stat_row(label, s) for label, s in ioi_stats)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Tempo Distribution Analysis — MAESTRO</title>
    <style>
        :root {{ font-family: system-ui, -apple-system, sans-serif; }}
        body {{ max-width: 1000px; margin: 0 auto; padding: 2rem; background: #f8f9fa; }}
        h1 {{ color: #1a1a2e; margin-bottom: 0.5rem; }}
        .subtitle {{ color: #666; margin-bottom: 2rem; }}
        h2 {{ color: #16213e; margin-top: 2.5rem; border-bottom: 2px solid #4a90d9; padding-bottom: 0.25rem; }}
        h3 {{ color: #16213e; margin-top: 1.5rem; font-size: 1.1rem; }}
        h4 {{ color: #333; margin: 0 0 0.5rem 0; font-size: 0.95rem; }}
        .interpretation {{ background: #e8f4fc; padding: 1rem 1.25rem; border-radius: 8px; margin: 1.5rem 0; }}
        .interpretation strong {{ color: #0d47a1; }}
        table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 0.6rem 0.8rem; text-align: left; }}
        th {{ background: #4a90d9; color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f5f7fa; }}
        .bimodal {{ color: #c62828; font-weight: 600; }}
        .unimodal {{ color: #2e7d32; }}
        .chart {{ margin: 1.5rem 0; background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
        .chart img {{ max-width: 100%; height: auto; display: block; }}
        .chart h3 {{ margin: 0 0 0.75rem 0; color: #333; font-size: 1rem; }}
    </style>
</head>
<body>
    <h1>Tempo Distribution Analysis</h1>
    <p class="subtitle">{data_source} — {n_pieces:,} pieces</p>

    <h2>Interpretation: Bell Curve vs Bimodal</h2>
    <div class="interpretation">
        <p><strong>Note lengths:</strong> BC = {bc_nl:.3f} → {'BIMODAL (BC &gt; 0.56)' if bc_nl > 0.56 else 'Unimodal (bell-like)'}</p>
        <p><strong>IOIs:</strong> BC = {bc_ioi:.3f} → {'BIMODAL (BC &gt; 0.56)' if bc_ioi > 0.56 else 'Unimodal (bell-like)'}</p>
        <p style="margin-bottom:0; font-size:0.9em; color:#555;">Bimodality coefficient (BC) &gt; 0.56 suggests two underlying modes rather than a single bell curve.</p>
    </div>

    <h2>Note Lengths (duration in seconds)</h2>
    <div class="chart">
        <h3>All pieces (trimmed)</h3>
        <img src="data:image/png;base64,{img_nl_all}" alt="Note lengths histogram">
    </div>
    <div class="chart">
        <h3>Slow vs Fast/Very fast — overall comparison</h3>
        <img src="data:image/png;base64,{img_nl_slow_vs_fast}" alt="Note lengths: slow vs fast/very_fast">
    </div>
    <div class="chart">
        <h3>Fast sections only</h3>
        <img src="data:image/png;base64,{img_nl_fast}" alt="Note lengths fast histogram">
    </div>
    <table>
        <thead><tr><th>Population</th><th>n</th><th>Mean</th><th>Median</th><th>Stdev</th><th>Skew</th><th>BC</th><th>p5 / p25 / p75 / p95</th></tr></thead>
        <tbody>{nl_rows}</tbody>
    </table>

    <h2>Inter-Onset Intervals (IOI in seconds)</h2>
    <div class="chart">
        <h3>All pieces (trimmed)</h3>
        <img src="data:image/png;base64,{img_ioi_all}" alt="IOI histogram">
    </div>
    <div class="chart">
        <h3>Slow vs Fast/Very fast — overall comparison</h3>
        <img src="data:image/png;base64,{img_ioi_slow_vs_fast}" alt="IOI: slow vs fast/very_fast">
    </div>
    <div class="chart">
        <h3>Fast sections only</h3>
        <img src="data:image/png;base64,{img_ioi_fast}" alt="IOI fast histogram">
    </div>
    <table>
        <thead><tr><th>Population</th><th>n</th><th>Mean</th><th>Median</th><th>Stdev</th><th>Skew</th><th>BC</th><th>p5 / p25 / p75 / p95</th></tr></thead>
        <tbody>{ioi_rows}</tbody>
    </table>

    {section_dist_html}
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze note length and IOI distributions")
    parser.add_argument("--data-dir", default="./maestro-v3.0.0")
    parser.add_argument("--max-pieces", type=int, default=None, help="Limit pieces (for testing)")
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument(
        "--tempo-songs",
        default="tempo_songs.csv",
        help="Use tempo_songs.csv for segment labels (default: tempo_songs.csv). "
             "If provided, labels come from song-level tempo analysis.",
    )
    parser.add_argument(
        "--tempo-sections",
        default="tempo_sections_distribution.csv",
        help="Include tempo_sections_distribution.csv label counts in report (default: tempo_sections_distribution.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        default="../public_html/distribution_report.html",
        help="Output HTML file (default: ../public_html/distribution_report.html)",
    )
    args = parser.parse_args()
    run_analysis(
        data_dir=args.data_dir,
        max_pieces=args.max_pieces,
        window_sec=args.window_sec,
        output_html=args.output,
        tempo_songs_csv=args.tempo_songs,
        tempo_sections_csv=args.tempo_sections,
    )
