"""
Counting: MaxN per species.

Owner: ML lead
Branch: feat/detect

MaxN = the highest number of individuals of one species visible in any single frame.
It is the standard count for underwater video because it never counts the same fish
twice. n_tracks (distinct tracks) is reported as a secondary number; it overcounts
fish that leave and come back.

This module is complete. Change it only if the counting rule changes.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import TypedDict

from tidesight import config
from tidesight.contracts import MaxNRow, Record, load_species_list

_SPECIAL_NAMES = {
    config.UNKNOWN_LABEL: "Unknown (needs expert review)",
    config.FISH_PLACEHOLDER: "Fish (not yet identified)",
}


def compute_maxn(records: list[Record], species_list: dict[str, dict] | None = None) -> list[MaxNRow]:
    if not records:
        return []
    if species_list is None:
        species_list = load_species_list()

    clip_id = records[0]["clip_id"]
    frame_time: dict[int, float] = {}
    tracks_in_frame: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    tracks_per_species: dict[str, set[int]] = defaultdict(set)

    for rec in records:
        frame_time[rec["frame"]] = rec["time_s"]
        tracks_in_frame[rec["species"]][rec["frame"]].add(rec["track_id"])
        tracks_per_species[rec["species"]].add(rec["track_id"])

    rows: list[MaxNRow] = []
    for species, frames in tracks_in_frame.items():
        best_frame = min(frames, key=lambda f: (-len(frames[f]), f))   # highest count, earliest frame
        common_name = (species_list.get(species, {}).get("common_name")
                       or _SPECIAL_NAMES.get(species, species))
        rows.append(MaxNRow(clip_id=clip_id, species_id=species, common_name=common_name,
                            maxn=len(frames[best_frame]), frame=best_frame,
                            time_s=frame_time[best_frame],
                            n_tracks=len(tracks_per_species[species])))
    rows.sort(key=lambda r: (-r["maxn"], r["species_id"]))
    return rows


class ProtocolError(ValueError):
    pass


class SamplePoint(TypedDict):
    index: int
    time_s: float
    frame: int


def nearest_frame(time_s: float, fps: float) -> int:
    return int(math.floor(time_s * fps + 0.5 + 1e-9))


def protocol_sample_frames(start_s: float, fps: float, frame_count: int,
                           n_samples: int | None = None, interval_s: float | None = None,
                           edge_s: float | None = None) -> list[SamplePoint]:
    n = config.PROTOCOL_N_SAMPLES if n_samples is None else n_samples
    step = config.PROTOCOL_INTERVAL_S if interval_s is None else interval_s
    edge = config.PROTOCOL_EDGE_S if edge_s is None else edge_s
    if fps <= 0 or frame_count <= 0:
        raise ProtocolError(f"Video has no usable fps/frame count (fps={fps}, frames={frame_count}).")
    if n < 1 or step <= 0:
        raise ProtocolError(f"Invalid protocol: {n} samples every {step} s.")
    if not math.isfinite(start_s) or start_s < edge:
        raise ProtocolError(f"Start {start_s} s is inside the first {edge} s, which the protocol discards.")
    duration = frame_count / fps
    last_t = start_s + (n - 1) * step
    if last_t > duration - edge:
        raise ProtocolError(f"Last sample at {last_t:.1f} s is beyond {duration - edge:.1f} s "
                            f"(video is {duration:.1f} s and the last {edge} s are discarded).")
    points = []
    for i in range(n):
        t = round(start_s + i * step, 6)
        f = nearest_frame(t, fps)
        if not 0 <= f < frame_count:
            raise ProtocolError(f"Sample {i} at {t} s maps to frame {f}, outside the video.")
        points.append(SamplePoint(index=i, time_s=t, frame=f))
    return points


def compute_meancount(run: dict) -> dict:
    samples = run.get("samples", [])
    n = run.get("n_samples")
    if not samples or n != len(samples):
        raise ProtocolError(f"Protocol run declares {n} samples but contains {len(samples)}.")
    if [s["index"] for s in samples] != list(range(n)):
        raise ProtocolError("Protocol samples are missing or out of order.")
    rows = []
    species_seen: set[str] = set()
    for s in samples:
        if s.get("status") != "done":
            raise ProtocolError(f"Sample {s['index']} was not completed (status={s.get('status')!r}).")
        if any(r["frame"] != s["frame"] for r in s["records"]):
            raise ProtocolError(f"Sample {s['index']} contains records from frames other than {s['frame']}.")
        by_species: dict[str, int] = defaultdict(int)
        for r in s["records"]:
            by_species[r["species"]] += 1
        species_seen.update(by_species)
        rows.append({"index": s["index"], "time_s": s["time_s"], "frame": s["frame"],
                     "fish": len(s["records"]), "by_species": dict(by_species)})
    return {
        "clip_id": run.get("clip_id"),
        "n_samples": n,
        "start_s": run.get("start_s"),
        "interval_s": run.get("interval_s"),
        "meancount_fish": round(sum(r["fish"] for r in rows) / n, 3),
        "meancount_by_species": {sp: round(sum(r["by_species"].get(sp, 0) for r in rows) / n, 3)
                                 for sp in sorted(species_seen)},
        "samples": rows,
    }
