from __future__ import annotations

from collections import defaultdict

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


def sample_times(start_s: float, interval_s: float = 60.0, n_samples: int = 15) -> list[float]:
    """Sub-sample points: start_s, start_s + interval_s, ... (n_samples of them)."""
    return [round(start_s + i * interval_s, 3) for i in range(n_samples)]


def compute_meancount(records: list[Record], times_s: list[float],
                      window_s: float = 0.1) -> dict:
    """MeanCount per species (and for all fish) over the given sub-sample times.

    Returns {"samples": [{"time_s", "total", "by_species": {species: n}}...],
             "meancount_total": float, "meancount_by_species": {species: float}}.
    A sample with no detections counts as 0 (that is what the protocol does too).
    Frames that triage skipped are therefore treated as empty - if triage misses fish,
    MeanCount is biased low.
    """
    samples = []
    species_seen: set[str] = set()
    for t in times_s:
        tracks: dict[int, str] = {}
        for rec in records:
            if abs(rec["time_s"] - t) <= window_s + 1e-9:
                tracks[rec["track_id"]] = rec["species"]
        by_species: dict[str, int] = defaultdict(int)
        for species in tracks.values():
            by_species[species] += 1
        species_seen.update(by_species)
        samples.append({"time_s": t, "total": len(tracks), "by_species": dict(by_species)})

    n = len(samples) or 1
    return {
        "samples": samples,
        "meancount_total": round(sum(s["total"] for s in samples) / n, 3),
        "meancount_by_species": {sp: round(sum(s["by_species"].get(sp, 0) for s in samples) / n, 3)
                                 for sp in sorted(species_seen)},
    }
