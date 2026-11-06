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
