"""
Species identification.

Owner: Species ID
Branch: feat/species

Contract:
    identify_species(video_path, records) -> list[Record]
    - decide ONE species per TRACK, then apply it with contracts.set_track_species(...)
      (never give rows of the same track different species)
    - species is a species_id from data/species_list.csv, or "unknown"
    - species = "unknown" when the best score < config.UNKNOWN_THRESHOLD
    - top3 = best three {"species", "score"}, best first, scores 0-1
    - leave reviewed = False (only the review screen sets it to True)
    - crops come from the ORIGINAL frame using the record's bbox
    - must not write files (pipeline.py does that).

PLACEHOLDER below: fake, deterministic labels so the pipeline runs.

Suggested real version (BioCLIP via open_clip, runs offline after the first download):
    1. for each track pick up to 5 crops (highest det_conf or largest boxes)
    2. embed crops with BioCLIP
    3. compare with (a) text embeddings of the scientific names in species_list.csv
       and/or (b) the mean embedding of photos in data/species_library/<species_id>/
    4. softmax -> average over the track's crops -> top 3
    5. best score < UNKNOWN_THRESHOLD -> "unknown"
"""
from __future__ import annotations

from pathlib import Path

from tidesight import config
from tidesight.contracts import Record, SpeciesScore, load_species_list, set_track_species


def identify_species(video_path: str | Path, records: list[Record]) -> list[Record]:
    species_ids = list(load_species_list().keys())
    if not species_ids:
        raise ValueError("data/species_list.csv is empty")

    # TODO(Species ID): replace this placeholder with real BioCLIP identification.
    for track_id in sorted({rec["track_id"] for rec in records}):
        best = 0.62 if track_id % 2 else 0.31          # even tracks become "unknown" on purpose
        top3 = [SpeciesScore(species=species_ids[(track_id + k) % len(species_ids)], score=score)
                for k, score in enumerate((best, 0.2, 0.1))]
        species = top3[0]["species"] if best >= config.UNKNOWN_THRESHOLD else config.UNKNOWN_LABEL
        set_track_species(records, track_id, species, best, top3)
    return records
