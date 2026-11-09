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

What it does now: every track is "unknown" (score 0, no suggestions) and the reviewer assigns
species in the review screen, which recalculates MaxN. This matches Coastal Action's own sheet,
which has an "Unknown/Unable to ID due to visibility" column.

Why the model is off (_ZERO_SHOT = False): we ran the BioCLIP path below on all 1,446 labelled
fish in Coastal Action's clip T1-STAT2 411 (crops from the original 4K frames). The fish there are
a silvery schooling species, yet BioCLIP called 47% of them "American eel" and 52 of them
"spotfin butterflyfish" (a tropical-visitor alert species), mostly with scores above 0.5, at
5.5 crops/s on a laptop CPU. Confidently wrong labels and false alerts are worse than "unknown".

The BioCLIP path (local, offline after the first download; set _ZERO_SHOT = True to use it):
    1. For each track pick up to _CROPS_PER_TRACK rows: the biggest boxes, then highest det_conf.
    2. Cut those boxes (plus a little context) out of the ORIGINAL frames.
    3. Embed the crops with BioCLIP and compare them with one class per species in
       species_list.csv: text prompts from its scientific and common name, averaged with the
       mean embedding of reference photos in data/species_library/<species_id>/ when there are any.
    4. Softmax over species, average over the track's crops -> top 3;
       best < UNKNOWN_THRESHOLD -> "unknown".
Next step: reviewed labels from Coastal Action become training data for a local classifier,
and its accuracy is measured on held-out reviewed fish before it is switched on.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from tidesight import config
from tidesight.contracts import Record, SpeciesScore, load_species_list, set_track_species, video_info

log = logging.getLogger(__name__)

_ZERO_SHOT = False          # BioCLIP off: measured wrong on real footage (see above)
_CROPS_PER_TRACK = 3        # crops embedded per track (more = slower, steadier)
_CONTEXT = 0.15             # extra border around each box, as a share of its size
_MIN_SIDE_PX = 4            # boxes thinner than this are only used if a track has nothing better
_BATCH = 32
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_PROMPTS = ("a photo of {sci}.", "a photo of {common}.", "a photo of {sci} with common name {common}.")

_model = None               # loaded once per process


class _BioClip:
    """Thin wrapper so tests can swap in a fake: embed_images(list of RGB arrays), embed_texts(list of str)."""

    def __init__(self, name: str = config.SPECIES_MODEL) -> None:
        import open_clip
        import torch
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, self.preprocess = open_clip.create_model_and_transforms(name)
        self.model = model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(name)
        self.logit_scale = float(model.logit_scale.exp().item())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        with self._torch.no_grad():
            emb = self.model.encode_text(self.tokenizer(texts).to(self.device)).float()
        return _normalise(emb.cpu().numpy())

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        out = []
        for i in range(0, len(images), _BATCH):
            batch = self._torch.stack([self.preprocess(Image.fromarray(im)) for im in images[i:i + _BATCH]])
            with self._torch.no_grad():
                out.append(self.model.encode_image(batch.to(self.device)).float().cpu().numpy())
        return _normalise(np.concatenate(out)) if out else np.zeros((0, 1), np.float32)


def _get_model():
    global _model
    if _model is None:
        log.info("Loading species model %s", config.SPECIES_MODEL)
        _model = _BioClip()
    return _model


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _library_photos(species_id: str) -> list[np.ndarray]:
    folder = Path(config.SPECIES_LIBRARY) / species_id
    photos = []
    for path in sorted(folder.glob("*")) if folder.is_dir() else []:
        if path.suffix.lower() in _IMAGE_EXTS:
            img = cv2.imread(str(path))
            if img is not None:
                photos.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return photos


def species_classes(model, species_list: dict[str, dict]) -> tuple[list[str], np.ndarray]:
    """One unit vector per species: text prompts, plus library photos when there are any."""
    ids, vectors = [], []
    for sid, row in species_list.items():
        prompts = [p.format(sci=row["scientific_name"], common=row["common_name"]) for p in _PROMPTS]
        vec = _normalise(model.embed_texts(prompts).mean(0))
        photos = _library_photos(sid)
        if photos:
            vec = _normalise(vec + _normalise(model.embed_images(photos).mean(0)))
        ids.append(sid)
        vectors.append(vec)
    return ids, np.stack(vectors).astype(np.float32)


def _pick_rows(rows: list[Record]) -> list[Record]:
    def size(rec):
        x1, y1, x2, y2 = rec["bbox"]
        return (x2 - x1) * (y2 - y1)
    usable = [r for r in rows if min(r["bbox"][2] - r["bbox"][0], r["bbox"][3] - r["bbox"][1]) >= _MIN_SIDE_PX]
    return sorted(usable or rows, key=lambda r: (size(r), r["det_conf"]), reverse=True)[:_CROPS_PER_TRACK]


def _crop(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    """Square RGB crop around the box, with some context, clipped to the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    half = max(x2 - x1, y2 - y1, 1) * (1 + 2 * _CONTEXT) / 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(w, cx + half)), int(min(h, cy + half))
    return cv2.cvtColor(frame[b:max(d, b + 1), a:max(c, a + 1)], cv2.COLOR_BGR2RGB)


def _read_crops(video_path: str | Path, wanted: dict[int, list[tuple[int, list[int]]]]) -> dict[int, list[np.ndarray]]:
    """wanted: frame -> [(track_id, bbox)]. Returns track_id -> RGB crops from the original frames."""
    from tidesight.detect import FrameReader          # verified seeking; does not load the detector
    info = video_info(video_path)
    reader = FrameReader(video_path, info["frame_count"], info["fps"])
    crops: dict[int, list[np.ndarray]] = defaultdict(list)
    try:
        for frame_no in sorted(wanted):
            frame = reader.read(frame_no)
            for track_id, bbox in wanted[frame_no]:
                crops[track_id].append(_crop(frame, bbox))
    finally:
        reader.close()
    return crops


def score_tracks(model, crops: dict[int, list[np.ndarray]], ids: list[str],
                 classes: np.ndarray) -> dict[int, np.ndarray]:
    """track_id -> probability per species (same order as ids), averaged over the track's crops."""
    order = [(tid, k) for tid in sorted(crops) for k in range(len(crops[tid]))]
    if not order:
        return {}
    emb = model.embed_images([crops[tid][k] for tid, k in order])
    logits = model.logit_scale * emb @ classes.T
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    per_track: dict[int, list[np.ndarray]] = defaultdict(list)
    for (tid, _), p in zip(order, probs):
        per_track[tid].append(p)
    return {tid: np.mean(ps, axis=0) for tid, ps in per_track.items()}


def identify_species(video_path: str | Path, records: list[Record]) -> list[Record]:
    species_list = load_species_list()
    if not species_list:
        raise ValueError("data/species_list.csv is empty")
    if not records:
        return records

    by_track: dict[int, list[Record]] = defaultdict(list)
    for rec in records:
        by_track[rec["track_id"]].append(rec)
    if not _ZERO_SHOT:
        for track_id, rows in by_track.items():               # rows share dicts with records
            set_track_species(rows, track_id, config.UNKNOWN_LABEL, 0.0, [])
        return records

    wanted: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for track_id, rows in by_track.items():
        for rec in _pick_rows(rows):
            wanted[rec["frame"]].append((track_id, rec["bbox"]))
    model = _get_model()
    ids, classes = species_classes(model, species_list)
    log.info("Species: %d tracks, %d crops from %d frames", len(by_track),
             sum(len(v) for v in wanted.values()), len(wanted))
    scores = score_tracks(model, _read_crops(video_path, wanted), ids, classes)

    for track_id, rows in by_track.items():
        p = scores.get(track_id, np.zeros(len(ids)))
        best = np.argsort(-p, kind="stable")[:3]
        top3 = [SpeciesScore(species=ids[i], score=float(p[i])) for i in best]
        top_score = round(float(p[best[0]]), 3)                # saved rounded, so decide on that
        species = ids[best[0]] if top_score >= config.UNKNOWN_THRESHOLD else config.UNKNOWN_LABEL
        set_track_species(rows, track_id, species, top_score, top3)
    return records
