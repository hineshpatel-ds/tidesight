"""
TideSight review app (runs locally; nothing leaves the laptop).

Owner: App + integration
Branch: feat/app

Start it from the project folder:
    streamlit run app.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import streamlit as st

from tidesight import config
from tidesight.contracts import (ContractError, clip_id_from_path, load_species_list, read_jsonl,
                                 read_maxn_csv, read_segments, set_track_species)
from tidesight.enhance import enhance
from tidesight.pipeline import run_pipeline, save_review


def inspector_frames(records):
    """Group only frames evidenced by saved detections, retaining original indices."""
    groups = {}
    for rec in records:
        groups.setdefault(rec["frame"], []).append(rec)
    return {frame: sorted(groups[frame], key=lambda r: r["track_id"])
            for frame in sorted(groups)}


def inspector_frame_label(frame, rows):
    return f"Original frame {frame} — {rows[0]['time_s']:.3f} s"


def inspector_evidence(rows, species_list):
    labels = {sid: item["common_name"] for sid, item in species_list.items()}
    labels.update({config.UNKNOWN_LABEL: "Unknown", config.FISH_PLACEHOLDER: "Unidentified fish"})
    return [{"Track": r["track_id"],
             "Saved species decision": labels.get(r["species"], "Unrecognized species"),
             "Review status": "Reviewed" if r["reviewed"] else "Not reviewed",
             "Detector model score": r["det_conf"],
             "Original species model score": r["species_conf"]} for r in rows]


def inspector_read_frame(video, frame):
    """Decode from the beginning; fail closed if original position cannot be verified.

    Deliberately avoid keyframe seeking and caching: a replaced source must be
    read afresh. Position is checked before and after every sequential decode.
    """
    cap = cv2.VideoCapture(video)
    try:
        if not cap.isOpened():
            raise ValueError("The source video could not be opened.")
        if not 0 <= frame < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            raise ValueError("The saved original frame is outside the source video.")
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        for index in range(frame + 1):
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != index:
                raise ValueError("The decoder cannot verify the requested original frame position.")
            ok, image = cap.read()
            if not ok or image is None:
                raise ValueError("The source video could not decode the requested original frame.")
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != index + 1:
                raise ValueError("The decoder cannot verify the requested original frame position.")
        if image.shape != (height, width, 3):
            raise ValueError("Decoded frame dimensions do not match the source video.")
        return image
    finally:
        cap.release()


def inspector_overlay(raw, rows, enhanced):
    """Draw saved original-resolution boxes after display enhancement, without mutation."""
    image = enhance(raw.copy()) if enhanced else raw.copy()
    if image is None or image.shape != raw.shape:
        raise ValueError("Enhanced frame dimensions changed. Choose Raw to inspect aligned boxes.")
    height, width = raw.shape[:2]
    for rec in rows:
        x1, y1, x2, y2 = rec["bbox"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("A saved box is outside the source dimensions; aligned display is unavailable.")
        # x2/y2 are exclusive crop bounds in the existing review workflow.
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 1)
        label = f"Track {rec['track_id']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, .5, 3)
        origin = (max(1, min(x1, width - text_width - 2)),
                  max(text_height + 1, min(y1 - 5, height - baseline - 2)))
        cv2.putText(image, label, origin, font, .5, (0, 0, 0), 3)
        cv2.putText(image, label, origin, font, .5, (0, 255, 0), 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="TideSight", layout="wide")
st.title("TideSight")
st.caption("Offline fish detection, identification and counting. No data leaves this computer.")

# ------------------------------------------------------------------ pick a clip
clips = sorted(p for p in config.CLIPS_DIR.glob("*") if p.suffix.lower() in config.VIDEO_EXTS)
if not clips:
    st.info("Put a video in data/clips/ (or run: python scripts/make_test_video.py), then refresh.")
    st.stop()

clip_path: Path = st.sidebar.selectbox("Video", clips, format_func=lambda p: p.name)
clip_id = clip_id_from_path(clip_path)
out_dir = config.OUTPUTS_DIR / clip_id

# A new form generation discards unsaved choices after switching clips, analysing,
# or successfully saving. Previously saved review status comes from the records.
if st.session_state.get("review_clip") != clip_id:
    st.session_state.review_clip = clip_id
    st.session_state.review_generation = st.session_state.get("review_generation", 0) + 1
    st.session_state.pop("review_saved", None)

if st.sidebar.button("Run analysis", type="primary"):
    st.session_state.pop("review_saved", None)
    failures = dict(st.session_state.get("analysis_failures", {}))
    try:
        with st.spinner("Analysing video..."):
            result = run_pipeline(clip_path)
    except Exception as exc:
        failures[clip_id] = str(exc)
    else:
        failures.pop(clip_id, None)
        st.session_state.review_generation += 1
        st.sidebar.success(f"Done: {result['n_tracks']} tracks, {result['skipped_pct']}% of video skipped")
    st.session_state.analysis_failures = failures

failure = st.session_state.get("analysis_failures", {}).get(clip_id)
if failure is not None:
    st.error(f"Analysis failed: {failure}")

det_path = out_dir / config.DETECTIONS_FILE
if not all((out_dir / name).is_file() for name in
           (config.SEGMENTS_FILE, config.DETECTIONS_FILE, config.MAXN_FILE)):
    st.info("No complete results are available. Click **Run analysis** to process this video.")
    st.stop()
if failure is not None:
    st.warning("Showing the last complete results. The latest analysis failed.")

records = read_jsonl(det_path)
segments = read_segments(out_dir / config.SEGMENTS_FILE)
maxn = read_maxn_csv(out_dir / config.MAXN_FILE)
species_list = load_species_list()


@st.cache_data(show_spinner=False)
def track_thumbnail(video: str, frame: int, bbox: tuple[int, int, int, int]):
    """Enhanced crop of one fish, for the review screen."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    x1, y1, x2, y2 = bbox
    crop = enhance(img)[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.size else None


# ------------------------------------------------------------------ overview
left, right = st.columns([3, 2])
with left:
    if clip_path.name == "test_clip.mp4":
        st.caption("Synthetic test clip — not real survey footage. Sample analysis does not "
                   "demonstrate detection or species accuracy.")
    st.video(str(clip_path))
    st.subheader("Busy segments (from triage)")
    st.dataframe(segments, width="stretch")
with right:
    st.subheader("MaxN per species")
    st.dataframe(maxn, width="stretch")
    st.download_button("Download MaxN CSV", (out_dir / config.MAXN_FILE).read_bytes(),
                       file_name=f"{clip_id}_maxn.csv", mime="text/csv")
    energy_path = out_dir / config.ENERGY_FILE
    if energy_path.exists():
        with open(energy_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            st.metric("Energy used by last analysis (Wh)", f"{float(last['energy_consumed']) * 1000:.3f}")
            st.metric("Emissions (g CO2e)", f"{float(last['emissions']) * 1000:.3f}")

# ------------------------------------------------------------------ frame inspector
st.subheader("Frame inspector")
st.caption("Inspect saved detections in their original frame. This control is independent of video playback. "
           "Only frames with saved detections are listed; analyzed frames with zero detections are not logged. "
           "Sequential decoding verifies frame position and may take longer for later frames.")
frame_groups = inspector_frames(records)
if not frame_groups:
    st.info("No saved detections. A completed analysis can validly return an empty result; "
            "this does not prove no fish were present. No saved detection frames are available to inspect.")
else:
    selected_frame = st.selectbox("Saved frame", list(frame_groups),
                                  format_func=lambda f: inspector_frame_label(f, frame_groups[f]),
                                  key=f"inspector_frame_{clip_id}")
    display = st.radio("Frame display", ["Raw", "Enhanced"], horizontal=True,
                       key=f"inspector_display_{clip_id}")
    selected_rows = frame_groups[selected_frame]
    try:
        raw = inspector_read_frame(str(clip_path), selected_frame)
        annotated = inspector_overlay(raw, selected_rows, display == "Enhanced")
    except (ValueError, cv2.error, OSError) as exc:
        st.warning(f"Frame image unavailable: {exc} Saved evidence remains available below.")
    else:
        st.image(annotated, caption=inspector_frame_label(selected_frame, selected_rows), width="stretch")
    st.caption("Saved species decisions and review status are shown below. Scores are original model scores, "
               "not calibrated probabilities or confidence in a human correction. "
               "Use Review each fish below to confirm or correct a track.")
    st.dataframe(inspector_evidence(selected_rows, species_list), width="stretch", hide_index=True)

# ------------------------------------------------------------------ review
st.subheader("Review each fish")
if st.session_state.pop("review_saved", None) == clip_id:
    st.success("Saved. MaxN recalculated.")
st.caption("Only checked tracks will be saved. Check a track to confirm its current species or a correction. "
           "Original model scores and suggestions are retained as evidence; they are not confidence "
           "in a human correction.")
options = list(species_list) + [config.UNKNOWN_LABEL]
labels = {sid: row["common_name"] for sid, row in species_list.items()}
labels[config.UNKNOWN_LABEL] = "Unknown (expert review)"

tracks: dict[int, dict] = {}
for rec in records:
    best = tracks.get(rec["track_id"])
    if best is None or rec["det_conf"] > best["det_conf"]:
        tracks[rec["track_id"]] = rec

form_key = f"review_{clip_id}_{st.session_state.review_generation}"
with st.form(form_key):
    choices: dict[int, str] = {}
    confirmed: dict[int, bool] = {}
    for track_id, rec in sorted(tracks.items()):
        c1, c2, c3 = st.columns([1, 2, 2])
        thumb = track_thumbnail(str(clip_path), rec["frame"], tuple(rec["bbox"]))
        if thumb is not None:
            c1.image(thumb, caption=f"Track {track_id} at {rec['time_s']} s")
        suggestions = ", ".join(f"{labels.get(t['species'], t['species'])} {t['score']:.2f}" for t in rec["top3"])
        c2.markdown(f"**Track {track_id}**  \nOriginal model score: {rec['species_conf']:.2f}  \n"
                    f"Original model suggestions: {suggestions or '-'}  \n"
                    f"Reviewed: {'yes' if rec['reviewed'] else 'no'}")
        current = rec["species"] if rec["species"] in options else config.UNKNOWN_LABEL
        choices[track_id] = c3.selectbox("Species", options, index=options.index(current),
                                         format_func=lambda s: labels.get(s, s), key=f"{form_key}_species_{track_id}")
        confirmed[track_id] = c3.checkbox("Confirm this track", key=f"{form_key}_confirm_{track_id}")
    if st.form_submit_button("Save review"):
        if not any(confirmed.values()):
            st.info("Check at least one track to confirm before saving. No changes saved.")
        else:
            for track_id, species in choices.items():
                if confirmed[track_id]:
                    # The model score is consistent per track under the contract.
                    # Keep it and top3 as model evidence, even after a correction.
                    set_track_species(records, track_id, species,
                                      tracks[track_id]["species_conf"], reviewed=True)
            try:
                save_review(clip_id, records)
            except (ContractError, OSError) as exc:
                st.error(f"Review could not be saved: {exc}")
            else:
                st.session_state.review_saved = clip_id
                st.session_state.review_generation += 1
                st.rerun()
