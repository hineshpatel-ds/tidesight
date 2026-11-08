"""Local video analysis and fish review. Run with: streamlit run app.py."""
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
    """Group saved detections by original frame number."""
    groups = {}
    for rec in records:
        groups.setdefault(rec["frame"], []).append(rec)
    return {frame: sorted(groups[frame], key=lambda r: r["track_id"])
            for frame in sorted(groups)}


def inspector_frame_label(frame, rows):
    return f"Frame {frame} ({rows[0]['time_s']:.3f} s)"


def inspector_evidence(rows, species_list):
    labels = {sid: item["common_name"] for sid, item in species_list.items()}
    labels.update({config.UNKNOWN_LABEL: "Unknown", config.FISH_PLACEHOLDER: "Unidentified fish"})
    return [{"Track": r["track_id"],
             "Saved species decision": labels.get(r["species"], "Unrecognized species"),
             "Review status": "Reviewed" if r["reviewed"] else "Not reviewed",
             "Detector model score": r["det_conf"],
             "Original species model score": r["species_conf"]} for r in rows]


def inspector_read_frame(video, frame):
    """Read sequentially and verify frame position; avoid inexact keyframe seeking."""
    cap = cv2.VideoCapture(video)
    try:
        if not cap.isOpened():
            raise ValueError("Could not open the video.")
        if not 0 <= frame < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            raise ValueError("This frame is outside the video.")
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        for index in range(frame + 1):
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != index:
                raise ValueError("Could not verify the frame position.")
            ok, image = cap.read()
            if not ok or image is None:
                raise ValueError("Could not read this frame.")
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != index + 1:
                raise ValueError("Could not verify the frame position.")
        if image.shape != (height, width, 3):
            raise ValueError("Frame dimensions do not match the video.")
        return image
    finally:
        cap.release()


def inspector_overlay(raw, rows, enhanced):
    """Draw saved boxes on a display copy at the original resolution."""
    image = enhance(raw.copy()) if enhanced else raw.copy()
    if image is None or image.shape != raw.shape:
        raise ValueError("Enhancement changed the frame dimensions. Choose Raw.")
    height, width = raw.shape[:2]
    for rec in rows:
        x1, y1, x2, y2 = rec["bbox"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("A detection box is outside the video frame.")
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
st.caption("Fish detection, species identification and counting. Runs locally on your computer.")

# Video selection
clips = sorted(p for p in config.CLIPS_DIR.glob("*") if p.suffix.lower() in config.VIDEO_EXTS)
if not clips:
    st.info("No videos found. Add a video to data/clips/ and refresh.")
    with st.expander("Try a demo video"):
        st.code("python scripts/make_test_video.py", language="shell")
    st.stop()

clip_path: Path = st.sidebar.selectbox("Video", clips, format_func=lambda p: p.name)
clip_id = clip_id_from_path(clip_path)
out_dir = config.OUTPUTS_DIR / clip_id

# Reset unsaved form choices when the clip or analysis changes.
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
    st.info("No complete results yet. Click **Run analysis** to start.")
    st.stop()
if failure is not None:
    st.warning("Showing the last complete results because the new analysis failed.")

records = read_jsonl(det_path)
segments = read_segments(out_dir / config.SEGMENTS_FILE)
# TODO(Samkit): confirm the MeanN formula and sampling rules before replacing MaxN.
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


# Results
left, right = st.columns([3, 2])
with left:
    if clip_path.name == "test_clip.mp4":
        st.caption("Synthetic demo video.")
    st.video(str(clip_path))
    st.subheader("Active video sections")
    st.dataframe(segments, width="stretch", hide_index=True, column_config={
        "start_s": "Start (s)", "end_s": "End (s)", "activity_score": "Activity score"})
with right:
    st.subheader("MaxN per species")
    st.dataframe(maxn, width="stretch", hide_index=True, column_config={
        "clip_id": "Video", "species_id": "Species ID", "common_name": "Species",
        "maxn": "MaxN", "frame": "Frame", "time_s": "Time (s)", "n_tracks": "Tracks"})
    st.download_button("Download MaxN CSV", (out_dir / config.MAXN_FILE).read_bytes(),
                       file_name=f"{clip_id}_maxn.csv", mime="text/csv")
    energy_path = out_dir / config.ENERGY_FILE
    if energy_path.exists():
        with open(energy_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            st.metric("Analysis energy (Wh)", f"{float(last['energy_consumed']) * 1000:.3f}")
            st.metric("Emissions (g CO2e)", f"{float(last['emissions']) * 1000:.3f}")

# Frame inspector
st.subheader("Frame inspector")
st.caption("Choose a saved frame to view its detections.")
with st.expander("About the frame inspector"):
    st.write("The frame selector works separately from the video player. Only frames with saved "
             "detections are listed. Later frames may take longer to load because the video is "
             "read from the beginning to check the frame position.")
frame_groups = inspector_frames(records)
if not frame_groups:
    st.info("No saved detections to inspect. Fish may still be present in the video.")
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
        st.warning(f"Frame unavailable: {exc} Detection details are shown below.")
    else:
        st.image(annotated, caption=inspector_frame_label(selected_frame, selected_rows), width="stretch")
    st.caption("Model scores are not calibrated probabilities. Review species below.")
    st.dataframe(inspector_evidence(selected_rows, species_list), width="stretch", hide_index=True,
                 column_config={"Saved species decision": "Species", "Review status": "Review",
                                "Detector model score": "Detection score",
                                "Original species model score": "Original species score"})

# Review
st.subheader("Review each fish")
if st.session_state.pop("review_saved", None) == clip_id:
    st.success("Saved. MaxN recalculated.")
st.caption("Check the tracks you want to save. Original model scores do not measure confidence "
           "in a human correction.")
options = list(species_list) + [config.UNKNOWN_LABEL]
labels = {sid: row["common_name"] for sid, row in species_list.items()}
labels[config.UNKNOWN_LABEL] = "Unknown"

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
                    f"Model suggestions: {suggestions or 'None'}  \n"
                    f"Reviewed: {'yes' if rec['reviewed'] else 'no'}")
        current = rec["species"] if rec["species"] in options else config.UNKNOWN_LABEL
        choices[track_id] = c3.selectbox("Species", options, index=options.index(current),
                                         format_func=lambda s: labels.get(s, s), key=f"{form_key}_species_{track_id}")
        confirmed[track_id] = c3.checkbox("Confirm this track", key=f"{form_key}_confirm_{track_id}")
    if st.form_submit_button("Save review"):
        if not any(confirmed.values()):
            st.info("Check at least one track before saving. No changes saved.")
        else:
            for track_id, species in choices.items():
                if confirmed[track_id]:
                    # A human correction must not replace the original model score.
                    set_track_species(records, track_id, species,
                                      tracks[track_id]["species_conf"], reviewed=True)
            try:
                save_review(clip_id, records)
            except (ContractError, OSError) as exc:
                st.error(f"Could not save review: {exc}")
            else:
                st.session_state.review_saved = clip_id
                st.session_state.review_generation += 1
                st.rerun()
