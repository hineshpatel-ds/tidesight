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
from tidesight.contracts import (clip_id_from_path, load_species_list, read_jsonl,
                                 read_maxn_csv, read_segments, set_track_species)
from tidesight.enhance import enhance
from tidesight.pipeline import run_pipeline, save_review

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

if st.sidebar.button("Run analysis", type="primary"):
    with st.spinner("Analysing video..."):
        result = run_pipeline(clip_path)
    st.sidebar.success(f"Done: {result['n_tracks']} tracks, {result['skipped_pct']}% of video skipped")

det_path = out_dir / config.DETECTIONS_FILE
if not det_path.exists():
    st.info("Click **Run analysis** to process this video.")
    st.stop()

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

# ------------------------------------------------------------------ review
st.subheader("Review each fish")
options = list(species_list) + [config.UNKNOWN_LABEL]
labels = {sid: row["common_name"] for sid, row in species_list.items()}
labels[config.UNKNOWN_LABEL] = "Unknown (expert review)"

tracks: dict[int, dict] = {}
for rec in records:
    best = tracks.get(rec["track_id"])
    if best is None or rec["det_conf"] > best["det_conf"]:
        tracks[rec["track_id"]] = rec

with st.form("review"):
    choices: dict[int, str] = {}
    for track_id, rec in sorted(tracks.items()):
        c1, c2, c3 = st.columns([1, 2, 2])
        thumb = track_thumbnail(str(clip_path), rec["frame"], tuple(rec["bbox"]))
        if thumb is not None:
            c1.image(thumb, caption=f"Track {track_id} at {rec['time_s']} s")
        suggestions = ", ".join(f"{labels.get(t['species'], t['species'])} {t['score']:.2f}" for t in rec["top3"])
        c2.markdown(f"**Track {track_id}**  \nSuggestions: {suggestions or '-'}  \n"
                    f"Reviewed: {'yes' if rec['reviewed'] else 'no'}")
        current = rec["species"] if rec["species"] in options else config.UNKNOWN_LABEL
        choices[track_id] = c3.selectbox("Species", options, index=options.index(current),
                                         format_func=lambda s: labels.get(s, s), key=f"track_{track_id}")
    if st.form_submit_button("Save review"):
        for track_id, species in choices.items():
            set_track_species(records, track_id, species, 1.0, reviewed=True)
        save_review(clip_id, records)
        st.success("Saved. MaxN recalculated.")
        st.rerun()
