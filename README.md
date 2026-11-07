# TideSight

Offline fish detection, identification and counting for Coastal Action's underwater video surveys.
Built by Team JPSHG for RBC HubHacks 2026.

**Read `CONTRACT.md` first.** It holds the team's rules: who owns which file, data formats, and when and how we merge.

## Quick start (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Install FFmpeg first (see Synthetic sample playback below).
python scripts\make_test_video.py                     # fake murky test clip -> data\clips\test_clip.mp4
python -m tidesight.pipeline data\clips\test_clip.mp4  # runs triage -> detect -> species -> MaxN
python scripts\validate_outputs.py outputs\test_clip   # checks outputs follow the contract
streamlit run app.py                                  # review app at http://localhost:8501
```

## Synthetic sample playback

The generator requires **FFmpeg with the libx264 encoder**, an external local
video tool. It does not install anything automatically or need model packages
for conversion. Python dependencies and their pins are unchanged.

On Windows, get the release essentials build from
[Gyan's FFmpeg builds](https://www.gyan.dev/ffmpeg/builds/), extract it outside
the repository, and add its `bin` directory to PATH. Open a new terminal and
check `ffmpeg -version` and `ffmpeg -encoders | Select-String libx264`.
Alternatively, pass the executable explicitly:

```powershell
python scripts\make_test_video.py --ffmpeg "C:\Tools\ffmpeg\bin\ffmpeg.exe"
```

Run `python scripts\make_test_video.py` again to replace an older sample. The
script renders the same synthetic scene locally, then converts it to H.264 with
`yuv420p` pixels and `faststart` MP4 metadata for browser loading/seeking. It keeps
`data/clips/test_clip.mp4`, 640×360, 25 fps and 375 frames (15 seconds). Temporary
files are removed; a missing tool or failed conversion exits with guidance and
preserves the previous sample. If FFmpeg is missing, fix PATH or use `--ffmpeg`;
if `libx264` is unavailable, use the essentials build. Close programs locking the
file if replacement fails, then retry. No videos are uploaded.

The extra conversion takes time and uses a second lossy encoding pass, acceptable
for this disposable synthetic fixture. The old MP4V encoding could be decoded by
OpenCV but failed in browsers; `.mp4` names the container, not its video codec.
[Streamlit documents this MP4V limitation and recommends H.264](https://docs.streamlit.io/develop/api-reference/media/st.video).

Start the app, select `test_clip.mp4`, run analysis if no results exist, and refresh
the browser after regenerating. Press Play, confirm motion and a 15-second duration,
then seek forward and backward using the player timeline. A passing pipeline or
codec check alone does not prove playback. This clip and current placeholder
analysis are synthetic; they establish no real detection or species accuracy.

The real-encoding regression test runs when FFmpeg is on PATH or
`TIDESIGHT_TEST_FFMPEG` points to its executable; otherwise that test is explicitly
skipped. The other sample failure tests need no FFmpeg installation.

## How it fits together

```
video ──> enhance()          (display only)
      ──> run_triage()        -> segments.json      busy parts of the video
      ──> run_detection()     -> boxes + track ids  raw frames, 5 fps, only inside segments
      ──> identify_species()  -> species per track  BioCLIP + local species library
      ──> compute_maxn()      -> maxn.csv           MaxN per species
      ──> review screen       -> corrections saved, MaxN recalculated
```

Every step currently runs with a placeholder, so the whole pipeline works from day one.
Each owner replaces the placeholder in their own file (see `TODO(...)` comments).

## Folders

| Folder | What goes there | In git? |
| --- | --- | --- |
| `tidesight/` | the pipeline code | yes |
| `data/clips/` | input videos | no — team drive |
| `data/species_list.csv` | master species list | yes |
| `data/species_library/<species_id>/` | 5–20 reference photos per species (CC-licensed; credits in `CREDITS.csv`) | yes |
| `models/` | model weights | no — team drive |
| `outputs/<clip_id>/` | results per clip | no |
| `docs/` | AI-use guideline, model card, metrics, pitch material | yes |
| `scripts/` | test-video maker, output validator | yes |
| `tests/` | `python -m pytest -q` | yes |
