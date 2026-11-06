# TideSight

Offline fish detection, identification and counting for Coastal Action's underwater video surveys.
Built by Team JPSHG for RBC HubHacks 2026.

**Read `CONTRACT.md` first.** It holds the team's rules: who owns which file, data formats, and when and how we merge.

## Quick start (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\make_test_video.py                     # fake murky test clip -> data\clips\test_clip.mp4
python -m tidesight.pipeline data\clips\test_clip.mp4  # runs triage -> detect -> species -> MaxN
python scripts\validate_outputs.py outputs\test_clip   # checks outputs follow the contract
streamlit run app.py                                  # review app at http://localhost:8501
```

The test clip is MPEG-4 Part 2 and may not play in the browser; real GoPro clips (H.264) play fine.
Everything else in the app works with the test clip.

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
