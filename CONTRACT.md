# TideSight — Team Rules (everyone follows these)

Team JPSHG · RBC HubHacks 2026 · Coastal Action. All times are Atlantic Time.
These are decisions, not suggestions. To change one, post in the team chat first; the App + integration owner makes the change.

---

## 1. Who owns what

You edit **only your own files**. Anyone may add a test file named `tests/test_<your_area>.py`.

| Role | Branch | Files you own |
| --- | --- | --- |
| Video + classical CV | `feat/video` | `tidesight/enhance.py`, `tidesight/triage.py` |
| ML lead | `feat/detect` | `tidesight/detect.py`, `tidesight/count.py` |
| Species ID | `feat/species` | `tidesight/species.py`, `data/species_list.csv`, `data/species_library/` |
| App + integration | `feat/app` | `app.py`, `tidesight/pipeline.py`, `tidesight/contracts.py`, `tidesight/config.py`, `requirements.txt`, `scripts/` |
| Product + pitch | `feat/pitch` | `docs/` |

Every file already runs with a **placeholder**. Replace the placeholder in your file; keep the function name and inputs/outputs exactly the same.

---

## 2. Setup (same for everyone)

- **Python 3.12**, virtual environment in `.venv`, packages from `requirements.txt` (pinned versions). Only the App owner edits `requirements.txt`; ask in chat to add a package.
- **Shared values** (paths, thresholds, file names) live in `tidesight/config.py`. Import them; never hard-code them.
- **Videos and model weights are never committed to git.** Share them on the team drive. Put videos in `data/clips/`, weights in `models/`.
- **Video names:** lowercase, underscores, no spaces, e.g. `mahone_bay_2024_07_12.mp4`. The `clip_id` is the file name without the extension.

---

## 3. Data rules

| Topic | Rule |
| --- | --- |
| Frame numbers | 0-based, counted at the video's **original** fps. Never renumber sampled frames 0, 1, 2… |
| Time | `time_s = round(frame / fps, 3)` |
| Detector sampling | Standard analysis uses about 5 fps on the global `frames_to_process()` grid. Protocol sampling uses exactly one nearest original frame for each requested point, as defined below. |
| Raw vs enhanced | Detector decodes original **raw** BGR frames and may apply its configured model-specific preprocessing internally. It must not call display `enhance()`. Species crops still come from original **raw** frames. `enhance()` is for review and demo display (triage may use either). |
| Box format | `[x1, y1, x2, y2]`, integer pixels, **original resolution**, origin top-left. `enhance()` must return the same image size. |
| Species | One species per **track**, set with `set_track_species()`. Allowed values: a `species_id` from `data/species_list.csv`, `"unknown"`, or `"fish"` (before the species step). |
| Unknown | `species = "unknown"` when the best score < `UNKNOWN_THRESHOLD` (0.5). `unknown` is true exactly then. |
| Reviewed | Only the review screen sets `reviewed = true`. |
| Writing files | Only `tidesight/pipeline.py` writes to `outputs/`. Your functions return data; they never write files. |
| Internet | No cloud AI anywhere (no OpenAI, Gemini, Claude, Roboflow-hosted or any other hosted API). No uploading clips. Models are downloaded once, then run offline. |
| Logging | Use `logging.getLogger(__name__)`, not `print`, inside `tidesight/`. |

Detector preprocessing is versioned by `config.DETECT_PREPROCESS`. The proposed
`stretch1-99_clahe3-8x8_v1` recipe resizes to `DETECT_INPUT_WIDTH`, stretches each
BGR channel using its 1st/99th percentiles, then applies LAB lightness CLAHE with
clip limit 3 and an 8-by-8 grid. The ML implementation must match its training
recipe; training equivalence and model accuracy require separate evidence.
Resizing is internal: return bounding boxes mapped back to original-resolution
integer pixels, retaining original frame numbers and timestamps. Never pass
detector-preprocessed frames to the species model or overwrite the source video.

### Protocol sampling (ML interface; app integration pending)

The ML owner's supplied interpretation of Coastal Action's **DRAFT FFH Video
Analysis Protocol** uses 15 samples, 60 seconds apart, discarding the first and
last 60 seconds. The original draft must be checked before claiming protocol
compliance. Shared defaults are `PROTOCOL_N_SAMPLES`, `PROTOCOL_INTERVAL_S` and
`PROTOCOL_EDGE_S` in `config.py`.

`run_protocol_samples(video_path, start_s)` samples at `start_s + i * interval_s`
for indices 0 through 14, independently of activity triage. Reject a start inside
the first excluded minute or a last requested point inside the final excluded
minute; never shorten the denominator or pad missing footage with zeros.
Choose the nearest original frame, rounding half-frame ties upward. Requested
times are retained as sample `time_s`; each detection retains its actual original
frame and `round(frame / fps, 3)` timestamp. These times may differ slightly.

The returned `ProtocolRun` includes source metadata, model/preprocessing settings
and an ordered list of `ProtocolSample` objects (`index`, requested `time_s`,
original `frame`, completion `status`, `records`). A requested frame must decode
and finish inference before its status is `"done"`. A failed frame raises an
error for the whole run. Tracks expire across unseen gaps over
`TRACK_MAX_GAP_S`, with track IDs remaining unique within the clip.

`compute_meancount(run)` consumes completed samples from that run. It counts
only the records in each sample's single frame, including completed empty
samples as zero. MeanCount total and each species mean use all 15 samples as
the denominator. Missing, incomplete or inconsistent samples are errors, never
zero detections. It returns data without writing files. MaxN remains a separate
measure and its existing computation and CSV schema are unchanged.

The app currently publishes standard MaxN analyses. Protocol persistence,
review/recount, display/export and validator schemas require separate integration;
the new ML interfaces do not add output files or permit modules other than the
pipeline to write them.

### The five functions (do not change their names or inputs)

| Function | Owner | Returns |
| --- | --- | --- |
| `enhance(frame)` | Video CV | BGR image, same shape |
| `run_triage(video_path)` | Video CV | `list[Segment]` |
| `run_detection(video_path, segments)` | ML lead | `list[Record]` (species fields at defaults) |
| `identify_species(video_path, records)` | Species ID | the same records with species fields filled |
| `compute_maxn(records)` | ML lead | `list[MaxNRow]` (already complete) |

### Output files (per clip, in `outputs/<clip_id>/`)

| File | Content |
| --- | --- |
| `segments.json` | list of `{"start_s", "end_s", "activity_score"}`, sorted, no overlaps, padded 1 s, gaps under 2 s merged |
| `detections.jsonl` | one line per fish per frame: `clip_id, frame, time_s, bbox, track_id, det_conf, species, species_conf, top3, unknown, reviewed` |
| `maxn.csv` | `clip_id, species_id, common_name, maxn, frame, time_s, n_tracks` |
| `emissions.csv` | energy and CO2e from CodeCarbon |

`tidesight/contracts.py` defines all of this in code. If `python scripts\validate_outputs.py outputs\<clip_id>` says **OK**, your work fits.

### Protocol app mode

The app also offers Samkit's merged 15-point sampling API: choose the first sample,
then sample one original frame every 60 seconds, excluding 60 seconds at each end.
This follows the merged implementation; the original draft has not been independently checked.
Protocol runs bypass triage and start species as Unknown for manual track review.
MaxN mode keeps its existing behavior.

The pipeline additionally writes `protocol.json` (completed samples, detection records,
source fps/frame count/dimensions and detector settings), `counts.json` (MeanCount
and per-sample counts), and `sample_counts.csv` (index, requested time, original
frame, fish count and species counts). Exactly 15 completed samples are required;
completed empty samples count as zero, failed or missing samples invalidate the run.
Review and recount update all these files together through staged publication.
The app and output validator check saved counts and CSV against sample evidence.
MeanCount is the average detections per sampled frame, not unique fish in the clip.

---

## 4. Daily rhythm

- **Stand-ups (5 min):** 10 AM, 4 PM, 9 PM — done, next, blocked.
- **Numbers for the pitch:** each owner adds measured results to `docs/METRICS.md` (via the pitch owner) by **Sun 3 PM**.

---

## 5. How we merge

### Merge windows

Merge only in these windows, in this order: **Video → ML → Species → App → Pitch**. Data flows in that order, so each next person tests against the real code before theirs.

| When | What gets merged |
| --- | --- |
| **Fri 5:30 PM** | App owner pushes this starter to `main`. Everyone clones and creates their branch. |
| **Fri 11 PM** | First real version of each part, even if rough. Not ready? Skip; the placeholder stays. |
| **Sat 11 AM** | Working versions of every part. |
| **Sat 12 PM** | **Integration call (everyone):** run the full pipeline on a real clip together and fix format mismatches. |
| **Sat 6 PM** | Improvements; review screen. |
| **Sat 10 PM** | Everything needed for the demo; mentor feedback. |
| **Sun 1 PM** | Last features. |
| **Sun 3 PM** | **Feature freeze.** After this: bug fixes only, one small pull request each, approved by the App owner. |
| **Sun 7:30 PM** | App owner tags `v1.0-submission`. Pitch owner submits by **8 PM** (deadline 11:59 PM, no late submissions). |

### Steps for each person at each window (PowerShell)

```powershell
# 1. Save your work
git add -A
git commit -m "detect: real tracking on sampled frames"

# 2. Bring in everything already merged
git fetch origin
git merge origin/main

# 3. Prove nothing broke
python -m pytest -q
python -m tidesight.pipeline data\clips\test_clip.mp4 --no-energy
python scripts\validate_outputs.py outputs\test_clip
#    (also run on one real clip if you have one)

# 4. Push and open a pull request
git push -u origin feat/detect
```

Then on GitHub: **Pull requests → New pull request → base: `main`, compare: your branch**. Paste the validator's `OK` line in the description and post "PR ready" in the team chat.

### What the App owner does with each pull request

1. Check that only the author's own files changed (section 1).
2. Check that the description shows the validator `OK` and the tests pass.
3. Click **Merge pull request** (merge commit), then post "merged feat/…" in chat. The next person in the order goes.

### After each window, everyone updates

```powershell
git switch feat/detect
git fetch origin
git merge origin/main
```

### Rules that keep merging painless

- Never commit directly to `main`. Never `git push --force` on `main`.
- A merge must never break `python -m tidesight.pipeline`. If your part isn't working, don't merge it; the placeholder stays until the next window.
- A merge conflict means someone edited a file they don't own. Stop and fix it together on a call; don't guess.
- When the App owner changes `contracts.py`, `config.py` or `requirements.txt`, everyone merges `main` immediately and runs `pip install -r requirements.txt`.

---

## 6. One-time setup

**App owner (once, by Fri 5:30 PM):** create an empty GitHub repo called `tidesight`, add the four teammates as collaborators, then, in the unzipped starter folder:

```powershell
git init
git add -A
git commit -m "Starter: team contract, placeholders, pipeline, app"
git branch -M main
git remote add origin https://github.com/<your-username>/tidesight.git
git push -u origin main
```

Optional but recommended: in GitHub, **Settings → Branches → add a rule for `main` → require a pull request before merging**.

**Everyone (once):**

```powershell
git clone https://github.com/<your-username>/tidesight.git
cd tidesight
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt       # takes a few minutes (PyTorch is large)
python scripts\make_test_video.py
python -m pytest -q
python -m tidesight.pipeline data\clips\test_clip.mp4
git switch -c feat/detect             # your own branch name from section 1
```

**ML lead:** download the detector weights into `models/` (and share them on the team drive):

```powershell
Invoke-WebRequest -Uri "https://github.com/WildHackers/community-fish-detector/releases/download/cfd-1.00-yolov12x/cfd-yolov12x-1.00.pt" -OutFile models\cfd-yolov12x-1.00.pt
```

**Species ID:** BioCLIP downloads automatically the first time (needs internet once). For the demo, run offline:

```powershell
$env:HF_HUB_OFFLINE = "1"
streamlit run app.py
```
