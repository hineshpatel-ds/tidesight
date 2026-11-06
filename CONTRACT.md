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
| Detector sampling | 5 frames per second: every `round(fps / 5)`th frame, on one global grid (`frames_to_process()` does this) |
| Raw vs enhanced | Detector and species model read **raw** frames. `enhance()` is for the review screen and the demo (triage may use either). |
| Box format | `[x1, y1, x2, y2]`, integer pixels, **original resolution**, origin top-left. `enhance()` must return the same image size. |
| Species | One species per **track**, set with `set_track_species()`. Allowed values: a `species_id` from `data/species_list.csv`, `"unknown"`, or `"fish"` (before the species step). |
| Unknown | `species = "unknown"` when the best score < `UNKNOWN_THRESHOLD` (0.5). `unknown` is true exactly then. |
| Reviewed | Only the review screen sets `reviewed = true`. |
| Writing files | Only `tidesight/pipeline.py` writes to `outputs/`. Your functions return data; they never write files. |
| Internet | No cloud AI anywhere (no OpenAI, Gemini, Claude, Roboflow-hosted or any other hosted API). No uploading clips. Models are downloaded once, then run offline. |
| Logging | Use `logging.getLogger(__name__)`, not `print`, inside `tidesight/`. |

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
