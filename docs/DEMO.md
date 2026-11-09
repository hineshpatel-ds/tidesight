# Demo and pitch plan

For tonight (Sun Nov 8). Hard deadline 11:59 PM, no late submissions. Numbers: `docs/METRICS.md`.

## 1. Merge, then tag (Gavin)

Merge in this order. Merged together they have no conflicts and the tests pass (251 passed, 1 skipped).

| Branch | What it does |
| --- | --- |
| `feat/video` (PR #3) | Triage that misses none of the 1,446 labelled fish (the old version missed 566). Fast-forward |
| `feat/detect` | Samkit's retrained held-out results (results file only) |
| `feat/species-bioclip` | Every fish starts "unknown" instead of fake species names. BioCLIP is in the code but off: it was measured wrong on our clip |
| `fix/app-inspector-seek` | The app can open the real 4K video: the frame inspector seeks instead of decoding from frame 0. Without it the app times out |
| `feat/pitch` | METRICS and this file |

Then tag `v1.0-submission` and submit.

## 2. Set up the demo laptop (at least 30 min before)

1. `git pull` on `main`, then `pip install -r requirements.txt`.
2. Download the weights from the GitHub release `model-v1` (`tidesight-fish-v1.pt`, 19 MB) into `models/`.
3. Copy Coastal Action's video to `data/clips/t1_stat2_411.mp4`. The name must be lowercase with no spaces. `.gitignore` keeps it out of git; never add it by hand.
4. Run `streamlit run app.py`. Pick `t1_stat2_411.mp4`, Counting method **Protocol (15 samples)**, First sample **480**, then **Run analysis**. It takes 16 s on an i5 laptop.
5. **Let the page finish loading once (about 2 min).** It cuts a thumbnail for each of the 133 fish. They are cached afterwards, and clicks then take 2–12 s.
6. Turn Wi-Fi off and reload. It must still work: that proves "nothing leaves the laptop". Practise this before the pitch, not during it.
7. **Record a 2-minute screen video of the demo** as a backup.
8. Laptop plugged in, notifications off, browser zoomed so tables are readable.

Don't run MaxN mode on the full video live. Triage alone takes 26 min, and detection would take hours on a CPU.

## 3. Live demo script (about 2 min)

1. "This is Coastal Action's real survey video: Sept 10, station 2, 36 minutes, 4K, murky water."
2. Show the Protocol run: "15 frames, one every 60 s, exactly like their FFH protocol. 16 seconds on this laptop, no internet."
3. **Protocol counts**: MeanCount 8.9 fish per frame. Point at sample 15 (22:00): **102 fish. Coastal Action's volunteer counted 102 on this frame.**
4. **Frame inspector**: pick sample 15 to show the boxes, then switch to **Enhanced**. That is the view a person uses to check fish in murky water.
5. **Review**: species start as Unknown, like the "Unknown/Unable to ID" column on their sheet. Set one track's species, tick Confirm, then Save review. Counts are recalculated.
6. Point at **Analysis energy: 0.27 Wh**, then **Download sample counts CSV** ("goes straight into their spreadsheet").

## 4. Pitch outline (fit it to the time limit)

| # | Part | Who | Point |
| --- | --- | --- | --- |
| 1 | Problem | anyone | Volunteers watch hours of murky video and count fish by hand. Cloud AI isn't allowed. |
| 2 | Solution | Gavin | TideSight runs on their laptop with open-source models and uploads nothing. It follows their protocol (15 samples, MeanCount) and also has MaxN. |
| 3 | Live demo | Gavin | Section 3 |
| 4 | Detection | Samkit | Trained on their footage. Held-out precision about 0.65, recall 0.48–0.65. Undercounts dense, distant schools, so a person reviews. |
| 5 | Triage + enhance | Hinesh | Skips empty footage without missing fish: 0 of 1,446 labelled fish missed (the first version missed 566). Enhanced view for review. |
| 6 | Species, honestly | Prabhat | We tested BioCLIP: it called 47% of the fish "American eel" and raised false tropical-species alerts. So we ship "unknown" plus human review. |
| 7 | Next steps | anyone | Reviewed corrections become training data, then a species model trained on their fish. More stations. A 12 GB GPU for full-video MaxN. |

## 5. Likely questions, honest answers

- **"102 = 102, is that real?"** That frame was kept out of training, but other frames of the same school were trained on, so it's optimistic. When the model never saw that school, it counted 48. Dense, distant schools get undercounted, which is why there's a review step.
- **"How accurate is it?"** Held-out precision 0.64–0.65 and recall 0.48–0.65, from one video and 26 labelled frames. That's a first estimate, not a benchmark.
- **"Why no species ID?"** It was measured wrong on their water (see above). A confident wrong label or a false invasive-species alert does more harm than "unknown". The code is ready to switch on once it's trained on their reviewed fish.
- **"How long for a whole video?"** Protocol mode: 16 s. Full-footage MaxN mode: triage 26 min, plus detection estimated at about 2.5 h on a laptop CPU. A GPU should cut that a lot, but we haven't measured it.
- **"Does it need internet?"** No. Models are downloaded once, and nothing is uploaded.
- **"Other stations?"** Not tested; only this clip had labels. It's the first thing to do with more of their footage.
