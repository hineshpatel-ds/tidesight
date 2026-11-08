# Measured results (for the pitch)

Owner: Product + pitch. Every owner sends their numbers by **Sun 3 PM**.
Only numbers we actually measured. Write down which clips and which laptop.

All numbers below come from Coastal Action's survey video **T1-STAT2 411** (Sept 10, station 2;
35.8 min, 4K HEVC, 29.97 fps), the only real clip we have. Fish labels: Samkit's `ml/labels`
(26 hand-labelled frames in two fish events, 13:55–14:30 and 21:25–22:40). Hand counts:
Coastal Action's own annotated sub-sample photos at 14:00 and 22:00.

| Metric | Owner | Value | Measured on (clips, laptop) |
| --- | --- | --- | --- |
| % of footage skipped by triage | Video CV | **13.5%** (1,858 of 2,147 s kept, 55 segments). Fish are on screen most of this clip, so little can be skipped safely. | T1-STAT2 411; Hinesh's laptop (Intel i5-10300H, 16 GB, CPU only); `feat/video` @ b0fe314 |
| Fish missed by triage (should be 0) | Video CV | **0 of 1,446 labelled fish**; no labelled frame dropped. MaxN of both events unchanged (165, 140). The first version (MOG2) missed 566 of 1,446. | Same clip and laptop; labels from `ml/labels` |
| Triage speed | Video CV | 25.6 min for 35.8 min of 4K video (1.4× real time) | Same clip and laptop |
| Detection precision / recall | ML lead | At `DETECT_CONF` 0.18, trained on one event and tested on the other: **precision 0.65, recall 0.48** (tested on 21:25–22:40, 946 fish) and **precision 0.64, recall 0.65** (tested on 13:55–14:30, 500 fish) | T1-STAT2 411, `ml/results/heldout.json` on `feat/detect` (retrain, Sun 10:43) |
| Detector speed (frames per second, CPU) | ML lead | not measured yet | |
| MaxN vs hand count (per clip) | ML lead | Coastal Action counted **102** fish at 22:00; model **48**. They counted **36** at 14:00; model **32**. Total count over each held-out event: −26% and +3% vs labels. Undercounts in dense, distant schools. | Same held-out models as above; hand counts from Coastal Action's sub-sample photos |
| Species top-1 / top-3 accuracy | Species ID | not measured: there is no species-labelled footage yet | |
| Share of fish marked "unknown" | Species ID | not measured | |
| Energy per hour of video (Wh) and CO2e (g) | Product + pitch | not measured yet: every full pipeline run writes `outputs/<clip_id>/emissions.csv` (CodeCarbon, offline, Canada grid) | |
| Minutes of footage a person must review, before vs after | Product + pitch | Before: the whole video in real time (35.8 min) plus hand-counting 15 sub-sample frames. After, MaxN mode: 31.0 min kept by triage on this busy clip. After, Protocol mode: the 15 frames arrive already counted; the person checks and corrects | T1-STAT2 411 |

## How to read these honestly

- One clip, one station, one day. Other stations (clearer or darker water) are untested.
- Detection numbers are held-out: the model never saw the event it was scored on. They are
  from 26 labelled frames, so treat them as a first estimate, not a benchmark.
- Triage is tuned to never drop fish (recall first). On quieter footage it will skip more;
  on this clip there is almost always something moving.
- Species ID has no measured accuracy. Coastal Action's protocol already has an
  "Unknown/Unable to ID due to visibility" column; TideSight uses "unknown" the same way.
