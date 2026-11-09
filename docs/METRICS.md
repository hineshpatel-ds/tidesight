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
| Detector speed (frames per second, CPU) | ML lead | About **1 frame/s** end to end: a Protocol run (15 frames of 4K) took 16 s, including seeking, decoding and the 1920-px preprocessing | T1-STAT2 411; Hinesh's laptop (i5-10300H, CPU only); `model-v1` release weights |
| MaxN vs hand count (per clip) | ML lead | Coastal Action counted **102** fish at 22:00; model **48**. They counted **36** at 14:00; model **32**. Total count over each held-out event: −26% and +3% vs labels. Undercounts in dense, distant schools. The shipped model (`model-v1`) in Protocol mode counts **102 at 22:00**, the same as Coastal Action. That exact frame was kept out of training, but other frames of the same school were trained on, so this is optimistic; use the held-out 48 when quoting accuracy. | Same held-out models as above; hand counts from Coastal Action's sub-sample photos; Protocol run starting at 8:00 |
| Species top-1 / top-3 accuracy | Species ID | Not measured: there is no species-labelled footage. BioCLIP zero-shot was run on all 1,446 labelled fish and was clearly wrong: it called 47% of the fish "American eel" and 52 of them "spotfin butterflyfish" (a tropical alert species). So the model is switched off. | T1-STAT2 411, crops from the original 4K frames |
| Share of fish marked "unknown" | Species ID | 100% by design: every fish starts "unknown" and the reviewer assigns species, as in Coastal Action's "Unknown/Unable to ID" column | `feat/species-bioclip` |
| Energy per hour of video (Wh) and CO2e (g) | Product + pitch | **0.27 Wh and 0.05 g CO2e** for one Protocol analysis (15 samples) of the 35.8-min video. That cost depends on the 15 frames, not on the video length. Full-footage MaxN mode was not run: triage alone takes 26 min, and detection on ~9,300 frames at ~1 frame/s would take about 2.5 h on this CPU (estimate) | T1-STAT2 411; Hinesh's laptop; CodeCarbon offline, Canada grid (estimate based on CPU power rating) |
| Minutes of footage a person must review, before vs after | Product + pitch | Before: the whole video in real time (35.8 min) plus hand-counting 15 sub-sample frames. After, MaxN mode: 31.0 min kept by triage on this busy clip. After, Protocol mode: the analysis takes 16 s, and the 15 frames arrive already counted (MeanCount 8.9 fish per frame, starting at 8:00), so the person checks and corrects | T1-STAT2 411 |

## How to read these honestly

- One clip, one station, one day. Other stations (clearer or darker water) are untested.
- Detection numbers are held-out: the model never saw the event it was scored on. They are
  from 26 labelled frames, so treat them as a first estimate, not a benchmark.
- Triage is tuned to never drop fish (recall first). On quieter footage it will skip more;
  on this clip there is almost always something moving.
- Species ID has no measured accuracy. Coastal Action's protocol already has an
  "Unknown/Unable to ID due to visibility" column; TideSight uses "unknown" the same way.
