# How the pieces fit together

```
WM-811K (811k wafers)
  └─ splits/wafer_split_v1.csv.gz      one lot-grouped split for everyone (train / val / test)
       ├─ CNN  (wafer.infer, models/v0-ens3)        primary classifier, calibrated, with needs_review
       ├─ YOLO-cls (wafer.classifiers.yolo)         experimental second opinion (uncalibrated classify)
       └─ combine step                              not enabled — YOLO split overlaps demo; not fair yet
            └─ factory stream (wafer.factory)       lots on tools over time; alert rule; impact; root causes
                 └─ agent (wafer.agent)             LLM on AMD Token Factory, calls the tools above
                      └─ dashboard (dashboard/)     operations view, wafer inspector, copilot, model quality
                           └─ engineer              hold lots, open ticket, dismiss
```

## Who owns what

| Piece | Owner | Contract other pieces rely on |
|---|---|---|
| Data, split, CNN, calibration, report | Akein (`akein/cnn-model`) | `wafer.infer.load_model(dir).predict(map)` and the split file |
| YOLO-cls (experimental) | teammate / integration | `get_classifier("yolo")` → common keys; uncalibrated; `models/yolo-cls-exp`; not trained on shared split |
| Factory stream, alerts, impact, root causes, agent | teammate (`nsr`) | `check_alert`, `impact_from_alert`, `lookup_root_cause`, `WaferAgent.ask` |
| Dashboard | Akein (`akein/dashboard`) | reads only the modules above |

## CNN + YOLO (current honesty rules)

The checked-in YOLO-cls checkpoint used a **private stratified sample**, not
`splits/wafer_split_v1.csv.gz`. It **overlaps** the val demo stream. Until it is
retrained on the shared train split only:

1. Treat YOLO as **experimental** — uncalibrated softmax; no CNN τ/θ; `needs_review` always false.
2. Do **not** claim fair comparison, held-out performance, or independent confirmation of CNN alerts.
3. Score a **separate** stream (`streams/stream_val_yolo.csv`); the dashboard agent binds to the
   stream's `classifier` column so history matches predictions.
4. **Do not enable a combine rule** until YOLO is retrained on shared train IDs and re-scored on val.

When a fair YOLO exists later: export `uid,pred` → `python -m wafer.evaluate --predictions …`,
then consider calibrated combine + `models_disagree` review on val only.

## Rules that keep the numbers honest

- The test split is scored once, at the end, with `--final`. Until then, demos stream val.
- No one tunes anything on test, including thresholds and the combine rule.
