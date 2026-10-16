# How the pieces fit together

```
WM-811K (811k wafers)
  └─ splits/wafer_split_v1.csv.gz      one lot-grouped split for everyone (train / val / test)
       ├─ CNN  (wafer.infer, models/v0-ens3)        primary classifier, calibrated, with needs_review
       ├─ YOLO (wafer.classifiers.yolo)             second opinion: pattern + boxes
       └─ combine step                              agree → trust; disagree → needs_review
            └─ factory stream (wafer.factory)       lots on tools over time; alert rule; impact; root causes
                 └─ agent (wafer.agent)             LLM on AMD Token Factory, calls the tools above
                      └─ dashboard (dashboard/)     operations view, wafer inspector, copilot, model quality
                           └─ engineer              hold lots, open ticket, dismiss
```

## Who owns what

| Piece | Owner | Contract other pieces rely on |
|---|---|---|
| Data, split, CNN, calibration, report | Akein (`akein/cnn-model`) | `wafer.infer.load_model(dir).predict(map)` and the split file |
| YOLO | teammate | `get_classifier("yolo").classify(map)` returns the same fields as the CNN backend |
| Factory stream, alerts, impact, root causes, agent | teammate (`nsr`) | `check_alert`, `impact_from_alert`, `lookup_root_cause`, `WaferAgent.ask` |
| Dashboard | Akein (`akein/dashboard`) | reads only the modules above |

## CNN + YOLO

1. **Same split.** YOLO must be trained on the `train` rows of `splits/wafer_split_v1.csv.gz` only.
   If it was trained on all 811k wafers (or a random split), it has seen our val/test wafers
   and its scores are inflated; retrain before comparing.
2. **Same scoring.** Export YOLO's predictions on val as `uid,pred` and run
   `python -m wafer.evaluate --predictions yolo.csv` for macro-F1 with lot-bootstrap CIs.
3. **Combine.** Average the two calibrated probability vectors; if the argmax of the two
   models differs, set `needs_review` with reason `models_disagree`. Measure on val:
   agreement rate, accuracy when they agree, and how many errors the disagreement flag catches.
4. **Dashboard and agent** pick YOLO up automatically once the adapter in
   `wafer/classifiers/yolo.py` returns results (the inspector shows both opinions and an
   agree/disagree badge).

## Rules that keep the numbers honest

- The test split is scored once, at the end, with `--final`. Until then, demos stream val.
- No one tunes anything on test, including thresholds and the combine rule.
