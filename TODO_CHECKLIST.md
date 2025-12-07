# Project Checklist

## Setup & Infrastructure
- [x] Create GitHub repo + README
- [x] Create branch task-1 & commit 3x/day
- [ ] DVC init + add raw data + push
- [x] Data cleaning script + tests
- [x] DVC pipeline (dvc.yaml) + reproducible commands
- [x] CI workflow (pytest + lint)

## EDA & Analysis
- [x] EDA notebook template
- [ ] EDA notebook + 3 plots (export PNGs) - *Run when data available*
- [ ] Hypothesis tests + statistical method notes - *Run when data available*
- [x] Loss ratio calculation by Province
- [x] Low-risk group identification

## Modeling
- [ ] Per-zipcode linear models (baseline) - *Optional*
- [x] Global ML pipeline (frequency × severity recommended)
- [x] Feature importance (SHAP)
- [x] Premium optimization rule + simulation
- [x] Low-risk recommendations with financial impact

## Reports & Documentation
- [ ] Interim report + submit by 2025-12-07 20:00 UTC
- [ ] Final report + slides + merge to main by 2025-12-09 20:00 UTC
- [x] A/B test plan (sample size, metrics, guardrails) - *In recommendations module*

## Timeline (2025-12-07)

### Now → 2 hours (Immediate)
- [x] Init repo, basic README
- [ ] DVC add raw data
- [ ] Run quick EDA (loss ratio by Province)
- [ ] Commit

### Next 4–6 hours (Before interim deadline)
- [ ] Produce 3 plots + hypothesis test scripts
- [ ] Interim report (pack into reports/interim/)
- [ ] Push DVC
- [ ] Create short slide with 3 key insights + 3 plots
- [ ] Submit by 20:00 UTC

### Final (2025-12-08 → 2025-12-09)
- [ ] Finish models
- [ ] Complete DVC pipeline runs
- [ ] Write final 3-page report
- [ ] Create slides
- [ ] Merge to main
- [ ] Submit by 20:00 UTC

## Notes

- Always set `random_state=42` everywhere
- Log steps and decisions in README
- Use stratified sample for CI runs if dataset is big
- Full DVC pipeline for final runs
- Use joblib.dump for models and version with DVC if large
- Prioritize SHAP for feature importance (business explainability)

