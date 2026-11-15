# NeuBacH

**Neu**ral **Bac**kground model for Insight-**H**XMT: a physics-informed, data-driven
background model that produces background spectra and light curves for pointed
observations of the HE, ME and LE telescopes.

This repository is being rebuilt as version 2. The design and the literature it
rests on are documented (in Chinese) in:

- [docs/plan.md](docs/plan.md) — technical plan, milestones and open questions
- [docs/literature.md](docs/literature.md) — survey of the HXMT instruments and
  background, background models of other missions, and machine-learning methods

## Status

| Component | State |
| --- | --- |
| Evaluation in the convention of the official background papers (`neubach.eval`) | implemented, tested |
| Activation-state features from the irradiation history (`neubach.physics`) | implemented (fixed time constants), tested |
| Data pipeline, models, product tools | planned, see the milestones in the plan |

## Install and test

```bash
pip install -e ".[dev]"
pytest
```

## Version 1

The first attempt, a transformer regressor over 32 s windows for HE, is kept
unchanged in [`legacy/`](legacy/) for reference. Its fine-tuned model and test
set are on Zenodo: https://zenodo.org/records/17476734
