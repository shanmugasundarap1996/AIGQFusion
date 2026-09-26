# AIGQFusion code

Code-only reproducibility package for the **ClinVar-MVE** study.

- `baseline_models.py` — baseline benchmark driver.
- `aigqfusion.py` — proposed **AIGQFusion (A5)** with the five-qubit **BioVQC** route.
- `data_protocol.py` — fold-local preprocessing and group-disjoint CV helpers.
- `config.py` — frozen ClinVar-MVE predictor/view definitions and protocol constants.

Place the released dataset as `ClinVar-MVE.csv` in the repository root, or pass another path with `--data`.

```bash
python baseline_models.py --data ClinVar-MVE.csv
python aigqfusion.py --data ClinVar-MVE.csv
```

The source files contain implementation logic only; manuscript result tables and reported scores are not hard-coded.
