# Multi-head skin-permeability predictor

Standalone inference package. Python 3.11 or 3.12; CUDA is used when available, otherwise CPU. Run all commands from this directory.

## Quick start

```bash
python -m pip install -r requirements.txt
sha256sum -c weights.sha256
python predict.py --input examples/molecules.csv --output predictions.csv
```

Input is a CSV with a `SMILES` column. Other columns are preserved. The default condition is `Kp` / `Human`.

```bash
# Choose an endpoint and animal
python predict.py --input molecules.csv --output predictions.csv \
  --metric Jss --animal Pig

# Set a stricter structural AD cutoff and export only in-domain rows
python predict.py --input molecules.csv --output predictions_ad.csv \
  --metric Kp --animal Human --ad-min-tanimoto 0.30 --ad-only-in-domain

# Optional combined structural + model-representation screen
python predict.py --input molecules.csv --output predictions_combined.csv \
  --ad-rule combined --ad-min-tanimoto 0.25 \
  --ad-max-nn-ratio 1.2 --ad-max-q-ratio 1.2
```

AD is available only for `Kp` / `Human`. The default rule is `--ad-rule structure`: a molecule is in-domain when its **maximum Morgan radius-2, 2048-bit Tanimoto similarity to the training structures is ≥ 0.25**. `--ad-min-tanimoto` accepts a value from 0 to 1. `--ad-rule combined` also requires the two model-representation ratios to be below their limits (default 1 each); `--ad-rule none` skips AD. By default, all rows are retained with `ad_status` and `ad_in_domain`; `--ad-only-in-domain` writes only passing rows.

The output includes `regression_z_prediction`, `direction_score`, `prediction_status`, and, for Kp/Human, AD scores and flags. `regression_z_prediction` is unitless; `direction_score` has an endpoint-specific scale. Run `python predict.py --help` for all options. Invalid SMILES and molecules exceeding the encoder's 202-token limit have no prediction.
