"""CSV inference for the frozen SMI-TED multi-head skin model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from ad import KpHumanAD
from model import SkinEnsemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with a SMILES column")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="Kp", choices=(
        "Jmax", "Jss", "Kp", "Percent absorbed", "Tlag"
    ))
    parser.add_argument("--animal", default="Human")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--ad-rule", choices=("structure", "combined", "none"), default="structure",
        help="Kp/Human AD: default train-calibrated structural rule; combined is exploratory",
    )
    parser.add_argument(
        "--ad-min-tanimoto", type=float, default=None,
        help="minimum nearest-train Tanimoto (0–1); default is 0.25",
    )
    parser.add_argument(
        "--ad-max-nn-ratio", type=float, default=1.0,
        help="maximum model-representation nearest-neighbor ratio (combined rule only)",
    )
    parser.add_argument(
        "--ad-max-q-ratio", type=float, default=1.0,
        help="maximum model-representation PCA residual ratio (combined rule only)",
    )
    parser.add_argument(
        "--ad-only-in-domain", action="store_true",
        help="write only Kp/Human rows passing the chosen AD rule; default retains all rows",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different files")
    if args.ad_min_tanimoto is not None and (
        not math.isfinite(args.ad_min_tanimoto)
        or not 0 <= args.ad_min_tanimoto <= 1
    ):
        parser.error("--ad-min-tanimoto must be finite and between 0 and 1")
    for option, value in (("--ad-max-nn-ratio", args.ad_max_nn_ratio),
                          ("--ad-max-q-ratio", args.ad_max_q_ratio)):
        if not math.isfinite(value) or value < 0:
            parser.error(f"{option} must be finite and nonnegative")
    calibrated_condition = args.metric == "Kp" and args.animal == "Human"
    if not calibrated_condition and (
        args.ad_only_in_domain or args.ad_min_tanimoto is not None
        or args.ad_max_nn_ratio != 1.0 or args.ad_max_q_ratio != 1.0
        or args.ad_rule == "combined"
    ):
        parser.error("AD thresholds and filtering are calibrated only for Kp/Human")
    if args.ad_rule == "none" and args.ad_only_in_domain:
        parser.error("--ad-only-in-domain requires --ad-rule structure or combined")
    if args.ad_rule == "none" and (
        args.ad_min_tanimoto is not None
        or args.ad_max_nn_ratio != 1.0 or args.ad_max_q_ratio != 1.0
    ):
        parser.error("AD thresholds require --ad-rule structure or combined")

    frame = pd.read_csv(args.input)
    if "SMILES" not in frame:
        parser.error("Input CSV needs a SMILES column")
    model = SkinEnsemble(Path(__file__).resolve().parent, args.device)
    if args.animal not in model.animals:
        parser.error(f"Unknown Animal {args.animal!r}; choose from {model.animals}")

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    canonical = []
    status = []
    for raw in frame["SMILES"]:
        mol = Chem.MolFromSmiles(str(raw)) if pd.notna(raw) else None
        value = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
        canonical.append(value)
        if value is None:
            status.append("invalid_smiles")
        elif model.encoder.token_count(value) > 202:
            status.append("over_202_tokens")
        else:
            status.append("ok")
    valid = [i for i, item in enumerate(status) if item == "ok"]
    unique = list(dict.fromkeys(canonical[i] for i in valid))
    ad_scores = None
    if calibrated_condition and args.ad_rule != "none":
        z, direction, contributions, disagreement = model.predict_kp_details(
            unique, args.batch_size
        )
        ad_scores = KpHumanAD(Path(__file__).resolve().parent).score(
            unique, contributions,
            min_tanimoto=args.ad_min_tanimoto,
            max_nn_ratio=args.ad_max_nn_ratio,
            max_q_ratio=args.ad_max_q_ratio,
        )
    else:
        z, direction = model.predict(unique, args.metric, args.animal, args.batch_size)
    values = dict(zip(unique, zip(z, direction), strict=True))

    frame["canonical_smiles"] = canonical
    frame["prediction_status"] = status
    frame["prediction_metric"] = args.metric
    frame["assumed_animal"] = args.animal
    frame["regression_z_prediction"] = np.nan
    frame["direction_score"] = np.nan
    for i in valid:
        frame.at[i, "regression_z_prediction"] = values[canonical[i]][0]
        frame.at[i, "direction_score"] = values[canonical[i]][1]
    if args.metric == "Kp":
        frame["predicted_logKp_cm_h"] = frame["direction_score"]
        frame["predicted_logKp_cm_s"] = frame["direction_score"] - np.log10(3600.0)
    frame["ad_status"] = (
        "not_evaluable" if ad_scores is not None else
        "not_requested" if args.ad_rule == "none" else "not_calibrated_for_condition"
    )
    if ad_scores is not None:
        frame["ad_rule"] = args.ad_rule
        frame["ad_reference_version"] = "kp_human_training_ref_v1"
        frame["ad_in_domain"] = pd.NA
        frame["ensemble_sd_z"] = np.nan
        for column in ad_scores:
            frame[column] = pd.NA
        ad_lookup = {value: j for j, value in enumerate(unique)}
        for i in valid:
            j = ad_lookup[canonical[i]]
            frame.at[i, "ensemble_sd_z"] = disagreement[j]
            for column, scores in ad_scores.items():
                frame.at[i, column] = scores[j]
            in_domain = bool(ad_scores[
                "ad_structure_in_domain" if args.ad_rule == "structure"
                else "ad_screen_in_domain"
            ][j])
            frame.at[i, "ad_in_domain"] = in_domain
            frame.at[i, "ad_status"] = "within_domain" if in_domain else "outside_domain"
    total_rows = len(frame)
    if args.ad_only_in_domain:
        frame = frame.loc[frame["ad_in_domain"].eq(True).fillna(False)].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} of {total_rows} input rows to {args.output}; "
          f"{len(valid)} predictions, {total_rows - len(valid)} skipped")


if __name__ == "__main__":
    main()
