"""Frozen skin-permeability ensemble and its SMI-TED feature extractor."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vendor_smi_ted.load import AutoEncoderLayer, MoLEncoder, MolTranBertTokenizer
from vendor_smi_ted.fast_transformers.masking import LengthMask


METRICS = ("Jmax", "Jss", "Kp", "Percent absorbed", "Tlag")
CONFIG = {
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "num_feats": 32,
    "max_len": 202,
    "n_output": 1,
    "dropout": 0.2,
    "d_dropout": 0.1,
    "seed": 42,
}


class SmiTedEncoder(nn.Module):
    """Only the two SMI-TED components used by the training embedding cache."""

    def __init__(self, vocab_path: Path):
        super().__init__()
        self.tokenizer = MolTranBertTokenizer(str(vocab_path))
        self.encoder = MoLEncoder(CONFIG, len(self.tokenizer.vocab), eval=True)
        del self.encoder.lang_model  # SMI-TED's language head is not used for embeddings.
        self.projection = AutoEncoderLayer.Encoder(
            CONFIG["max_len"] * CONFIG["n_embd"], CONFIG["n_embd"]
        )

    def token_count(self, smiles: str) -> int:
        return len(self.tokenizer._tokenize(smiles)) + 2

    def forward(self, smiles: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            smiles, padding=True, truncation=False, add_special_tokens=True,
            return_tensors="pt",
        )
        device = self.encoder.tok_emb.weight.device
        idx = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        if idx.shape[1] > CONFIG["max_len"]:
            raise ValueError("SMILES exceeds the 202-token training limit")
        x = self.encoder.drop(self.encoder.tok_emb(idx))
        x = self.encoder.blocks(
            x, length_mask=LengthMask(mask.sum(-1), max_len=idx.shape[1])
        )
        x = x * mask.unsqueeze(-1).to(x.dtype)
        x = F.pad(x, (0, 0, 0, CONFIG["max_len"] - x.shape[1]))
        return self.projection(x.reshape(-1, CONFIG["max_len"] * CONFIG["n_embd"]))


class ChemistryTrunk(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, residual: bool):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input_projection(x)
        update = self.block(self.input_norm(y))
        return self.output_norm(y + update if self.residual else update)


class MultiHeadRegression(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.gate_scale = float(config["gate_scale"])
        hidden = int(config["hidden_dim"])
        self.trunk = ChemistryTrunk(
            int(config["input_dim"]), hidden, float(config["dropout"]),
            bool(config["residual"]),
        )
        self.animal_embedding = nn.Embedding(int(config["num_animals"]), hidden)
        reference = nn.Linear(hidden, 1)
        self.heads = nn.ModuleList(
            [reference] + [deepcopy(reference) for _ in range(int(config["num_metrics"]) - 1)]
        )

    def forward(
        self, embeddings: torch.Tensor, metric_ids: torch.Tensor, animal_ids: torch.Tensor
    ) -> torch.Tensor:
        features = self.trunk(embeddings)
        gate = self.gate_scale * torch.tanh(self.animal_embedding(animal_ids))
        features = features * (1.0 + gate)
        all_scores = torch.stack([head(features).squeeze(-1) for head in self.heads], dim=1)
        return all_scores.gather(1, metric_ids[:, None]).squeeze(1)


class SkinEnsemble:
    def __init__(self, folder: Path, device: str = "cpu"):
        self.device = torch.device(device)
        backbone = torch.load(folder / "weights/smi_ted_encoder.pt", map_location="cpu", weights_only=True)
        self.encoder = SmiTedEncoder(folder / "vendor_smi_ted/bert_vocab_curated.txt")
        self.encoder.load_state_dict(backbone, strict=True)
        self.encoder.to(self.device).eval()
        payload = torch.load(folder / "weights/multihead_ensemble.pt", map_location="cpu", weights_only=True)
        self.metrics = tuple(payload["vocabulary"]["metric_classes"])
        self.animals = tuple(payload["vocabulary"]["animal_classes"])
        self.mean = np.asarray(payload["feature_scaler"]["mean"], dtype=np.float64)
        self.scale = np.asarray(payload["feature_scaler"]["scale"], dtype=np.float64)
        self.targets = payload["target_state"]
        self.heads = []
        for state in payload["states"]:
            model = MultiHeadRegression(payload["model_config"])
            model.load_state_dict(state, strict=True)
            self.heads.append(model.to(self.device).eval())

    @torch.inference_mode()
    def predict(
        self, smiles: list[str], metric: str = "Kp", animal: str = "Human", batch_size: int = 32
    ) -> tuple[np.ndarray, np.ndarray]:
        if metric not in self.metrics or animal not in self.animals:
            raise ValueError(f"Choose Metric from {self.metrics} and Animal from {self.animals}")
        if not smiles:
            return np.empty(0), np.empty(0)
        metric_id = self.metrics.index(metric) + 1
        animal_id = self.animals.index(animal) + 1
        results = []
        for start in range(0, len(smiles), batch_size):
            part = smiles[start : start + batch_size]
            features = self.encoder(part).cpu().numpy().astype(np.float32)
            scaled = ((features.astype(np.float64) - self.mean) / self.scale).astype(np.float32)
            x = torch.from_numpy(scaled).to(self.device)
            m = torch.full((len(part),), metric_id, dtype=torch.long, device=self.device)
            a = torch.full((len(part),), animal_id, dtype=torch.long, device=self.device)
            z = torch.stack([head(x, m, a) for head in self.heads]).mean(0)
            results.append(z.cpu().numpy().astype(np.float64))
        standardized = np.concatenate(results)
        direction = (
            standardized * float(self.targets["scales"][metric])
            + float(self.targets["means"][metric])
        )
        return standardized, direction

    @torch.inference_mode()
    def predict_kp_details(
        self, smiles: list[str], batch_size: int = 32
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Kp/Human prediction plus exact five-seed pre-bias contributions.

        Contribution vectors are the model-specific representation used by the
        separately calibrated Kp/Human applicability-domain screen.
        """
        if not smiles:
            return np.empty(0), np.empty(0), np.empty((len(self.heads), 0, 128)), np.empty(0)
        metric_id = self.metrics.index("Kp") + 1
        animal_id = self.animals.index("Human") + 1
        z_parts, contribution_parts, disagreement_parts = [], [], []
        for start in range(0, len(smiles), batch_size):
            part = smiles[start : start + batch_size]
            features = self.encoder(part).cpu().numpy().astype(np.float32)
            scaled = ((features.astype(np.float64) - self.mean) / self.scale).astype(np.float32)
            x = torch.from_numpy(scaled).to(self.device)
            m = torch.full((len(part),), metric_id, dtype=torch.long, device=self.device)
            a = torch.full((len(part),), animal_id, dtype=torch.long, device=self.device)
            seed_z, seed_contributions = [], []
            for model in self.heads:
                seed_z.append(model(x, m, a))
                conditioned = model.trunk(x) * (
                    1.0 + model.gate_scale * torch.tanh(model.animal_embedding(a))
                )
                seed_contributions.append(
                    (conditioned * model.heads[metric_id].weight).cpu().numpy().astype(np.float64)
                )
            matrix = torch.stack(seed_z).cpu().numpy().astype(np.float64)
            z_parts.append(matrix.mean(axis=0))
            disagreement_parts.append(matrix.std(axis=0, ddof=0))
            contribution_parts.append(np.stack(seed_contributions, axis=0))
        standardized = np.concatenate(z_parts)
        direction = (
            standardized * float(self.targets["scales"]["Kp"])
            + float(self.targets["means"]["Kp"])
        )
        contributions = np.concatenate(contribution_parts, axis=1)
        return standardized, direction, contributions, np.concatenate(disagreement_parts)
