"""Carregamento e formatação do dataset de fine-tuning.

Puramente Python/JSON — não depende de torch/transformers — para que a
lógica de formatação de prompt seja testável mesmo neste ambiente sem GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from finetuning.config import format_prompt


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_training_texts(records: list[dict]) -> list[str]:
    """Converte registros {instruction, input, output, ...} no texto final
    usado para treino (SFTTrainer com `dataset_text_field="text"`).
    """
    texts = []
    for record in records:
        texts.append(
            format_prompt(
                instruction=record["instruction"],
                input_text=record.get("input", ""),
                output=record["output"],
            )
        )
    return texts


def load_training_dataset(path: str | Path) -> list[dict]:
    """Retorna uma lista de dicts com a chave `text`, no formato esperado
    por `datasets.Dataset.from_list` dentro de `train.py`.
    """
    records = read_jsonl(path)
    texts = build_training_texts(records)
    return [{"text": t} for t in texts]
