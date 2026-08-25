"""Etapa 1, item 4 — Curadoria: filtragem, balanceamento e split final.

Critérios aplicados (documentados também em data/processed/curation_report.md
para entrar no relatório técnico, conforme exigido no enunciado):

1. Descarte de respostas incompletas: `output` com menos de `MIN_OUTPUT_TOKENS`
   tokens (aproximação por split em espaços) é descartado.
2. Limite de tokens: exemplos com `instruction + output` acima de
   `MAX_TOTAL_TOKENS` tokens são descartados, para manter o custo de
   fine-tuning previsível.
3. Balanceamento por especialidade: nenhuma especialidade pode ter mais que
   `BALANCE_FACTOR` vezes o número de exemplos da especialidade menos
   representada (undersampling aleatório determinístico).
4. Split determinístico em train/val/test (80/10/10) estratificado por
   especialidade.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

MIN_OUTPUT_TOKENS = 15
MAX_TOTAL_TOKENS = 800
BALANCE_FACTOR = 1.5
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def _token_count(text: str) -> int:
    return len(text.split())


def filter_records(records: list[dict]) -> tuple[list[dict], dict]:
    kept = []
    dropped_incomplete = 0
    dropped_too_long = 0

    for record in records:
        output = record.get("output", "")
        instruction = record.get("instruction", "")

        if _token_count(output) < MIN_OUTPUT_TOKENS:
            dropped_incomplete += 1
            continue
        if _token_count(instruction) + _token_count(output) > MAX_TOTAL_TOKENS:
            dropped_too_long += 1
            continue
        kept.append(record)

    stats = {
        "input_count": len(records),
        "dropped_incomplete": dropped_incomplete,
        "dropped_too_long": dropped_too_long,
        "kept_after_filter": len(kept),
    }
    return kept, stats


def balance_by_specialty(records: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    by_specialty: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_specialty[record.get("especialidade", "geral")].append(record)

    if not by_specialty:
        return records, {}

    min_count = min(len(v) for v in by_specialty.values())
    cap = max(min_count, int(min_count * BALANCE_FACTOR)) if min_count > 0 else 0

    balanced = []
    per_specialty_stats = {}
    for specialty, items in by_specialty.items():
        rng.shuffle(items)
        selected = items[:cap] if cap else items
        per_specialty_stats[specialty] = {"before": len(items), "after": len(selected)}
        balanced.extend(selected)

    rng.shuffle(balanced)
    return balanced, per_specialty_stats


def split_dataset(records: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    by_specialty: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_specialty[record.get("especialidade", "geral")].append(record)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    train_ratio, val_ratio, _ = SPLIT_RATIOS

    for items in by_specialty.values():
        rng.shuffle(items)
        n = len(items)
        if n < 3:
            # Poucos exemplos para dividir com sentido: tudo vai para treino.
            splits["train"].extend(items)
            continue
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
        n_train = n - n_val - n_test
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for key in splits:
        rng.shuffle(splits[key])
    return splits


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_report(path: Path, filter_stats: dict, balance_stats: dict, split_sizes: dict) -> None:
    lines = [
        "# Relatório de Curadoria do Dataset",
        "",
        "## Critérios aplicados",
        f"- Resposta mínima: {MIN_OUTPUT_TOKENS} tokens (descarta respostas incompletas).",
        f"- Tamanho máximo (instrução + resposta): {MAX_TOTAL_TOKENS} tokens.",
        f"- Balanceamento por especialidade: fator máximo {BALANCE_FACTOR}x em relação à "
        "especialidade menos representada (undersampling).",
        f"- Split: {int(SPLIT_RATIOS[0]*100)}% treino / {int(SPLIT_RATIOS[1]*100)}% validação / "
        f"{int(SPLIT_RATIOS[2]*100)}% teste, estratificado por especialidade.",
        "",
        "## Estatísticas de filtragem",
        f"- Registros de entrada: {filter_stats['input_count']}",
        f"- Descartados por resposta incompleta: {filter_stats['dropped_incomplete']}",
        f"- Descartados por excesso de tokens: {filter_stats['dropped_too_long']}",
        f"- Restantes após filtragem: {filter_stats['kept_after_filter']}",
        "",
        "## Balanceamento por especialidade",
    ]
    for specialty, counts in balance_stats.items():
        lines.append(f"- {specialty}: {counts['before']} -> {counts['after']}")

    lines += [
        "",
        "## Tamanho final dos splits",
        f"- train: {split_sizes['train']}",
        f"- val: {split_sizes['val']}",
        f"- test: {split_sizes['test']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = read_jsonl(args.in_file)

    filtered, filter_stats = filter_records(records)
    balanced, balance_stats = balance_by_specialty(filtered, rng)
    splits = split_dataset(balanced, rng)

    for name, items in splits.items():
        write_jsonl(items, args.out_dir / f"{name}.jsonl")

    split_sizes = {name: len(items) for name, items in splits.items()}
    write_report(args.out_dir / "curation_report.md", filter_stats, balance_stats, split_sizes)

    print(f"Curadoria concluída. Splits: {split_sizes}")
    print(f"Relatório salvo em {args.out_dir / 'curation_report.md'}")


if __name__ == "__main__":
    main()
