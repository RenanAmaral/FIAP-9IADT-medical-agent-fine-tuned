"""Orquestra a Etapa 1 completa: geração sintética -> limpeza -> anonimização
-> curadoria -> splits finais.

Uso:
    python -m preprocessing.run_pipeline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessing import anonymize, clean, curate, generate_synthetic_data


def run(seed: int = 42, data_dir: Path = Path("data")) -> None:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    print("== 1/4 Gerando dados sintéticos ==")
    import random

    from faker import Faker

    rng = random.Random(seed)
    fake = Faker("pt_BR")
    Faker.seed(seed)

    qa_pairs = generate_synthetic_data.build_qa_pairs(rng)
    hospital_records = generate_synthetic_data.build_hospital_records(fake, rng)
    generate_synthetic_data.write_jsonl(qa_pairs, raw_dir / "qa_pairs.jsonl")
    generate_synthetic_data.write_jsonl(hospital_records, raw_dir / "registros_hospitalares.jsonl")
    generate_synthetic_data.write_protocol_docs(raw_dir)

    print("== 2/4 Limpando texto (preprocessing) ==")
    qa_clean, qa_clean_stats = clean.clean_records(clean.read_jsonl(raw_dir / "qa_pairs.jsonl"))
    clean.write_jsonl(qa_clean, processed_dir / "qa_pairs.clean.jsonl")

    records_clean, records_clean_stats = clean.clean_records(
        clean.read_jsonl(raw_dir / "registros_hospitalares.jsonl")
    )
    clean.write_jsonl(records_clean, processed_dir / "registros_hospitalares.clean.jsonl")

    print("== 3/4 Anonimizando registros hospitalares ==")
    original_records = clean.read_jsonl(raw_dir / "registros_hospitalares.jsonl")
    anonymized = [anonymize.anonymize_record(r) for r in records_clean]
    anonymize.write_jsonl(anonymized, processed_dir / "registros_hospitalares.anonimizado.jsonl")
    anonymization_report = anonymize.validate_against_ground_truth(original_records, anonymized)

    print("== 4/4 Curando e dividindo dataset de fine-tuning ==")
    filtered, filter_stats = curate.filter_records(qa_clean)
    balanced, balance_stats = curate.balance_by_specialty(filtered, __import__("random").Random(seed))
    splits = curate.split_dataset(balanced, __import__("random").Random(seed))
    for name, items in splits.items():
        curate.write_jsonl(items, processed_dir / f"{name}.jsonl")
    split_sizes = {name: len(items) for name, items in splits.items()}
    curate.write_report(processed_dir / "curation_report.md", filter_stats, balance_stats, split_sizes)

    manifest = {
        "seed": seed,
        "qa_pairs_generated": len(qa_pairs),
        "qa_pairs_after_clean": qa_clean_stats,
        "hospital_records_generated": len(hospital_records),
        "hospital_records_after_clean": records_clean_stats,
        "anonymization_validation": anonymization_report,
        "curation_filter_stats": filter_stats,
        "final_split_sizes": split_sizes,
    }
    (processed_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nPipeline concluído. Manifesto salvo em", processed_dir / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    run(seed=args.seed, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
