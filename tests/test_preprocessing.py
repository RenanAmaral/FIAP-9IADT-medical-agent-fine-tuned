from preprocessing.anonymize import anonymize_record, anonymize_text
from preprocessing.clean import clean_records, clean_text
from preprocessing.curate import filter_records, split_dataset
import random


def test_anonymize_text_masks_all_labeled_pii():
    text = (
        "Paciente: Maria da Silva\n"
        "CPF: 123.456.789-01 | RG: 314159265\n"
        "Data de nascimento: 01/02/1980\n"
        "Telefone: (11) 91234-5678\n"
        "Endereço: Rua das Flores, 123\n"
        "Prontuário nº PRT-000123\n"
    )
    masked, found = anonymize_text(text)

    assert "Maria da Silva" not in masked
    assert "123.456.789-01" not in masked
    assert "314159265" not in masked
    assert "01/02/1980" not in masked
    assert "91234-5678" not in masked
    assert "Rua das Flores, 123" not in masked
    assert "PRT-000123" not in masked
    assert found


def test_anonymize_record_strips_ground_truth_field():
    record = {"raw_text": "Paciente: João\nCPF: 111.222.333-44", "_pii_ground_truth": {"nome": "João"}}
    result = anonymize_record(record)
    assert "_pii_ground_truth" not in result
    assert "João" not in result["raw_text"]


def test_clean_text_removes_noise_and_normalizes_whitespace():
    dirty = "Texto   com <<confidencial>> ruído***\n\n\n\nextra"
    cleaned = clean_text(dirty)
    assert "<<confidencial>>" not in cleaned
    assert "***" not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_records_deduplicates():
    records = [
        {"instruction": "Qual a conduta?", "output": "Resposta A"},
        {"instruction": "Qual a conduta?", "output": "Resposta A"},
        {"instruction": "Outra pergunta?", "output": "Resposta B"},
    ]
    cleaned, stats = clean_records(records)
    assert stats["duplicates_removed"] == 1
    assert len(cleaned) == 2


def test_filter_records_drops_short_answers():
    records = [
        {"instruction": "Pergunta curta", "output": "Resposta curta"},
        {"instruction": "Pergunta", "output": " ".join(["palavra"] * 30)},
    ]
    kept, stats = filter_records(records)
    assert stats["dropped_incomplete"] == 1
    assert len(kept) == 1


def test_split_dataset_covers_all_records_without_overlap():
    records = [
        {"especialidade": "cardiologia", "id": i} for i in range(20)
    ] + [{"especialidade": "pediatria", "id": i} for i in range(20, 30)]
    rng = random.Random(0)
    splits = split_dataset(records, rng)

    all_ids = [r["id"] for part in splits.values() for r in part]
    assert sorted(all_ids) == list(range(30))
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0
