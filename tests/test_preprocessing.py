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


def test_birth_dates_do_not_depend_on_system_clock():
    """Faker.date_of_birth calcula a idade a partir de "hoje", então o mesmo
    seed produzia datas diferentes conforme os dias passavam — o dataset
    deixava de ser reprodutível apesar do seed fixo."""
    import datetime
    from unittest import mock

    from faker import Faker

    from preprocessing.generate_synthetic_data import build_hospital_records

    def gerar():
        rng = random.Random(42)
        fake = Faker("pt_BR")
        Faker.seed(42)
        return [r["_pii_ground_truth"]["nascimento"] for r in build_hospital_records(fake, rng)]

    agora = gerar()

    real_date = datetime.date

    class DataFutura(real_date):
        @classmethod
        def today(cls):
            return real_date(2030, 1, 1)

    with mock.patch("datetime.date", DataFutura):
        depois = gerar()

    assert agora == depois


def test_birth_dates_respect_age_bounds():
    from preprocessing.generate_synthetic_data import (
        MAX_AGE_YEARS,
        MIN_AGE_YEARS,
        REFERENCE_DATE,
        _random_birth_date,
    )

    rng = random.Random(7)
    for _ in range(200):
        nascimento = _random_birth_date(rng)
        idade_anos = (REFERENCE_DATE - nascimento).days / 365
        assert MIN_AGE_YEARS <= idade_anos <= MAX_AGE_YEARS
