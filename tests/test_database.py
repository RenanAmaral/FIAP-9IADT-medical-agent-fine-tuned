from assistant.database import (
    connect,
    get_patient,
    get_patient_record,
    get_pending_exams,
    list_patients,
)


def test_seed_creates_expected_patients(db_path):
    conn = connect(db_path)
    try:
        patients = list_patients(conn)
    finally:
        conn.close()
    assert len(patients) == 5
    assert {p.codigo_paciente for p in patients} == {
        "PAC-0001",
        "PAC-0002",
        "PAC-0003",
        "PAC-0004",
        "PAC-0005",
    }


def test_get_patient_returns_none_for_unknown_code(db_path):
    conn = connect(db_path)
    try:
        assert get_patient(conn, "PAC-9999") is None
        assert get_patient_record(conn, "PAC-9999") is None
    finally:
        conn.close()


def test_pending_exams_detected(db_path):
    conn = connect(db_path)
    try:
        pendentes = get_pending_exams(conn, "PAC-0002")
        nomes = {e.nome_exame for e in pendentes}
    finally:
        conn.close()
    assert "Hemoglobina glicada (HbA1c)" in nomes
    assert all(e.status == "pendente" for e in pendentes)


def test_patient_without_pending_exams(db_path):
    conn = connect(db_path)
    try:
        record = get_patient_record(conn, "PAC-0001")
    finally:
        conn.close()
    assert record is not None
    assert record.exames_pendentes == []


def test_patient_record_context_includes_exams_and_vitals(db_path):
    conn = connect(db_path)
    try:
        record = get_patient_record(conn, "PAC-0003")
    finally:
        conn.close()

    context = record.to_context_string()
    assert "PAC-0003" in context
    assert "pneumonia" in context.lower()
    assert "Lactato sérico" in context
    assert "SpO2" in context


def test_seed_is_idempotent(db_path):
    from assistant.database import seed_database

    conn = connect(db_path)
    try:
        first = seed_database(conn, seed=42)
        second = seed_database(conn, seed=42)
    finally:
        conn.close()
    assert first == second
