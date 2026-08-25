import json

from security.audit import (
    AuditRecord,
    get_audit_logger,
    log_interaction,
    new_session_id,
    read_audit_log,
    utc_now_iso,
)


def _record(**overrides) -> AuditRecord:
    base = dict(
        session_id="sess-teste",
        timestamp=utc_now_iso(),
        pergunta="Qual a conduta?",
    )
    base.update(overrides)
    return AuditRecord(**base)


def test_log_writes_one_json_line_per_interaction(log_path):
    log_interaction(_record(), log_path)
    log_interaction(_record(pergunta="Outra pergunta"), log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # cada linha é JSON válido isoladamente


def test_repeated_logger_calls_do_not_duplicate_entries(log_path):
    """Obter o logger várias vezes não pode acumular handlers — isso
    duplicaria cada linha do log."""
    get_audit_logger(log_path)
    get_audit_logger(log_path)
    get_audit_logger(log_path)

    log_interaction(_record(), log_path)
    assert len(read_audit_log(log_path)) == 1


def test_all_required_fields_are_persisted(log_path):
    log_interaction(
        _record(
            codigo_paciente="PAC-0001",
            contexto_recuperado=[{"protocol_id": "PROT-CARD-001", "score": 0.5}],
            resposta="Resposta gerada",
            fontes=[{"tipo": "protocolo", "identificador": "PROT-CARD-001"}],
            confianca=0.85,
            grafo_nos_executados=["entrada", "verificacao"],
            bloqueios_seguranca=[{"reason": "ok"}],
            alerta_emitido=True,
            llm_backend="template",
        ),
        log_path,
    )
    entry = read_audit_log(log_path)[0]

    for field in (
        "timestamp",
        "session_id",
        "pergunta",
        "codigo_paciente",
        "contexto_recuperado",
        "resposta",
        "fontes",
        "confianca",
        "grafo_nos_executados",
        "bloqueios_seguranca",
        "alerta_emitido",
        "requer_validacao_humana",
        "llm_backend",
    ):
        assert field in entry, f"campo ausente no log: {field}"


def test_requires_human_validation_defaults_to_true(log_path):
    log_interaction(_record(), log_path)
    assert read_audit_log(log_path)[0]["requer_validacao_humana"] is True


def test_read_missing_log_returns_empty(tmp_path):
    assert read_audit_log(tmp_path / "inexistente.jsonl") == []


def test_session_ids_are_unique():
    assert len({new_session_id() for _ in range(100)}) == 100


def test_accents_are_preserved_in_log(log_path):
    log_interaction(_record(pergunta="Avaliação de hipertensão e sepse"), log_path)
    assert read_audit_log(log_path)[0]["pergunta"] == "Avaliação de hipertensão e sepse"
