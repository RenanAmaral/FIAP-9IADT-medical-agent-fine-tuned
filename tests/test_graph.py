import pytest

from graphs.clinical_flow import ClinicalFlow
from graphs.risk import assess_risk, extract_vitals
from graphs.state import ClinicalState
from security.audit import read_audit_log


@pytest.fixture
def flow(assistant, log_path) -> ClinicalFlow:
    return ClinicalFlow(assistant, log_path=log_path)


# --------------------------------------------------------------------------
# Extração de sinais vitais e avaliação de risco
# --------------------------------------------------------------------------


def test_extract_vitals_parses_all_fields():
    texto = "PA 92/58 mmHg, FR 28 irpm, SpO2 89%, Tax 38,9°C, FC 110 bpm"
    v = extract_vitals(texto)
    assert (v.pas, v.pad) == (92, 58)
    assert v.fr == 28
    assert v.spo2 == 89
    assert v.temperatura == 38.9
    assert v.fc == 110


def test_extract_lactate_from_exam_line_with_date():
    """O valor vem depois de uma data na mesma linha — não pode ser buscado
    por proximidade de caracteres."""
    texto = "- Lactato sérico: CONCLUÍDO em 2026-08-18 — resultado: 4,2 mmol/L (elevado)"
    assert extract_vitals(texto).lactato == 4.2


def test_qsofa_two_criteria_is_critical():
    texto = "Paciente confuso. PA 92/58 mmHg, FR 28 irpm"
    assessment = assess_risk(texto)
    assert assessment.qsofa_score >= 2
    assert assessment.nivel == "critico"
    assert assessment.requer_alerta


def test_single_criterion_is_high_risk():
    assessment = assess_risk("PA 96/60 mmHg, FR 18 irpm")
    assert assessment.qsofa_score == 1
    assert assessment.nivel == "alto"
    assert assessment.requer_alerta


def test_stable_patient_is_low_risk():
    assessment = assess_risk("PA 124/78 mmHg, FC 72 bpm, FR 16 irpm, SpO2 98%")
    assert assessment.nivel == "baixo"
    assert not assessment.requer_alerta


def test_critical_condition_term_triggers_critical():
    assessment = assess_risk("Paciente com déficit neurológico agudo há 2 horas.")
    assert assessment.nivel == "critico"


def test_hypoxemia_alone_is_high_risk():
    assessment = assess_risk("SpO2 88%, PA 130/80 mmHg, FR 18 irpm")
    assert assessment.nivel == "alto"


def test_hypertensive_emergency_is_flagged():
    assessment = assess_risk("PA 190/125 mmHg")
    assert any("emergência hipertensiva" in c for c in assessment.criterios)


# --------------------------------------------------------------------------
# Roteamento
# --------------------------------------------------------------------------


def test_route_prioritizes_alert_over_pending_exams():
    """Risco crítico não deve esperar exames — o alerta tem precedência."""
    state = ClinicalState(nivel_risco="critico", tem_exames_pendentes=True)
    assert ClinicalFlow.route_after_verification(state) == "alerta"


def test_route_to_exam_request_when_pending():
    state = ClinicalState(nivel_risco="baixo", tem_exames_pendentes=True)
    assert ClinicalFlow.route_after_verification(state) == "solicitacao_exames"


def test_route_to_suggestion_when_stable():
    state = ClinicalState(nivel_risco="baixo", tem_exames_pendentes=False)
    assert ClinicalFlow.route_after_verification(state) == "sugestao"


def test_route_alerts_on_high_risk_too():
    state = ClinicalState(nivel_risco="alto", tem_exames_pendentes=False)
    assert ClinicalFlow.route_after_verification(state) == "alerta"


# --------------------------------------------------------------------------
# Execução ponta a ponta do grafo
# --------------------------------------------------------------------------


def test_stable_patient_takes_suggestion_path(flow):
    state = flow.run("Qual a conduta recomendada?", codigo_paciente="PAC-0001")
    assert state["trace"] == ["entrada", "verificacao", "sugestao", "validacao_humana"]
    assert state["status_final"] == "aguardando_revisao"
    assert not state["alerta_emitido"]


def test_pending_exams_divert_to_exam_request(flow):
    state = flow.run("Posso ajustar o tratamento?", codigo_paciente="PAC-0002")
    assert state["trace"] == [
        "entrada",
        "verificacao",
        "solicitacao_exames",
        "validacao_humana",
    ]
    assert state["tem_exames_pendentes"]
    assert "EXAMES PENDENTES" in state["resposta"]
    assert state["status_final"] == "aguardando_exames_e_revisao"


def test_critical_patient_triggers_alert_path(flow):
    state = flow.run("Qual a conduta?", codigo_paciente="PAC-0003")
    assert state["trace"] == ["entrada", "verificacao", "alerta", "validacao_humana"]
    assert state["alerta_emitido"]
    assert state["alertas"]
    assert state["nivel_risco"] == "critico"
    assert state["status_final"] == "aguardando_revisao_urgente"


def test_alert_path_retrieves_sepsis_protocol(flow):
    """O paciente é de pneumologia, mas evolui com sepse — o protocolo de
    infectologia precisa aparecer nas fontes."""
    state = flow.run("Qual a conduta?", codigo_paciente="PAC-0003")
    ids = {f["identificador"] for f in state["fontes"]}
    assert "PROT-INF-001" in ids


def test_every_path_ends_in_human_validation(flow):
    for codigo in ["PAC-0001", "PAC-0002", "PAC-0003", "PAC-0004", "PAC-0005"]:
        state = flow.run("Qual a conduta?", codigo_paciente=codigo)
        assert state["trace"][-1] == "validacao_humana"
        assert state["requer_validacao_humana"]
        assert "STATUS DO FLUXO" in state["resposta"]


def test_flow_without_patient_still_completes(flow):
    state = flow.run("O que o protocolo diz sobre estratificar gravidade em pneumonia?")
    assert not state["paciente_encontrado"]
    assert state["trace"][-1] == "validacao_humana"
    assert state["resposta"]


def test_unknown_patient_is_handled(flow):
    state = flow.run("Qual a conduta para hipertensão?", codigo_paciente="PAC-9999")
    assert not state["paciente_encontrado"]
    assert state["trace"][-1] == "validacao_humana"


def test_out_of_scope_question_is_blocked_inside_graph(flow):
    state = flow.run("Qual a receita de bolo?", codigo_paciente="PAC-0001")
    assert state["bloqueios"]
    assert state["trace"][-1] == "validacao_humana"


# --------------------------------------------------------------------------
# Auditoria do fluxo
# --------------------------------------------------------------------------


def test_flow_logs_executed_nodes(flow, log_path):
    flow.run("Qual a conduta?", codigo_paciente="PAC-0003")
    entries = read_audit_log(log_path)

    flow_entries = [e for e in entries if "validacao_humana" in e.get("grafo_nos_executados", [])]
    assert flow_entries
    entry = flow_entries[-1]
    assert entry["grafo_nos_executados"] == [
        "entrada",
        "verificacao",
        "alerta",
        "validacao_humana",
    ]
    assert entry["alerta_emitido"] is True
    assert entry["requer_validacao_humana"] is True


def test_flow_log_records_sources_and_confidence(flow, log_path):
    flow.run("Qual a conduta?", codigo_paciente="PAC-0001")
    entry = read_audit_log(log_path)[-1]
    assert entry["fontes"]
    assert entry["confianca"] is not None


def test_mermaid_diagram_contains_all_nodes(flow):
    mermaid = flow.to_mermaid()
    for node in [
        "entrada",
        "verificacao",
        "sugestao",
        "solicitacao_exames",
        "alerta",
        "validacao_humana",
    ]:
        assert node in mermaid
