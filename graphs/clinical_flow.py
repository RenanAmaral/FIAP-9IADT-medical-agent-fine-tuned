"""Etapa 4, item 11 — Grafo de decisão clínica com LangGraph.

Modela o fluxo clínico como um grafo de estados com os cinco nós do enunciado
e arestas condicionais que fazem o caminho depender do estado do paciente:

    entrada → verificacao → ┬→ alerta ─────────────→ validacao_humana → END
                            ├→ solicitacao_exames →  validacao_humana → END
                            └→ sugestao ──────────→  validacao_humana → END

Roteamento (`route_after_verification`), em ordem de precedência:

1. **Risco crítico/alto** → `alerta`. Precede tudo: um paciente com critérios
   de sepse não deve esperar a chegada de exames pendentes para que a equipe
   seja acionada.
2. **Exames pendentes** → `solicitacao_exames`. É o desvio que o enunciado
   pede explicitamente: sem os exames, o fluxo solicita em vez de sugerir
   tratamento.
3. **Caso contrário** → `sugestao`, a conduta baseada nos protocolos.

Todos os caminhos convergem para `validacao_humana`, que é terminal: nenhum
fluxo se encerra sem marcar que a revisão de um profissional é obrigatória
(item 12 do enunciado).
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from assistant.chains import MedicalAssistantChain
from assistant.database import DEFAULT_DB_PATH
from graphs.risk import assess_risk
from graphs.state import ClinicalState, initial_state
from security.audit import AuditRecord, log_interaction, utc_now_iso


class ClinicalFlow:
    """Constrói e executa o grafo clínico.

    Recebe a chain da Etapa 3 por injeção, então o grafo funciona com
    qualquer backend de LLM sem alteração.
    """

    def __init__(
        self,
        assistant: MedicalAssistantChain,
        log_path: str | Path = "logs/audit.jsonl",
    ):
        self.assistant = assistant
        self.log_path = Path(log_path)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Nó 1 — Entrada: recebimento e normalização das informações
    # ------------------------------------------------------------------
    def node_entrada(self, state: ClinicalState) -> dict:
        codigo = state.get("codigo_paciente")
        record = self.assistant.get_patient_record(codigo) if codigo else None

        if record is None:
            return {
                "paciente_encontrado": False,
                "contexto_paciente": None,
                "condicao_principal": None,
                "especialidade": None,
                "sinais_vitais": None,
                "trace": ["entrada"],
            }

        sinais = record.evolucoes[0].sinais_vitais if record.evolucoes else None
        return {
            "paciente_encontrado": True,
            "contexto_paciente": record.to_context_string(),
            "condicao_principal": record.patient.condicao_principal,
            "especialidade": record.patient.especialidade,
            "sinais_vitais": sinais,
            "trace": ["entrada"],
        }

    # ------------------------------------------------------------------
    # Nó 2 — Verificação: exames pendentes e critérios de risco
    # ------------------------------------------------------------------
    def node_verificacao(self, state: ClinicalState) -> dict:
        codigo = state.get("codigo_paciente")
        contexto = state.get("contexto_paciente") or ""

        pendentes: list[str] = []
        if codigo and state.get("paciente_encontrado"):
            record = self.assistant.get_patient_record(codigo)
            if record:
                pendentes = [e.nome_exame for e in record.exames_pendentes]

        assessment = assess_risk(contexto) if contexto else None

        return {
            "exames_pendentes": pendentes,
            "tem_exames_pendentes": bool(pendentes),
            "nivel_risco": assessment.nivel if assessment else "baixo",
            "criterios_risco": assessment.criterios if assessment else [],
            "trace": ["verificacao"],
        }

    # ------------------------------------------------------------------
    # Nó 3a — Sugestão de conduta
    # ------------------------------------------------------------------
    def node_sugestao(self, state: ClinicalState) -> dict:
        response = self.assistant.run(
            pergunta=state["pergunta"],
            codigo_paciente=state.get("codigo_paciente"),
            session_id=state["session_id"],
            grafo_nos_executados=list(state.get("trace", [])) + ["sugestao"],
        )
        return {
            "resposta": response.texto_completo,
            "fontes": [s.to_dict() for s in response.explanation.sources]
            if response.explanation
            else [],
            "confianca": response.explanation.confidence if response.explanation else None,
            "bloqueios": [{"reason": response.motivo_bloqueio}] if response.bloqueado else [],
            "trace": ["sugestao"],
        }

    # ------------------------------------------------------------------
    # Nó 3b — Solicitação de exames (desvio por exames pendentes)
    # ------------------------------------------------------------------
    def node_solicitacao_exames(self, state: ClinicalState) -> dict:
        pendentes = state.get("exames_pendentes", [])

        # Mesmo no desvio, consultamos os protocolos: o profissional precisa
        # saber por que aqueles exames importam para a conduta.
        response = self.assistant.run(
            pergunta=(
                f"{state['pergunta']} Considere que os seguintes exames ainda estão "
                f"pendentes: {', '.join(pendentes)}. Explique o que o protocolo "
                "determina antes de definir a conduta."
            ),
            codigo_paciente=state.get("codigo_paciente"),
            session_id=state["session_id"],
            grafo_nos_executados=list(state.get("trace", [])) + ["solicitacao_exames"],
        )

        cabecalho = (
            "🔬 **CONDUTA SUSPENSA — EXAMES PENDENTES**\n\n"
            "O fluxo não avançou para sugestão de tratamento porque há exames "
            "pendentes no prontuário:\n"
            + "\n".join(f"- {nome}" for nome in pendentes)
            + "\n\nRecomenda-se aguardar/solicitar estes resultados antes de "
            "definir a conduta definitiva.\n\n---\n\n"
        )

        return {
            "resposta": cabecalho + response.texto_completo,
            "fontes": [s.to_dict() for s in response.explanation.sources]
            if response.explanation
            else [],
            "confianca": response.explanation.confidence if response.explanation else None,
            "bloqueios": [{"reason": response.motivo_bloqueio}] if response.bloqueado else [],
            "trace": ["solicitacao_exames"],
        }

    # ------------------------------------------------------------------
    # Nó 4 — Alerta à equipe médica
    # ------------------------------------------------------------------
    def node_alerta(self, state: ClinicalState) -> dict:
        criterios = state.get("criterios_risco", [])
        nivel = state.get("nivel_risco", "baixo")
        codigo = state.get("codigo_paciente") or "não identificado"

        alerta = (
            f"🚨 ALERTA DE RISCO {nivel.upper()} — paciente {codigo}. "
            f"Critérios detectados: {'; '.join(criterios)}. "
            "Acionamento imediato da equipe médica recomendado."
        )

        response = self.assistant.run(
            pergunta=(
                f"{state['pergunta']} ATENÇÃO: o paciente apresenta critérios de "
                f"risco {nivel} ({'; '.join(criterios)}). Qual a conduta de "
                "urgência prevista nos protocolos internos?"
            ),
            codigo_paciente=state.get("codigo_paciente"),
            session_id=state["session_id"],
            grafo_nos_executados=list(state.get("trace", [])) + ["alerta"],
        )

        cabecalho = (
            f"🚨 **ALERTA — RISCO {nivel.upper()}**\n\n"
            "Critérios de risco identificados automaticamente no prontuário:\n"
            + "\n".join(f"- {c}" for c in criterios)
            + "\n\n**A equipe médica deve ser acionada imediatamente.** "
            "As orientações abaixo são de apoio e não substituem o atendimento "
            "presencial.\n\n---\n\n"
        )

        pendentes = state.get("exames_pendentes", [])
        if pendentes:
            cabecalho += (
                "Observação: há exames pendentes ("
                + ", ".join(pendentes)
                + "), mas o acionamento da equipe tem precedência sobre aguardá-los.\n\n"
            )

        return {
            "alerta_emitido": True,
            "alertas": [alerta],
            "resposta": cabecalho + response.texto_completo,
            "fontes": [s.to_dict() for s in response.explanation.sources]
            if response.explanation
            else [],
            "confianca": response.explanation.confidence if response.explanation else None,
            "bloqueios": [{"reason": response.motivo_bloqueio}] if response.bloqueado else [],
            "trace": ["alerta"],
        }

    # ------------------------------------------------------------------
    # Nó 5 — Validação humana (terminal)
    # ------------------------------------------------------------------
    def node_validacao_humana(self, state: ClinicalState) -> dict:
        alerta = state.get("alerta_emitido", False)
        pendentes = state.get("tem_exames_pendentes", False)

        if alerta:
            status = "aguardando_revisao_urgente"
        elif pendentes:
            status = "aguardando_exames_e_revisao"
        else:
            status = "aguardando_revisao"

        nota = (
            "\n\n---\n"
            "✅ **STATUS DO FLUXO:** encerrado em validação humana "
            f"(`{status}`). Nenhuma conduta é considerada final até a revisão "
            "e aprovação de um profissional de saúde responsável."
        )

        resposta = (state.get("resposta") or "") + nota

        return {
            "requer_validacao_humana": True,
            "status_final": status,
            "resposta": resposta,
            "trace": ["validacao_humana"],
        }

    # ------------------------------------------------------------------
    # Roteamento condicional
    # ------------------------------------------------------------------
    @staticmethod
    def route_after_verification(state: ClinicalState) -> str:
        """Decide o caminho a partir do estado clínico do paciente."""
        if state.get("nivel_risco") in {"critico", "alto"}:
            return "alerta"
        if state.get("tem_exames_pendentes"):
            return "solicitacao_exames"
        return "sugestao"

    # ------------------------------------------------------------------
    def _build_graph(self):
        builder = StateGraph(ClinicalState)

        builder.add_node("entrada", self.node_entrada)
        builder.add_node("verificacao", self.node_verificacao)
        builder.add_node("sugestao", self.node_sugestao)
        builder.add_node("solicitacao_exames", self.node_solicitacao_exames)
        builder.add_node("alerta", self.node_alerta)
        builder.add_node("validacao_humana", self.node_validacao_humana)

        builder.add_edge(START, "entrada")
        builder.add_edge("entrada", "verificacao")
        builder.add_conditional_edges(
            "verificacao",
            self.route_after_verification,
            {
                "alerta": "alerta",
                "solicitacao_exames": "solicitacao_exames",
                "sugestao": "sugestao",
            },
        )
        builder.add_edge("sugestao", "validacao_humana")
        builder.add_edge("solicitacao_exames", "validacao_humana")
        builder.add_edge("alerta", "validacao_humana")
        builder.add_edge("validacao_humana", END)

        return builder.compile()

    # ------------------------------------------------------------------
    def run(
        self,
        pergunta: str,
        codigo_paciente: str | None = None,
        session_id: str | None = None,
    ) -> ClinicalState:
        state = initial_state(pergunta, codigo_paciente, session_id)
        final_state = self.graph.invoke(state)
        self._log_flow(final_state)
        return final_state

    def _log_flow(self, state: ClinicalState) -> None:
        """Registro de auditoria do fluxo completo.

        A chain já registra cada chamada individual à LLM; esta entrada
        adicional consolida o caminho percorrido no grafo, que é o que o item
        13 do enunciado pede ("nós do grafo executados").
        """
        record = AuditRecord(
            session_id=state.get("session_id", "desconhecida"),
            timestamp=utc_now_iso(),
            pergunta=state.get("pergunta", ""),
            codigo_paciente=state.get("codigo_paciente"),
            contexto_paciente=state.get("contexto_paciente"),
            resposta=state.get("resposta"),
            fontes=state.get("fontes", []),
            confianca=state.get("confianca"),
            grafo_nos_executados=list(state.get("trace", [])),
            bloqueios_seguranca=state.get("bloqueios", []),
            alerta_emitido=state.get("alerta_emitido", False),
            requer_validacao_humana=state.get("requer_validacao_humana", True),
            llm_backend=self.assistant.llm_backend_name,
        )
        log_interaction(record, self.log_path)

    # ------------------------------------------------------------------
    def to_mermaid(self) -> str:
        """Diagrama Mermaid do grafo, para o relatório técnico (item 17)."""
        return self.graph.get_graph().draw_mermaid()


def build_clinical_flow(
    backend: str = "template",
    db_path: str | Path = DEFAULT_DB_PATH,
    protocols_dir: str | Path = "data/protocols",
    log_path: str | Path = "logs/audit.jsonl",
    **llm_kwargs,
) -> ClinicalFlow:
    from assistant.chains import build_assistant

    assistant = build_assistant(
        backend=backend,
        db_path=db_path,
        protocols_dir=protocols_dir,
        log_path=log_path,
        **llm_kwargs,
    )
    return ClinicalFlow(assistant, log_path=log_path)


__all__ = ["ClinicalFlow", "build_clinical_flow"]
