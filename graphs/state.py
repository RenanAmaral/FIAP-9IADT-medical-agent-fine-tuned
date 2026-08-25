"""Estado compartilhado do grafo clínico (Etapa 4).

`ClinicalState` é o objeto que trafega entre os nós do LangGraph. Cada nó lê
o que precisa e devolve apenas as chaves que alterou — o LangGraph faz o
merge. `trace` e `alertas` usam `operator.add` como reducer para acumularem
ao longo do fluxo em vez de serem sobrescritos, o que é o que permite auditar
depois exatamente quais nós rodaram (item 13 do enunciado).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

#: Nível de risco atribuído pelo nó de verificação, usado no roteamento.
RiskLevel = Literal["critico", "alto", "moderado", "baixo"]


class ClinicalState(TypedDict, total=False):
    # --- Entrada -----------------------------------------------------------
    pergunta: str
    codigo_paciente: str | None
    session_id: str

    # --- Nó 1: Entrada -----------------------------------------------------
    paciente_encontrado: bool
    contexto_paciente: str | None
    condicao_principal: str | None
    especialidade: str | None
    sinais_vitais: str | None

    # --- Nó 2: Verificação -------------------------------------------------
    exames_pendentes: list[str]
    tem_exames_pendentes: bool
    nivel_risco: RiskLevel
    criterios_risco: list[str]

    # --- Nó 3: Sugestão / solicitação de exames ----------------------------
    resposta: str | None
    fontes: list[dict[str, Any]]
    confianca: float | None
    bloqueios: list[dict[str, Any]]

    # --- Nó 4: Alerta ------------------------------------------------------
    alerta_emitido: bool
    alertas: Annotated[list[str], operator.add]

    # --- Nó 5: Validação humana -------------------------------------------
    requer_validacao_humana: bool
    status_final: str

    # --- Auditoria ---------------------------------------------------------
    trace: Annotated[list[str], operator.add]


def initial_state(
    pergunta: str, codigo_paciente: str | None = None, session_id: str | None = None
) -> ClinicalState:
    from security.audit import new_session_id

    return ClinicalState(
        pergunta=pergunta,
        codigo_paciente=codigo_paciente,
        session_id=session_id or new_session_id(),
        exames_pendentes=[],
        criterios_risco=[],
        fontes=[],
        bloqueios=[],
        alertas=[],
        trace=[],
        alerta_emitido=False,
        requer_validacao_humana=True,
    )


__all__ = ["ClinicalState", "RiskLevel", "initial_state"]
