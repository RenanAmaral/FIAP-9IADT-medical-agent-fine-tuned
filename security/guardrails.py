"""Etapa 5, item 12 — Limites de atuação (guardrails).

Camada de validação **programática**, independente do modelo. O system prompt
(`finetuning/config.py`) já instrui o modelo a não prescrever, exigir validação
humana e recusar temas fora de escopo, mas instrução em prompt não é um
controle de segurança confiável: o modelo pode ignorá-la, e o comportamento
muda a cada fine-tuning. Estas funções são determinísticas e rodam sempre,
antes e depois da chamada ao LLM.

Três controles:

1. `check_input_scope` (pré-LLM) — recusa perguntas fora do escopo dos
   protocolos internos e bloqueia tentativas explícitas de obter prescrição
   direta ou de burlar as regras (prompt injection básico).
2. `check_output_safety` (pós-LLM) — detecta prescrição direta na resposta
   gerada (posologia concreta, imperativo de administrar) e bloqueia.
3. `enforce_human_validation` (pós-LLM) — garante que toda resposta liberada
   carregue o aviso de validação humana obrigatória, anexando-o se o modelo
   não o produziu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

HUMAN_VALIDATION_NOTICE = (
    "⚠️ VALIDAÇÃO HUMANA OBRIGATÓRIA: esta é uma sugestão de apoio à decisão "
    "gerada por um assistente virtual a partir dos protocolos internos e dos "
    "dados do prontuário. Não constitui prescrição. A conduta final, incluindo "
    "qualquer medicação, dose ou procedimento, deve ser definida e validada "
    "pelo médico responsável."
)

OUT_OF_SCOPE_RESPONSE = (
    "Esta pergunta está fora do escopo dos protocolos clínicos internos do "
    "hospital, que é o domínio para o qual este assistente foi autorizado. "
    "Posso ajudar com dúvidas sobre condutas clínicas, exames e fluxos "
    "cobertos pelos protocolos internos."
)

PRESCRIPTION_BLOCKED_RESPONSE = (
    "Não posso fornecer prescrição direta (medicamento, dose ou posologia "
    "específica para administrar a um paciente). Posso apresentar o que o "
    "protocolo interno recomenda como linha de tratamento, para que o médico "
    "responsável avalie e prescreva."
)


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


class GuardrailReason(str, Enum):
    OUT_OF_SCOPE = "out_of_scope"
    DIRECT_PRESCRIPTION_REQUEST = "direct_prescription_request"
    DIRECT_PRESCRIPTION_OUTPUT = "direct_prescription_output"
    PROMPT_INJECTION = "prompt_injection"
    OK = "ok"


@dataclass
class GuardrailResult:
    action: GuardrailAction
    reason: GuardrailReason
    message: str = ""
    matched_patterns: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action is GuardrailAction.BLOCK

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "reason": self.reason.value,
            "message": self.message,
            "matched_patterns": self.matched_patterns,
        }


# --------------------------------------------------------------------------
# Vocabulário de escopo
# --------------------------------------------------------------------------

# Termos que indicam que a pergunta pertence ao domínio clínico coberto pelos
# protocolos internos. Derivado das condições do banco de protocolos mais
# vocabulário clínico geral.
CLINICAL_TERMS = {
    "paciente", "protocolo", "conduta", "exame", "exames", "diagnóstico",
    "diagnostico", "tratamento", "sintoma", "sintomas", "clínico", "clinico",
    "prontuário", "prontuario", "internação", "internacao", "encaminhamento",
    "urgência", "urgencia", "emergência", "emergencia", "risco", "alerta",
    "dose", "medicação", "medicacao", "medicamento", "antibiótico", "antibiotico",
    "hipertensão", "hipertensao", "pressão", "pressao", "diabetes", "glicemia",
    "pneumonia", "dpoc", "avc", "sepse", "febre", "lombalgia", "gestação",
    "gestacao", "gestante", "pré-natal", "pre-natal", "fibrilação", "fibrilacao",
    "arritmia", "cardiologia", "endocrinologia", "pneumologia", "neurologia",
    "infectologia", "pediatria", "ginecologia", "ortopedia", "laudo",
    "sinais vitais", "saturação", "saturacao", "anticoagulação", "anticoagulacao",
    "triagem", "estratificação", "estratificacao", "comorbidade", "alergia",
    "hemograma", "creatinina", "lactato", "ecg", "radiografia", "tomografia",
    "hba1c", "curb-65", "qsofa", "trombólise", "trombolise", "evolução", "evolucao",
}

# Temas explicitamente fora do escopo autorizado.
OUT_OF_SCOPE_PATTERNS = [
    (re.compile(r"\b(receita|recipe)\s+de\s+(bolo|comida|massa|pizza)", re.I), "culinária"),
    (re.compile(r"\b(futebol|jogo|campeonato|placar)\b", re.I), "esportes"),
    (re.compile(r"\b(bitcoin|criptomoeda|investir|ações da bolsa|day trade)\b", re.I), "finanças"),
    (re.compile(r"\b(piada|poema|história de ficção|conto)\b", re.I), "entretenimento"),
    (re.compile(r"\b(código|script|programa)\s+(python|java|javascript)\b", re.I), "programação"),
    (re.compile(r"\b(previsão do tempo|clima amanhã)\b", re.I), "meteorologia"),
]

# Pedidos explícitos de prescrição direta ao assistente.
PRESCRIPTION_REQUEST_PATTERNS = [
    re.compile(r"\b(prescreva|receite|me\s+d[êe]\s+a\s+receita|passe\s+a\s+receita)\b", re.I),
    re.compile(r"\bqual\s+(a\s+)?dose\s+(exata|que\s+devo\s+dar|que\s+eu\s+dou)\b", re.I),
    re.compile(r"\bpode\s+(prescrever|receitar)\b", re.I),
    re.compile(r"\bj[áa]\s+(prescreve|receita)\s+(pra|para)\s+mim\b", re.I),
]

# Tentativas de burlar as instruções de segurança.
INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(as\s+)?(instru[çc][õo]es|regras|orienta[çc][õo]es)\b", re.I),
    re.compile(r"\besque[çc]a\s+(as\s+)?(instru[çc][õo]es|regras)\b", re.I),
    re.compile(r"\b(aja|atue|finja|comporte-se)\s+como\s+(se\s+)?(voc[êe]\s+)?(fosse|um)\b", re.I),
    re.compile(r"\bsem\s+(o\s+)?(aviso|disclaimer|valida[çc][ãa]o)\b", re.I),
    re.compile(r"\bvoc[êe]\s+(pode|deve)\s+prescrever\s+sim\b", re.I),
    re.compile(r"\bdesconsidere\s+(o\s+)?(protocolo|prompt|sistema)\b", re.I),
]

# Prescrição direta na SAÍDA: posologia concreta em modo imperativo.
# Exige a combinação de um verbo imperativo de administração com uma dose
# numérica — mencionar que "o protocolo recomenda metformina" é legítimo e
# não deve ser bloqueado.
DOSAGE_RE = re.compile(
    r"\b\d+[\.,]?\d*\s*(mg|g|ml|mcg|µg|ui|mg/kg|g/dia|mg/dia|comprimidos?|c[áa]psulas?|gotas?)\b",
    re.I,
)
IMPERATIVE_ADMIN_RE = re.compile(
    r"\b(administre|prescreva|receite|d[êe]\s+ao\s+paciente|inicie\s+com|"
    r"tome|tomar|use\s+agora|aplique)\b",
    re.I,
)


def _normalize(text: str) -> str:
    return text.lower().strip()


def check_input_scope(question: str) -> GuardrailResult:
    """Validação pré-LLM da pergunta recebida."""
    normalized = _normalize(question)

    for pattern in INJECTION_PATTERNS:
        if pattern.search(question):
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=GuardrailReason.PROMPT_INJECTION,
                message=OUT_OF_SCOPE_RESPONSE,
                matched_patterns=[pattern.pattern],
            )

    for pattern, tema in OUT_OF_SCOPE_PATTERNS:
        if pattern.search(question):
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=GuardrailReason.OUT_OF_SCOPE,
                message=OUT_OF_SCOPE_RESPONSE,
                matched_patterns=[f"tema:{tema}"],
            )

    for pattern in PRESCRIPTION_REQUEST_PATTERNS:
        if pattern.search(question):
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=GuardrailReason.DIRECT_PRESCRIPTION_REQUEST,
                message=PRESCRIPTION_BLOCKED_RESPONSE,
                matched_patterns=[pattern.pattern],
            )

    # Exige ao menos um termo clínico: uma pergunta sem nenhuma âncora no
    # domínio quase certamente está fora do escopo dos protocolos internos.
    if not any(term in normalized for term in CLINICAL_TERMS):
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            reason=GuardrailReason.OUT_OF_SCOPE,
            message=OUT_OF_SCOPE_RESPONSE,
            matched_patterns=["sem_termo_clinico"],
        )

    return GuardrailResult(action=GuardrailAction.ALLOW, reason=GuardrailReason.OK)


def check_output_safety(answer: str) -> GuardrailResult:
    """Validação pós-LLM: bloqueia prescrição direta na resposta gerada."""
    has_dosage = DOSAGE_RE.search(answer)
    has_imperative = IMPERATIVE_ADMIN_RE.search(answer)

    if has_dosage and has_imperative:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            reason=GuardrailReason.DIRECT_PRESCRIPTION_OUTPUT,
            message=PRESCRIPTION_BLOCKED_RESPONSE,
            matched_patterns=[has_dosage.group(0), has_imperative.group(0)],
        )

    return GuardrailResult(action=GuardrailAction.ALLOW, reason=GuardrailReason.OK)


def enforce_human_validation(answer: str) -> str:
    """Garante o aviso de validação humana em toda resposta liberada."""
    if "VALIDAÇÃO HUMANA OBRIGATÓRIA" in answer:
        return answer
    return f"{answer.rstrip()}\n\n{HUMAN_VALIDATION_NOTICE}"


__all__ = [
    "GuardrailAction",
    "GuardrailReason",
    "GuardrailResult",
    "HUMAN_VALIDATION_NOTICE",
    "OUT_OF_SCOPE_RESPONSE",
    "PRESCRIPTION_BLOCKED_RESPONSE",
    "check_input_scope",
    "check_output_safety",
    "enforce_human_validation",
]
