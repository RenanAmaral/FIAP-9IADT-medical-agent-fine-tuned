"""Prompts do assistente médico.

O `SYSTEM_PROMPT` vem de `finetuning/config.py` — mesma string usada no
treino da Etapa 2 — para que o modelo fine-tuned receba na inferência
exatamente o formato em que foi treinado. É o contrato de dados combinado
entre as duas frentes do projeto.

Os limites de atuação descritos no system prompt (não prescrever, exigir
validação humana, recusar fora de escopo) são a **primeira** camada de
guardrail, exigida pelo item 12 do enunciado. A segunda camada, programática
e independente do modelo, está em `security/guardrails.py` — prompt sozinho
não é um controle de segurança confiável.
"""

from __future__ import annotations

from finetuning.config import SYSTEM_PROMPT, format_prompt

CLINICAL_QUERY_TEMPLATE = """Pergunta do profissional de saúde:
{pergunta}

=== DADOS ATUAIS DO PACIENTE (prontuário) ===
{contexto_paciente}

=== PROTOCOLOS INTERNOS RELEVANTES ===
{contexto_protocolos}

Responda à pergunta considerando SIMULTANEAMENTE os protocolos internos e os
dados atuais do paciente acima. Cite explicitamente o identificador do
protocolo que embasou cada orientação. Se houver exames pendentes relevantes,
sinalize que a conduta definitiva depende deles."""

NO_PATIENT_TEMPLATE = """Pergunta do profissional de saúde:
{pergunta}

=== PROTOCOLOS INTERNOS RELEVANTES ===
{contexto_protocolos}

Responda à pergunta com base nos protocolos internos acima, citando
explicitamente o identificador do protocolo que embasou cada orientação."""


def build_clinical_prompt(
    pergunta: str,
    contexto_protocolos: str,
    contexto_paciente: str | None = None,
) -> str:
    """Monta o prompt final no formato de treino (`format_prompt`)."""
    if contexto_paciente:
        instruction = CLINICAL_QUERY_TEMPLATE.format(
            pergunta=pergunta,
            contexto_paciente=contexto_paciente,
            contexto_protocolos=contexto_protocolos,
        )
    else:
        instruction = NO_PATIENT_TEMPLATE.format(
            pergunta=pergunta,
            contexto_protocolos=contexto_protocolos,
        )
    return format_prompt(instruction=instruction, system_prompt=SYSTEM_PROMPT)


__all__ = [
    "CLINICAL_QUERY_TEMPLATE",
    "NO_PATIENT_TEMPLATE",
    "SYSTEM_PROMPT",
    "build_clinical_prompt",
    "format_prompt",
]
