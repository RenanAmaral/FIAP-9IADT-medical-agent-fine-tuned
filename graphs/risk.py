"""Detecção de critérios de risco clínico (Etapa 4, nó de Verificação/Alerta).

Extrai sinais vitais do texto das evoluções e resultados de exames do
prontuário e aplica os critérios de gravidade que os próprios protocolos
internos definem — qSOFA (PROT-INF-001), hipoxemia (PROT-PNE-002), tempo
crítico em AVC (PROT-NEU-001), entre outros.

Deliberadamente determinístico e independente do LLM: a decisão de acionar a
equipe médica não deve depender de o modelo ter gerado o texto certo. O LLM
compõe a explicação; quem decide se há risco é este módulo.

Os limiares abaixo replicam os dos protocolos sintéticos deste projeto e não
constituem referência clínica real.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Extração de sinais vitais -------------------------------------------
PA_RE = re.compile(r"PA\s*:?\s*(\d{2,3})\s*/\s*(\d{2,3})", re.I)
FC_RE = re.compile(r"FC\s*:?\s*(\d{2,3})", re.I)
FR_RE = re.compile(r"FR\s*:?\s*(\d{1,3})", re.I)
SPO2_RE = re.compile(r"SpO2\s*:?\s*(\d{2,3})", re.I)
TAX_RE = re.compile(r"Tax\s*:?\s*(\d{2,3})[,.]?(\d)?\s*°?C", re.I)
#: O valor de lactato é procurado dentro de uma linha que mencione "lactato",
#: e não por proximidade de caracteres: no prontuário a linha do exame traz
#: datas entre o nome e o resultado ("Lactato sérico: CONCLUÍDO em 2026-08-18
#: — resultado: 4,2 mmol/L"), o que quebraria um regex baseado em distância.
LACTATO_VALUE_RE = re.compile(r"(\d+)[,.](\d+)\s*mmol", re.I)

# --- Termos de gravidade em texto livre -----------------------------------
CONSCIOUSNESS_TERMS = [
    "confuso", "confusão", "rebaixamento", "letárgico", "letargia",
    "sonolento", "torporoso", "desorientado", "alteração do nível de consciência",
]
CRITICAL_CONDITION_TERMS = {
    "sepse": "suspeita de sepse registrada no prontuário",
    "choque séptico": "choque séptico registrado no prontuário",
    "avc": "suspeita de AVC — fluxo de tempo crítico (PROT-NEU-001)",
    "déficit neurológico": "déficit neurológico agudo — avaliar AVC (PROT-NEU-001)",
    "cauda equina": "suspeita de síndrome da cauda equina — emergência cirúrgica",
    "emergência hipertensiva": "emergência hipertensiva registrada",
}

# Limiares (espelham os protocolos internos deste projeto)
PAS_HIPOTENSAO = 100        # qSOFA — PROT-INF-001
FR_TAQUIPNEIA = 22          # qSOFA — PROT-INF-001
SPO2_HIPOXEMIA = 92         # PROT-PNE-002
TAX_FEBRE_ALTA = 38.5
LACTATO_CRITICO = 4.0       # PROT-INF-001
PAS_EMERGENCIA_HIPERTENSIVA = 180
PAD_EMERGENCIA_HIPERTENSIVA = 120


@dataclass
class VitalSigns:
    pas: int | None = None
    pad: int | None = None
    fc: int | None = None
    fr: int | None = None
    spo2: int | None = None
    temperatura: float | None = None
    lactato: float | None = None


@dataclass
class RiskAssessment:
    nivel: str  # "critico" | "alto" | "moderado" | "baixo"
    criterios: list[str] = field(default_factory=list)
    qsofa_score: int = 0
    vitais: VitalSigns = field(default_factory=VitalSigns)

    @property
    def requer_alerta(self) -> bool:
        return self.nivel in {"critico", "alto"}


def extract_vitals(texto: str) -> VitalSigns:
    vitals = VitalSigns()

    if (m := PA_RE.search(texto)):
        vitals.pas, vitals.pad = int(m.group(1)), int(m.group(2))
    if (m := FC_RE.search(texto)):
        vitals.fc = int(m.group(1))
    if (m := FR_RE.search(texto)):
        vitals.fr = int(m.group(1))
    if (m := SPO2_RE.search(texto)):
        vitals.spo2 = int(m.group(1))
    if (m := TAX_RE.search(texto)):
        vitals.temperatura = float(f"{m.group(1)}.{m.group(2) or 0}")

    for linha in texto.splitlines():
        if "lactato" in linha.lower() and (m := LACTATO_VALUE_RE.search(linha)):
            vitals.lactato = float(f"{m.group(1)}.{m.group(2)}")
            break

    return vitals


def assess_risk(texto_prontuario: str) -> RiskAssessment:
    """Avalia o risco a partir do texto consolidado do prontuário."""
    texto_lower = texto_prontuario.lower()
    vitals = extract_vitals(texto_prontuario)
    criterios: list[str] = []

    # --- qSOFA (PROT-INF-001) --------------------------------------------
    qsofa = 0
    tem_alteracao_consciencia = any(t in texto_lower for t in CONSCIOUSNESS_TERMS)
    if tem_alteracao_consciencia:
        qsofa += 1
        criterios.append("alteração do nível de consciência (qSOFA)")
    if vitals.fr is not None and vitals.fr >= FR_TAQUIPNEIA:
        qsofa += 1
        criterios.append(f"frequência respiratória {vitals.fr} irpm (>= {FR_TAQUIPNEIA}, qSOFA)")
    if vitals.pas is not None and vitals.pas <= PAS_HIPOTENSAO:
        qsofa += 1
        criterios.append(f"pressão sistólica {vitals.pas} mmHg (<= {PAS_HIPOTENSAO}, qSOFA)")

    # --- Outros critérios de gravidade ------------------------------------
    if vitals.spo2 is not None and vitals.spo2 < SPO2_HIPOXEMIA:
        criterios.append(f"saturação de O2 {vitals.spo2}% (< {SPO2_HIPOXEMIA}%)")
    if vitals.lactato is not None and vitals.lactato >= LACTATO_CRITICO:
        criterios.append(f"lactato {vitals.lactato} mmol/L (>= {LACTATO_CRITICO}, PROT-INF-001)")
    if vitals.temperatura is not None and vitals.temperatura >= TAX_FEBRE_ALTA:
        criterios.append(f"temperatura {vitals.temperatura}°C")
    if (
        vitals.pas is not None
        and vitals.pad is not None
        and vitals.pas >= PAS_EMERGENCIA_HIPERTENSIVA
        and vitals.pad >= PAD_EMERGENCIA_HIPERTENSIVA
    ):
        criterios.append(
            f"PA {vitals.pas}/{vitals.pad} mmHg — possível emergência hipertensiva (PROT-CARD-001)"
        )

    condicoes_criticas = [
        descricao for termo, descricao in CRITICAL_CONDITION_TERMS.items() if termo in texto_lower
    ]
    criterios.extend(condicoes_criticas)

    # --- Classificação ----------------------------------------------------
    # Crítico: qSOFA >= 2 (critério formal de sepse), lactato crítico, ou
    # menção explícita a uma condição de tempo crítico.
    if qsofa >= 2 or condicoes_criticas or (vitals.lactato or 0) >= LACTATO_CRITICO:
        nivel = "critico"
    elif qsofa == 1 or (vitals.spo2 is not None and vitals.spo2 < SPO2_HIPOXEMIA):
        nivel = "alto"
    elif criterios:
        nivel = "moderado"
    else:
        nivel = "baixo"

    return RiskAssessment(nivel=nivel, criterios=criterios, qsofa_score=qsofa, vitais=vitals)


__all__ = ["RiskAssessment", "VitalSigns", "assess_risk", "extract_vitals"]
