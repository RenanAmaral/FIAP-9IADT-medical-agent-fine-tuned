"""Etapa 5, item 14 — Explainability.

Toda resposta do assistente indica a fonte que a embasou (protocolo interno,
documento ou registro de prontuário) e um grau de confiança.

Sobre o grau de confiança
-------------------------
É um score de **recuperação**, não de correção clínica. Deriva de:

- a similaridade dos trechos de protocolo recuperados em relação à pergunta
  (média dos scores do vector store, o sinal dominante);
- se havia dados do paciente disponíveis para contextualizar;
- uma penalidade quando existem exames pendentes, já que nesse caso o quadro
  está incompleto por definição.

Ou seja: mede o quanto o assistente encontrou base documental e contexto para
responder — não afirma que a conduta sugerida está clinicamente correta. Essa
distinção é explicitada junto ao número na resposta, para não induzir o
profissional a ler o score como um aval clínico.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from assistant.rag import RetrievedChunk


@dataclass
class Source:
    tipo: str  # "protocolo" | "prontuario"
    identificador: str
    descricao: str
    score: float | None = None

    def to_dict(self) -> dict:
        return {
            "tipo": self.tipo,
            "identificador": self.identificador,
            "descricao": self.descricao,
            "score": self.score,
        }

    def to_line(self) -> str:
        if self.score is not None:
            return f"- [{self.tipo}] {self.identificador} — {self.descricao} (relevância {self.score:.2f})"
        return f"- [{self.tipo}] {self.identificador} — {self.descricao}"


@dataclass
class Explanation:
    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0
    confidence_label: str = "baixa"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
            "notes": self.notes,
        }

    def to_block(self) -> str:
        """Bloco de explicabilidade anexado ao final da resposta."""
        lines = ["", "---", "**Fontes consultadas:**"]
        if self.sources:
            lines.extend(s.to_line() for s in self.sources)
        else:
            lines.append("- Nenhuma fonte interna foi recuperada para esta consulta.")

        lines.append("")
        lines.append(
            f"**Grau de confiança na recuperação: {self.confidence:.0%} "
            f"({self.confidence_label})** — mede o quanto o assistente encontrou "
            "base documental e contexto do paciente para responder; não é uma "
            "avaliação de correção clínica."
        )
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"> {note}")
        return "\n".join(lines)


def _label_for(confidence: float) -> str:
    if confidence >= 0.7:
        return "alta"
    if confidence >= 0.4:
        return "média"
    return "baixa"


def build_explanation(
    chunks: list[RetrievedChunk],
    codigo_paciente: str | None = None,
    tem_dados_paciente: bool = False,
    exames_pendentes: list[str] | None = None,
) -> Explanation:
    exames_pendentes = exames_pendentes or []
    sources: list[Source] = []
    notes: list[str] = []

    # Uma fonte por protocolo (o mesmo protocolo pode ter vários chunks
    # recuperados; agrupamos para não poluir o bloco de fontes).
    seen: dict[str, Source] = {}
    for chunk in chunks:
        existing = seen.get(chunk.protocol_id)
        if existing is None:
            seen[chunk.protocol_id] = Source(
                tipo="protocolo",
                identificador=chunk.protocol_id,
                descricao=chunk.titulo,
                score=chunk.score,
            )
        elif existing.score is not None and chunk.score > existing.score:
            existing.score = chunk.score
    sources.extend(seen.values())

    if tem_dados_paciente and codigo_paciente:
        sources.append(
            Source(
                tipo="prontuario",
                identificador=codigo_paciente,
                descricao="registro estruturado do paciente (dados atuais, exames e evoluções)",
            )
        )

    # Confiança: similaridade média da recuperação é o sinal dominante.
    if chunks:
        retrieval_score = sum(c.score for c in chunks) / len(chunks)
        # Os scores do FAISS/TF-IDF ficam tipicamente em 0,2–0,6 neste corpus;
        # reescalamos para que a faixa útil ocupe o intervalo (0, 1).
        retrieval_component = min(1.0, retrieval_score / 0.5)
    else:
        retrieval_component = 0.0
        notes.append(
            "Nenhum protocolo interno relevante foi recuperado — a resposta não "
            "tem respaldo documental interno."
        )

    confidence = 0.7 * retrieval_component
    if tem_dados_paciente:
        confidence += 0.3
    else:
        notes.append(
            "Resposta baseada apenas nos protocolos internos, sem dados de um "
            "paciente específico."
        )

    if exames_pendentes:
        confidence *= 0.75
        notes.append(
            "Há exames pendentes ("
            + ", ".join(exames_pendentes)
            + "), portanto o quadro clínico está incompleto e a conduta "
            "definitiva depende desses resultados."
        )

    confidence = max(0.0, min(1.0, confidence))

    return Explanation(
        sources=sources,
        confidence=round(confidence, 4),
        confidence_label=_label_for(confidence),
        notes=notes,
    )


__all__ = ["Explanation", "Source", "build_explanation"]
