"""Avaliação da recuperação (RAG) do assistente.

Mede acurácia top-1 e recall@k do vector store contra um conjunto de
perguntas clínicas com o protocolo correto anotado manualmente. Complementa
a avaliação da LLM (`finetuning/evaluate.py`): se a recuperação erra, nenhuma
qualidade do modelo salva a resposta, então vale medir as duas coisas
separadamente no relatório técnico.

Uso:
    python -m assistant.evaluate_rag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assistant.rag import ProtocolRetriever

#: (pergunta, protocolo esperado) — gabarito anotado manualmente a partir do
#: banco de protocolos da Etapa 1.
BENCHMARK_QUERIES: list[tuple[str, str]] = [
    ("paciente com pressão alta, quais exames solicitar?", "PROT-CARD-001"),
    ("escore CHA2DS2-VASc anticoagulação fibrilação atrial", "PROT-CARD-002"),
    ("qual meta de hemoglobina glicada no diabetes tipo 2", "PROT-END-001"),
    ("escore CURB-65 para pneumonia adquirida na comunidade", "PROT-PNE-001"),
    ("exacerbação de DPOC, alvo de saturação de oxigênio", "PROT-PNE-002"),
    ("paciente com déficit neurológico agudo, janela de trombólise", "PROT-NEU-001"),
    ("sinais de sepse e pacote de 1 hora", "PROT-INF-001"),
    ("febre em lactente menor de 3 meses", "PROT-PED-001"),
    ("rastreamento de diabetes gestacional no pré-natal", "PROT-GIN-001"),
    ("dor lombar aguda, preciso pedir ressonância?", "PROT-ORT-001"),
    ("quando internar um paciente com pneumonia", "PROT-PNE-001"),
    ("primeira linha de tratamento para hipertensão", "PROT-CARD-001"),
    ("critérios de encaminhamento para endocrinologia", "PROT-END-001"),
    ("sinais de alarme na gestação", "PROT-GIN-001"),
    ("red flags na lombalgia", "PROT-ORT-001"),
]


def evaluate_retrieval(
    retriever: ProtocolRetriever, k: int = 3, queries: list[tuple[str, str]] | None = None
) -> dict:
    queries = queries or BENCHMARK_QUERIES
    top1_hits = 0
    topk_hits = 0
    details = []

    for question, expected in queries:
        chunks = retriever.retrieve(question, k=k)
        retrieved_ids = [c.protocol_id for c in chunks]
        top1 = bool(retrieved_ids) and retrieved_ids[0] == expected
        topk = expected in retrieved_ids
        top1_hits += int(top1)
        topk_hits += int(topk)
        details.append(
            {
                "pergunta": question,
                "esperado": expected,
                "recuperados": retrieved_ids,
                "top1_correto": top1,
                f"recall_at_{k}": topk,
            }
        )

    n = len(queries)
    return {
        "n_queries": n,
        "k": k,
        "top1_accuracy": round(top1_hits / n, 4) if n else 0.0,
        f"recall_at_{k}": round(topk_hits / n, 4) if n else 0.0,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols-dir", type=Path, default=Path("data/protocols"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("docs/rag_evaluation.json"))
    args = parser.parse_args()

    retriever = ProtocolRetriever.from_protocols(args.protocols_dir)
    results = evaluate_retrieval(retriever, k=args.k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Consultas avaliadas: {results['n_queries']}")
    print(f"Acurácia top-1: {results['top1_accuracy']:.1%}")
    print(f"Recall@{args.k}: {results[f'recall_at_{args.k}']:.1%}")
    print(f"Detalhes salvos em {args.output}")

    errors = [d for d in results["details"] if not d["top1_correto"]]
    if errors:
        print("\nConsultas com top-1 incorreto:")
        for d in errors:
            print(f"  - {d['pergunta']!r}: esperado {d['esperado']}, veio {d['recuperados']}")


if __name__ == "__main__":
    main()
