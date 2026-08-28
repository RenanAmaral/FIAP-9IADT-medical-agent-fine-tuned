"""CLI do assistente médico — consulta direta à chain (Etapa 3).

Para executar o fluxo clínico completo com o grafo de decisão (Etapa 4),
use `python -m graphs.cli`.

Exemplos:
    # Pergunta geral, sem paciente
    python -m assistant.cli --pergunta "Qual o escore usado para gravidade em pneumonia?"

    # Pergunta contextualizada em um paciente
    python -m assistant.cli --paciente PAC-0002 --pergunta "Posso ajustar o tratamento?"

    # Usando o modelo com fine-tuning (requer os adapters da Etapa 2)
    python -m assistant.cli --backend finetuned --paciente PAC-0003 \
        --pergunta "Qual a conduta para este paciente?"

    # Modo interativo
    python -m assistant.cli --interativo --paciente PAC-0001
"""

from __future__ import annotations

import argparse
from pathlib import Path

from assistant.chains import build_assistant
from assistant.database import DEFAULT_DB_PATH, connect, list_patients


def _print_response(response) -> None:
    print("=" * 78)
    print(response.texto_completo)
    print("=" * 78)
    print(
        f"[sessão {response.session_id} | {response.duracao_ms:.0f} ms"
        + (f" | BLOQUEADO: {response.motivo_bloqueio}" if response.bloqueado else "")
        + "]"
    )


def _list_patients(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        patients = list_patients(conn)
    finally:
        conn.close()

    if not patients:
        print("Nenhum paciente na base. Rode: python -m assistant.database")
        return

    print("Pacientes disponíveis:")
    for p in patients:
        print(f"  {p.codigo_paciente} — {p.idade} anos, {p.condicao_principal} ({p.especialidade})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pergunta", help="Pergunta clínica a ser respondida.")
    parser.add_argument("--paciente", help="Código do paciente (ex.: PAC-0002).")
    parser.add_argument(
        "--backend",
        default="template",
        choices=["template", "base", "finetuned"],
        help="Backend da LLM. 'template' é o stub offline de desenvolvimento; "
        "'finetuned' usa os adapters da Etapa 2 (requer GPU/modelo baixado).",
    )
    parser.add_argument(
        "--adapter-dir",
        default="finetuning/adapters/medical-assistant-lora",
        help="Diretório dos adapters LoRA (usado com --backend finetuned). "
        "Se o treino gravou em outro caminho (ex.: Google Drive), passe-o aqui.",
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--protocols-dir", type=Path, default=Path("data/protocols"))
    parser.add_argument("--log-path", type=Path, default=Path("logs/audit.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--interativo", action="store_true", help="Modo de perguntas em loop.")
    parser.add_argument("--listar-pacientes", action="store_true")
    args = parser.parse_args()

    if args.listar_pacientes:
        _list_patients(args.db_path)
        return

    if not args.pergunta and not args.interativo:
        parser.error("informe --pergunta ou use --interativo (ou --listar-pacientes)")

    if not args.db_path.exists():
        parser.error(
            f"Base estruturada não encontrada em {args.db_path}. "
            "Rode primeiro: python -m assistant.database"
        )

    # adapter_dir só faz sentido para o backend com fine-tuning; passá-lo aos
    # demais quebraria a construção da LLM.
    llm_kwargs = {"adapter_dir": args.adapter_dir} if args.backend == "finetuned" else {}

    assistant = build_assistant(
        backend=args.backend,
        db_path=args.db_path,
        protocols_dir=args.protocols_dir,
        log_path=args.log_path,
        **llm_kwargs,
    )
    assistant.top_k = args.top_k

    if args.pergunta:
        _print_response(assistant.run(args.pergunta, codigo_paciente=args.paciente))

    if args.interativo:
        print("\nModo interativo. Digite 'sair' para encerrar.")
        if args.paciente:
            print(f"Contexto: paciente {args.paciente}")
        while True:
            try:
                pergunta = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if pergunta.lower() in {"sair", "exit", "quit", ""}:
                break
            _print_response(assistant.run(pergunta, codigo_paciente=args.paciente))


if __name__ == "__main__":
    main()
