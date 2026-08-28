"""Localização e validação dos artefatos de treino (checkpoints e adapters).

Compartilhado entre `finetuning/` (treino e avaliação) e `assistant/llm.py`
(inferência), para que o assistente e a avaliação encontrem os adapters da
mesma forma.
"""

from __future__ import annotations

from pathlib import Path

#: Arquivo que identifica um diretório como contendo adapters PEFT.
ADAPTER_CONFIG_NAME = "adapter_config.json"


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Último checkpoint salvo em `output_dir`, se houver.

    O Trainer salva em subpastas `checkpoint-<passo global>`; ordenamos pelo
    número do passo, não alfabeticamente (senão `checkpoint-90` viria depois
    de `checkpoint-135`).
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None

    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))

    if not checkpoints:
        return None
    return max(checkpoints)[1]


def resolve_adapter_dir(adapter_dir: str | Path) -> str:
    """Valida o diretório de adapters e devolve o caminho a carregar.

    Sem esta validação, um caminho local inexistente é repassado ao PEFT, que
    o interpreta como identificador de repositório da Hugging Face Hub e
    devolve `HFValidationError: Repo id must be in the form ...` — uma
    mensagem que não diz ao usuário o que de fato aconteceu (o diretório não
    existe, ou o treino gravou em outro lugar).

    Se o diretório não tiver `adapter_config.json` mas contiver checkpoints,
    usa o checkpoint mais recente: é o caso de um treino interrompido antes
    de salvar o modelo final, situação comum no Colab.
    """
    path = Path(adapter_dir)

    if (path / ADAPTER_CONFIG_NAME).is_file():
        return str(path)

    if path.is_dir():
        latest = find_latest_checkpoint(path)
        if latest is not None and (latest / ADAPTER_CONFIG_NAME).is_file():
            print(
                f"[adapters] {path} não tem {ADAPTER_CONFIG_NAME} (treino não "
                f"finalizado?). Usando o checkpoint mais recente: {latest.name}"
            )
            return str(latest)

        conteudo = sorted(p.name for p in path.iterdir())[:10]
        raise FileNotFoundError(
            f"O diretório '{path}' existe mas não contém '{ADAPTER_CONFIG_NAME}' "
            f"nem nenhum checkpoint válido.\nConteúdo: {conteudo or '(vazio)'}\n"
            "Verifique se o treino terminou e se --adapter-dir aponta para o "
            "mesmo caminho usado em --output-dir no treino."
        )

    raise FileNotFoundError(
        f"Diretório de adapters não encontrado: '{path}'.\n"
        "Causas comuns:\n"
        "  • o treino gravou em outro caminho — se você usou --output-dir "
        "(por exemplo no Google Drive), passe o MESMO caminho em --adapter-dir;\n"
        "  • o treino ainda não foi executado;\n"
        "  • a máquina do Colab foi reciclada e /content/ foi apagado."
    )


__all__ = ["ADAPTER_CONFIG_NAME", "find_latest_checkpoint", "resolve_adapter_dir"]
