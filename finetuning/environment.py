"""Checagens de ambiente feitas antes de carregar modelos.

Todas seguem o mesmo princípio: falhar (ou avisar) em segundos, e não depois
de baixar gigabytes de pesos ou gastar minutos de GPU.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import itertools

#: Versão mínima de torchao exigida pelo PEFT recente.
TORCHAO_MINIMUM = (0, 16, 0)


def _installed_torchao_version() -> tuple[int, ...] | None:
    if importlib.util.find_spec("torchao") is None:
        return None
    try:
        raw = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return None

    # Só os dígitos iniciais de cada componente: em "0.10.0+cu121" o terceiro
    # componente é "0+cu121" e vale 0, não 121 — filtrar todos os dígitos
    # juntaria o sufixo da build ao número da versão.
    parts: list[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(itertools.takewhile(str.isdigit, chunk))
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_torchao_conflict() -> None:
    """Falha cedo se o torchao instalado for velho demais para o PEFT.

    O PEFT percorre uma cadeia de dispatchers ao aplicar os adapters LoRA e
    chega em `dispatch_torchao`, que chama `is_torchao_available()`. Essa
    função **levanta ImportError** quando encontra um torchao anterior a
    0.16 — não retorna False. O Colab traz o 0.10 pré-instalado, então o
    carregamento do modelo com adapters quebra ali.

    Curiosamente o treino não é afetado: com o modelo em 4 bits o
    `dispatch_bnb_4bit` casa antes e a cadeia nunca alcança o torchao. O erro
    só aparece na inferência/avaliação, que carrega o modelo sem quantização.

    Este projeto não usa torchao para nada, então a saída mais simples é
    removê-lo do ambiente.
    """
    version = _installed_torchao_version()
    if version is None or version >= TORCHAO_MINIMUM:
        return

    atual = ".".join(str(p) for p in version)
    minimo = ".".join(str(p) for p in TORCHAO_MINIMUM)
    raise RuntimeError(
        f"torchao {atual} instalado, mas o PEFT exige >= {minimo} e levanta "
        "ImportError ao encontrar uma versão anterior.\n"
        "Este projeto não usa torchao. Remova-o e rode de novo:\n"
        "    pip uninstall -y torchao\n"
        f"Ou atualize:  pip install -U 'torchao>={minimo}'"
    )


__all__ = ["TORCHAO_MINIMUM", "check_torchao_conflict"]
