"""Compatibilidade entre versões de trl/transformers.

O treino roda no Colab/Kaggle, onde as versões das bibliotecas mudam sem
aviso. Estes testes cobrem a tradução de nomes de parâmetro que já quebrou o
script uma vez em produção (`SFTConfig` passou a exigir `max_length` no lugar
de `max_seq_length`).
"""

from dataclasses import dataclass

import pytest

from finetuning.config import TrainingConfig
from finetuning.hf_auth import dtype_kwarg
from finetuning.train import (
    MIN_USEFUL_OPTIMIZER_STEPS,
    _report_training_plan,
    resolve_config_kwargs,
)


@dataclass
class NewApiConfig:
    """trl >= 0.20 + transformers 5.x."""

    output_dir: str = ""
    max_length: int = 1024
    warmup_steps: float = 0.0
    eval_strategy: str = "no"
    dataset_text_field: str = "text"


@dataclass
class OldApiConfig:
    """trl < 0.20 + transformers < 4.41."""

    output_dir: str = ""
    max_seq_length: int = 1024
    warmup_ratio: float = 0.0
    evaluation_strategy: str = "no"
    dataset_text_field: str = "text"


DESIRED = {
    "output_dir": "out",
    "max_length": 1024,
    "warmup_ratio": 0.03,
    "eval_strategy": "epoch",
    "dataset_text_field": "text",
}


def test_new_api_keeps_max_length():
    kwargs, _ = resolve_config_kwargs(NewApiConfig, DESIRED)
    assert kwargs["max_length"] == 1024
    assert "max_seq_length" not in kwargs


def test_old_api_translates_to_max_seq_length():
    """Este é o erro que derrubou o treino no Colab."""
    kwargs, warnings = resolve_config_kwargs(OldApiConfig, DESIRED)
    assert kwargs["max_seq_length"] == 1024
    assert "max_length" not in kwargs
    assert any("max_seq_length" in w for w in warnings)


def test_warmup_ratio_maps_to_warmup_steps_on_transformers_5():
    kwargs, _ = resolve_config_kwargs(NewApiConfig, DESIRED)
    assert kwargs["warmup_steps"] == 0.03
    assert "warmup_ratio" not in kwargs


def test_eval_strategy_translates_for_old_transformers():
    kwargs, _ = resolve_config_kwargs(OldApiConfig, DESIRED)
    assert kwargs["evaluation_strategy"] == "epoch"


@pytest.mark.parametrize("config_cls", [NewApiConfig, OldApiConfig])
def test_resolved_kwargs_actually_construct(config_cls):
    """A garantia que importa: o resultado tem que instanciar sem TypeError."""
    kwargs, _ = resolve_config_kwargs(config_cls, DESIRED)
    config_cls(**kwargs)


@pytest.mark.parametrize("config_cls", [NewApiConfig, OldApiConfig])
def test_unknown_parameter_is_dropped_with_warning(config_cls):
    """Um parâmetro desconhecido deve ser descartado com aviso, nunca
    derrubar um treino que já baixou gigabytes de pesos."""
    kwargs, warnings = resolve_config_kwargs(
        config_cls, {**DESIRED, "parametro_que_nao_existe": 1}
    )
    assert "parametro_que_nao_existe" not in kwargs
    assert any("parametro_que_nao_existe" in w for w in warnings)
    config_cls(**kwargs)


def test_dtype_kwarg_picks_a_single_supported_name():
    kwargs = dtype_kwarg("float16")
    assert len(kwargs) == 1
    assert next(iter(kwargs)) in {"dtype", "torch_dtype"}
    assert next(iter(kwargs.values())) == "float16"


# --------------------------------------------------------------------------
# Plano de treino
# --------------------------------------------------------------------------


def test_default_config_produces_enough_optimizer_steps():
    """Com os padrões atuais o treino precisa render passos suficientes para
    que o modelo saia da inicialização."""
    c = TrainingConfig()
    effective_batch = c.per_device_train_batch_size * c.gradient_accumulation_steps
    steps = max(1, 59 // effective_batch) * c.num_train_epochs
    assert steps >= MIN_USEFUL_OPTIMIZER_STEPS


def test_plan_warns_when_steps_are_too_few(capsys):
    _report_training_plan(
        TrainingConfig(num_train_epochs=3, gradient_accumulation_steps=4), n_train=59
    )
    out = capsys.readouterr().out
    assert "[aviso]" in out


def test_plan_is_quiet_when_steps_are_sufficient(capsys):
    _report_training_plan(TrainingConfig(), n_train=59)
    out = capsys.readouterr().out
    assert "[plano]" in out
    assert "[aviso]" not in out
