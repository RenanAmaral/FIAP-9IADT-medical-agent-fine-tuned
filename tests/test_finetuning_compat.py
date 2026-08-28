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


# --------------------------------------------------------------------------
# Seleção de hardware e dtype
# --------------------------------------------------------------------------


def test_default_compute_dtype_is_auto():
    """Fixar bfloat16 quebraria na T4 do Colab, que é Turing e não tem bf16
    nativo. O padrão precisa ser resolvido em runtime."""
    assert TrainingConfig().bnb_compute_dtype == "auto"


def test_compute_dtype_is_exposed_on_cli():
    from finetuning.train import _build_arg_parser, build_config_from_args

    args = _build_arg_parser().parse_args(["--compute-dtype", "float16"])
    assert build_config_from_args(args).bnb_compute_dtype == "float16"


def test_accelerator_check_is_safe_without_torch(capsys):
    """Nunca pode levantar exceção: roda antes do treino, inclusive em
    ambientes sem torch."""
    from finetuning.train import check_accelerator

    check_accelerator()
    capsys.readouterr()


# --------------------------------------------------------------------------
# Retomada de treino interrompido
# --------------------------------------------------------------------------


def test_latest_checkpoint_is_picked_numerically(tmp_path):
    """checkpoint-135 é mais recente que checkpoint-90, mas vem antes na
    ordem alfabética — a comparação tem que ser numérica."""
    from finetuning.train import find_latest_checkpoint

    for step in (15, 45, 90, 135):
        (tmp_path / f"checkpoint-{step}").mkdir()

    assert find_latest_checkpoint(tmp_path).name == "checkpoint-135"


def test_non_checkpoint_entries_are_ignored(tmp_path):
    from finetuning.train import find_latest_checkpoint

    (tmp_path / "checkpoint-30").mkdir()
    (tmp_path / "checkpoint-invalido").mkdir()
    (tmp_path / "checkpoint-99.txt").write_text("x")

    assert find_latest_checkpoint(tmp_path).name == "checkpoint-30"


def test_no_checkpoints_returns_none(tmp_path):
    from finetuning.train import find_latest_checkpoint

    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "nao-existe") is None


def test_resume_auto_without_checkpoints_trains_from_scratch(tmp_path, capsys):
    """Retomar é conveniência: sem checkpoint, treina do zero em vez de
    falhar."""
    from finetuning.train import resolve_resume_target

    result = resolve_resume_target(TrainingConfig(output_dir=str(tmp_path)), "auto")
    assert result is None
    assert "do zero" in capsys.readouterr().out


def test_resume_auto_finds_checkpoint(tmp_path):
    from finetuning.train import resolve_resume_target

    (tmp_path / "checkpoint-135").mkdir()
    result = resolve_resume_target(TrainingConfig(output_dir=str(tmp_path)), "auto")
    assert result.endswith("checkpoint-135")


def test_resume_with_explicit_path(tmp_path):
    from finetuning.train import resolve_resume_target

    ckpt = tmp_path / "checkpoint-45"
    ckpt.mkdir()
    assert resolve_resume_target(TrainingConfig(), str(ckpt)) == str(ckpt)


def test_resume_with_missing_path_raises():
    """Caminho explícito e inexistente é erro do usuário — falhar aqui é
    melhor que treinar do zero em silêncio."""
    from finetuning.train import resolve_resume_target

    with pytest.raises(FileNotFoundError):
        resolve_resume_target(TrainingConfig(), "/caminho/que/nao/existe")


def test_no_resume_flag_means_fresh_training():
    from finetuning.train import resolve_resume_target

    assert resolve_resume_target(TrainingConfig(), None) is None


def test_resume_flag_parsing():
    from finetuning.train import _build_arg_parser

    parser = _build_arg_parser()
    assert parser.parse_args([]).resume is None
    assert parser.parse_args(["--resume"]).resume == "auto"
    assert parser.parse_args(["--resume", "/x/checkpoint-1"]).resume == "/x/checkpoint-1"


# --------------------------------------------------------------------------
# Resolução do diretório de adapters
# --------------------------------------------------------------------------


def test_adapter_dir_with_config_is_used_directly(tmp_path):
    from finetuning.paths import resolve_adapter_dir

    (tmp_path / "adapter_config.json").write_text("{}")
    assert resolve_adapter_dir(tmp_path) == str(tmp_path)


def test_adapter_dir_falls_back_to_latest_checkpoint(tmp_path, capsys):
    """Treino interrompido antes de salvar o modelo final: só há checkpoints."""
    from finetuning.paths import resolve_adapter_dir

    for step in (45, 135):
        ckpt = tmp_path / f"checkpoint-{step}"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text("{}")

    assert resolve_adapter_dir(tmp_path).endswith("checkpoint-135")
    assert "checkpoint-135" in capsys.readouterr().out


def test_missing_adapter_dir_names_the_likely_cause(tmp_path):
    """A mensagem precisa apontar o descompasso de caminho — sem ela, o PEFT
    trata o caminho local como repo da Hub e devolve um HFValidationError
    incompreensível."""
    from finetuning.paths import resolve_adapter_dir

    with pytest.raises(FileNotFoundError) as exc:
        resolve_adapter_dir(tmp_path / "nao-existe")

    message = str(exc.value)
    assert "--output-dir" in message
    assert "--adapter-dir" in message


def test_existing_but_empty_adapter_dir_reports_contents(tmp_path):
    from finetuning.paths import resolve_adapter_dir

    (tmp_path / "algum_arquivo.txt").write_text("x")

    with pytest.raises(FileNotFoundError) as exc:
        resolve_adapter_dir(tmp_path)

    assert "algum_arquivo.txt" in str(exc.value)


def test_checkpoint_without_adapter_config_is_not_used(tmp_path):
    """Um checkpoint incompleto não deve ser aceito silenciosamente."""
    from finetuning.paths import resolve_adapter_dir

    (tmp_path / "checkpoint-15").mkdir()

    with pytest.raises(FileNotFoundError):
        resolve_adapter_dir(tmp_path)


@pytest.mark.parametrize("module", ["assistant.cli", "graphs.cli"])
def test_adapter_dir_flag_is_available_on_clis(module):
    """As duas CLIs precisam aceitar --adapter-dir: sem isso, um treino que
    gravou no Drive não tem como ser usado na demonstração."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--adapter-dir" in result.stdout


def test_volatile_output_warning_is_silent_outside_colab(capsys):
    from finetuning.train import warn_if_output_is_volatile

    warn_if_output_is_volatile("/tmp/qualquer")
    assert capsys.readouterr().out == ""


def test_volatile_output_warning_fires_for_content_paths(monkeypatch, capsys):
    """No Colab, gravar em /content/ significa perder o treino quando a
    máquina é reciclada — o usuário precisa saber ANTES de gastar a GPU."""
    import pathlib

    from finetuning import train as train_mod

    real_is_dir = pathlib.Path.is_dir
    monkeypatch.setattr(
        pathlib.Path,
        "is_dir",
        lambda self: True if str(self) == "/content" else real_is_dir(self),
    )

    train_mod.warn_if_output_is_volatile("/content/repo/finetuning/adapters/x")
    out = capsys.readouterr().out
    assert "[aviso]" in out
    assert "drive.mount" in out


def test_drive_output_is_reported_as_persistent(monkeypatch, capsys):
    import pathlib

    from finetuning import train as train_mod

    real_is_dir = pathlib.Path.is_dir
    monkeypatch.setattr(
        pathlib.Path,
        "is_dir",
        lambda self: True if str(self) == "/content" else real_is_dir(self),
    )

    train_mod.warn_if_output_is_volatile("/content/drive/MyDrive/tc3/adapters")
    out = capsys.readouterr().out
    assert "[aviso]" not in out
    assert "persiste" in out


# --------------------------------------------------------------------------
# Conflito de versão do torchao
# --------------------------------------------------------------------------


def test_torchao_check_passes_when_absent(monkeypatch):
    """Sem torchao instalado, o PEFT simplesmente pula o dispatcher."""
    from finetuning import environment

    monkeypatch.setattr(environment, "_installed_torchao_version", lambda: None)
    environment.check_torchao_conflict()


def test_torchao_check_passes_on_recent_version(monkeypatch):
    from finetuning import environment

    monkeypatch.setattr(environment, "_installed_torchao_version", lambda: (0, 16, 0))
    environment.check_torchao_conflict()


def test_torchao_check_fails_on_old_version(monkeypatch):
    """torchao 0.10 é o que o Colab traz; o PEFT levanta ImportError ao
    encontrá-lo, e queremos falhar antes de carregar o modelo."""
    from finetuning import environment

    monkeypatch.setattr(environment, "_installed_torchao_version", lambda: (0, 10, 0))

    with pytest.raises(RuntimeError) as exc:
        environment.check_torchao_conflict()

    message = str(exc.value)
    assert "0.10.0" in message
    assert "pip uninstall -y torchao" in message


def test_torchao_version_parsing_tolerates_suffixes(monkeypatch):
    """Versões como '0.10.0+cu121' ou '0.17.0rc1' não podem quebrar a checagem."""
    import importlib.metadata

    from finetuning import environment

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.10.0+cu121")
    assert environment._installed_torchao_version() == (0, 10, 0)
