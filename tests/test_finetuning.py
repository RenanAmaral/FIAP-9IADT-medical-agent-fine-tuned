import json

from finetuning.config import SYSTEM_PROMPT, format_prompt
from finetuning.dataset import build_training_texts, load_training_dataset


def test_format_prompt_contains_system_and_roles():
    prompt = format_prompt("Qual a conduta?", output="Resposta clínica.")
    assert SYSTEM_PROMPT in prompt
    assert "<|user|>" in prompt
    assert "Qual a conduta?" in prompt
    assert "<|assistant|>" in prompt
    assert prompt.endswith("<|end|>")


def test_format_prompt_without_output_ends_ready_for_generation():
    prompt = format_prompt("Pergunta sem resposta ainda")
    assert prompt.rstrip().endswith("<|assistant|>")
    assert "<|end|>" not in prompt


def test_build_training_texts_matches_record_count():
    records = [
        {"instruction": "Pergunta 1", "input": "", "output": "Resposta 1"},
        {"instruction": "Pergunta 2", "input": "contexto", "output": "Resposta 2"},
    ]
    texts = build_training_texts(records)
    assert len(texts) == 2
    assert "Resposta 1" in texts[0]
    assert "contexto" in texts[1]


def test_load_training_dataset_from_file(tmp_path):
    path = tmp_path / "train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"instruction": "Q", "output": "A que precisa ser validada"}) + "\n")

    dataset = load_training_dataset(path)
    assert len(dataset) == 1
    assert "text" in dataset[0]
    assert "Q" in dataset[0]["text"]
