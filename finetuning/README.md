# finetuning/

Implementa a Etapa 2 do Tech Challenge: fine-tuning da LLM com LoRA/QLoRA.

## Por que isso não roda no ambiente de desenvolvimento deste repositório

Este projeto foi desenvolvido em um ambiente sandbox **sem GPU e sem acesso
à Hugging Face Hub** (rede corporativa restrita). Fine-tuning real de uma
LLM exige as duas coisas. Por isso:

- O código em `train.py` e `evaluate.py` é completo e correto, mas os
  imports pesados (`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`)
  são carregados de forma tardia (lazy), então `finetuning.config` e
  `finetuning.dataset` (contrato de prompt + carregamento do dataset)
  puderam ser desenvolvidos e testados com `pytest` sem GPU.
- O treinamento real deve ser feito no **Google Colab** ou **Kaggle**
  (sugestão do próprio enunciado), seguindo os passos abaixo.

## Como rodar no Google Colab

1. Abra um notebook com GPU (Runtime > Change runtime type > GPU, T4 é
   suficiente para os modelos sugeridos).
2. Clone o repositório e instale as dependências:
   ```bash
   !git clone <url-do-repo> && cd FIAP-9IADT-medical-agent-fine-tuned
   !pip install -r requirements.txt
   ```
3. Gere o dataset (ou use o já versionado em `data/processed/`):
   ```bash
   !python -m preprocessing.run_pipeline
   ```
4. Treine:
   ```bash
   !python -m finetuning.train \
       --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --epochs 3 --batch-size 4 --lora-r 16
   ```
5. Avalie (métricas quantitativas + comparação antes/depois):
   ```bash
   !python -m finetuning.evaluate \
       --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --adapter-dir finetuning/adapters/medical-assistant-lora \
       --test-file data/processed/test.jsonl
   ```
6. Baixe a pasta `finetuning/adapters/medical-assistant-lora/` (contém os
   adapters LoRA + `hyperparameters.json`) e `finetuning/eval_results/`
   (métricas + comparação qualitativa) para usar no relatório técnico e
   trocar o modelo base pelo fine-tuned no assistente (Etapa 3).

Um notebook pronto com essas células está em
`finetuning/notebooks/colab_finetune.ipynb`.

## Modelo base

Padrão: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — recomendado no próprio
enunciado para times com recursos limitados (treina em GPU única, T4 do
Colab gratuito). Trocável via `--base-model` por `meta-llama/Meta-Llama-3-8B`,
`mistralai/Mistral-7B-v0.1`, `microsoft/Phi-3-mini-4k-instruct` etc.,
dependendo da GPU disponível.

## Hiperparâmetros (padrão, ajustáveis via CLI)

| Parâmetro | Padrão | Flag |
|---|---|---|
| Learning rate | 2e-4 | `--learning-rate` |
| Épocas | 3 | `--epochs` |
| Batch size (por device) | 4 | `--batch-size` |
| Gradient accumulation | 4 | `--grad-accum` |
| LoRA r | 16 | `--lora-r` |
| LoRA alpha | 32 | `--lora-alpha` |
| Quantização | 4-bit (QLoRA) | `--no-4bit` desativa |

Todos os hiperparâmetros efetivamente usados e as métricas de treino/
avaliação são salvos em `finetuning/adapters/<nome>/hyperparameters.json`
ao final do treino (item 7 do passo a passo).

## Contrato de prompt (`finetuning/config.py`)

`SYSTEM_PROMPT` e `format_prompt()` definem o formato exato usado tanto no
treino quanto na inferência — é o ponto de integração combinado entre a
Etapa 2 (fine-tuning) e a Etapa 3 (assistente). Qualquer mudança nesse
formato deve ser feita aqui e vale para as duas frentes.

## Smoke test (validação mecânica da pipeline, sem GPU/internet)

```bash
pip install torch transformers peft trl tokenizers
python -m finetuning.train --smoke-test
```

Isso treina um modelo GPT-2 minúsculo, **inicializado aleatoriamente e com
tokenizer treinado localmente** (nenhum download da internet), por uma
época em poucos exemplos, só para validar que a pipeline de tokenização +
LoRA + `SFTTrainer` + salvamento executa sem erros antes de gastar
tempo/GPU real no Colab. Não produz um modelo útil.

## Avaliação (item 7 e caixa "Avaliação do modelo" do enunciado)

`finetuning/evaluate.py` gera, para o mesmo conjunto de perguntas de
`data/processed/test.jsonl`:

- **Perplexidade** do modelo base vs. fine-tuned.
- **ROUGE-1 / ROUGE-L** das respostas geradas contra a resposta de
  referência (o texto do protocolo).
- **Comparação qualitativa lado a lado** (pergunta, referência, resposta
  base, resposta fine-tuned) para um subconjunto de exemplos.

Saída em `finetuning/eval_results/evaluation_report.md` e
`evaluation_results.json` — usados diretamente na seção de avaliação do
relatório técnico (`docs/relatorio_tecnico.md`).
