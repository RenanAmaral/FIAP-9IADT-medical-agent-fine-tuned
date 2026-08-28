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

## Token da Hugging Face

Sem um token, os downloads da Hub são anônimos e compartilham o limite de taxa
do IP — no Colab, isso significa dividir a cota com todos os outros notebooks
que saem pelo mesmo IP. É a origem do aviso:

```
Warning: You are sending unauthenticated requests to the HF Hub.
```

O download de modelos públicos ainda funciona, mas fica mais lento e pode
falhar com HTTP 429 no meio de um arquivo de vários GB. Para modelos de
licença restrita (Llama 3, Gemma), o token é **obrigatório**.

Crie um token de leitura em https://huggingface.co/settings/tokens e:

- **No Colab:** abra **Secrets** (🔑), adicione `HF_TOKEN` e ative
  *Notebook access*.
- **Localmente:** `export HF_TOKEN=hf_xxx`

`finetuning/hf_auth.py` detecta o token automaticamente (variável de
ambiente, secrets do Colab ou login já persistido) e é chamado por `train.py`,
`evaluate.py` e `assistant/llm.py` antes de qualquer download. Sem token, o
fluxo continua — apenas com o aviso e sujeito ao limite de taxa.

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
   !python -m finetuning.train --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
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
| Épocas | 12 | `--epochs` |
| Batch size (por device) | 4 | `--batch-size` |
| Gradient accumulation | 1 | `--grad-accum` |
| LoRA r | 16 | `--lora-r` |
| LoRA alpha | 32 | `--lora-alpha` |
| Quantização | 4-bit (QLoRA) | `--no-4bit` desativa |

### Por que 12 épocas e não 3

O dataset tem ~59 exemplos de treino. Com o padrão inicial (3 épocas, batch
efetivo 16) o treino rendia apenas **~9 atualizações de peso** — os adapters
LoRA mal saíam da inicialização e o modelo treinado ficaria indistinguível do
base na avaliação, esvaziando a comparação antes/depois que o relatório
precisa mostrar.

Com batch efetivo 4 e 12 épocas são **~168 atualizações**, e o treino ainda
leva poucos minutos numa T4. O script imprime o plano antes de começar e
avisa se o número de passos cair abaixo do útil:

```
[plano] 59 exemplos | batch efetivo 4 | 14 passos/época × 12 épocas = ~168 atualizações de peso
```

Todos os hiperparâmetros efetivamente usados e as métricas de treino/
avaliação são salvos em `finetuning/adapters/<nome>/hyperparameters.json`
ao final do treino (item 7 do passo a passo).

## Compatibilidade entre versões de `trl` / `transformers`

O treino roda no Colab/Kaggle, onde as versões das bibliotecas mudam sem aviso
e sem que possamos fixá-las. Alguns parâmetros foram renomeados entre versões,
e passar o nome errado derruba o script **depois** de já ter baixado
gigabytes de pesos:

| Nome canônico no projeto | Nome alternativo | Onde mudou |
|---|---|---|
| `max_length` | `max_seq_length` | `SFTConfig`, trl < 0.20 |
| `warmup_ratio` | `warmup_steps` (aceita float como proporção) | transformers 5.x |
| `eval_strategy` | `evaluation_strategy` | transformers < 4.41 |
| `dtype` | `torch_dtype` | transformers 5.0 |

`resolve_config_kwargs` (em `train.py`) e `dtype_kwarg` (em `hf_auth.py`)
resolvem esses nomes **em runtime**, inspecionando o que a versão instalada
aceita. Um parâmetro sem correspondência é descartado com aviso em vez de
interromper o treino — perder um hiperparâmetro secundário é preferível a
perder a execução inteira.

O caso do `dtype` merece atenção especial: como `from_pretrained` recebe
`**kwargs`, passar o nome errado **não levanta erro** — em transformers 4.x um
`dtype=` seria simplesmente ignorado e o modelo carregaria em fp32, o dobro da
memória e cerca de metade da velocidade, sem nenhum sinal de que algo deu
errado. Por isso a escolha é feita pelo número de versão, não por tentativa.

`tests/test_finetuning_compat.py` cobre as duas convenções de API.

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
