# assistant/

Implementa a Etapa 3 do Tech Challenge: o assistente médico com LangChain.

| Módulo | Responsabilidade | Item do enunciado |
|---|---|---|
| `llm.py` | LLM plugável: fine-tuned, base ou stub offline. | 8 |
| `database.py` | Base estruturada SQLite (prontuários, exames, evoluções). | 9 |
| `rag.py` | Vector store FAISS sobre os protocolos internos. | 10 |
| `prompts.py` | Templates de prompt (formato de treino da Etapa 2). | 8 |
| `explainability.py` | Fontes citadas e grau de confiança. | 14 |
| `chains.py` | Chain principal que orquestra tudo. | 3, 4, 5 |
| `cli.py` | Interface de linha de comando. | — |
| `evaluate_rag.py` | Benchmark de recuperação (top-1 / recall@k). | — |
| `compare_backends.py` | Compara o assistente completo com modelo base vs. fine-tuned. | — |

## Uso

```bash
# 1. Criar a base estruturada (uma vez)
python -m assistant.database

# 2. Consultar
python -m assistant.cli --listar-pacientes
python -m assistant.cli --paciente PAC-0002 --pergunta "Posso ajustar o tratamento?"
python -m assistant.cli --interativo --paciente PAC-0001
```

Para o fluxo clínico completo com o grafo de decisão, use `python -m graphs.cli`.

## Backends de LLM (`--backend`)

- `template` (padrão) — **não é um modelo de linguagem**. Stub determinístico
  que remonta a resposta a partir do contexto já recuperado. Existe porque o
  ambiente de desenvolvimento deste projeto não tem GPU nem acesso à Hugging
  Face Hub; permite testar toda a orquestração (RAG, base estruturada, grafo,
  guardrails, logging) sem o modelo. É o backend usado nos testes
  automatizados e **não deve ser usado na demonstração final**.
- `base` — modelo base sem fine-tuning, para comparação antes/depois.
- `finetuned` — modelo base + adapters LoRA da Etapa 2. É o backend de
  produção; requer os adapters treinados (ver `finetuning/README.md`).

Trocar o backend é a única mudança necessária para passar do desenvolvimento
offline para o modelo treinado — a chain e o grafo não mudam.

## Decisões de design

**Tools com SQL parametrizado, não SQL Agent.** O enunciado sugere "SQL
Chain/Agent ou tools customizadas". Optamos por funções tipadas com SQL
parametrizado (`get_patient_record`, `get_pending_exams`): deixar o LLM
escrever SQL livre sobre uma base de prontuários é uma superfície de risco
desnecessária (injeção, leitura de registros de outros pacientes), e o
conjunto de consultas que o fluxo precisa é pequeno e bem definido.

**A especialidade acrescenta contexto, não restringe.** O RAG faz busca
global e só anexa o protocolo da especialidade do paciente se ele não veio
naturalmente. Filtrar rigidamente pela especialidade do prontuário parecia
natural, mas escondia protocolos críticos: um paciente de pneumologia que
evolui com sepse precisa do protocolo de infectologia. Em triagem clínica,
não recuperar um protocolo crítico é muito pior que recuperar um a mais.

**Chunks indexados com cabeçalho de contexto.** Cada trecho é embeddado
junto de título, condições e especialidade do protocolo — sem isso, um chunk
que menciona só "escore CURB-65" não casa com uma pergunta sobre "pneumonia".
O texto exibido ao usuário continua sendo apenas o corpo do protocolo.

**Embeddings TF-IDF por padrão.** Determinísticos, sem GPU e sem download de
modelo — o RAG roda e é testável offline. `load_huggingface_embeddings()`
oferece embeddings densos para produção (requer rede na primeira execução).

## Comparando modelo base e fine-tuned no assistente completo

```bash
python -m assistant.compare_backends \
    --paciente PAC-0003 \
    --pergunta "Qual a conduta para este paciente?" \
    --adapter-dir finetuning/adapters/medical-assistant-lora
```

Sem `--pergunta`, roda um conjunto padrão de casos que cobre os três caminhos
do grafo e as perguntas de limiar numérico (HbA1c, qSOFA), onde a alucionação
do modelo aparece.

Isso é diferente de `finetuning/evaluate.py`, que compara os modelos **crus**,
recebendo só a pergunta. Aqui a comparação passa por toda a pilha — RAG,
prontuário, guardrails, explainability — e os dois backends recebem
**exatamente o mesmo contexto recuperado** (um único `ProtocolRetriever` é
compartilhado, e o script avisa se o contexto divergir). A diferença observada
é só o que cada modelo faz com esse contexto.

É a pergunta que de fato importa aqui: **com o texto do protocolo entregue no
contexto, o modelo base já resolve?** Se sim, o fine-tuning agrega pouco; se
não, ele se justifica. A resposta vai para `docs/comparacao_backends.md`, em
formato pronto para o relatório.

Os modelos são carregados em sequência e liberados após o uso, então o par não
precisa caber na memória ao mesmo tempo.

## Qualidade da recuperação

```bash
python -m assistant.evaluate_rag
```

Benchmark de 15 perguntas clínicas com o protocolo correto anotado
manualmente. Resultado atual: **93,3% de acurácia top-1** e **100% de
recall@3**. O teste `test_retrieval_benchmark_meets_quality_bar` trava esses
patamares para que uma regressão na recuperação quebre o build.
