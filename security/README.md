# security/

Implementa a Etapa 5 do Tech Challenge: segurança, validação e explainability.

| Módulo | Responsabilidade | Item |
|---|---|---|
| `guardrails.py` | Limites de atuação (entrada e saída). | 12 |
| `audit.py` | Logging estruturado para auditoria. | 13 |
| `inspect_logs.py` | Leitura e resumo do log de auditoria. | 13 |

A explainability (item 14) vive em `assistant/explainability.py`, junto do
código que a produz.

## Guardrails (item 12)

Há **duas camadas**, e isso é deliberado:

1. **Prompt de sistema** (`finetuning/config.py`) — instrui o modelo a não
   prescrever, exigir validação humana e recusar temas fora de escopo.
2. **Validação programática** (`guardrails.py`) — determinística,
   independente do modelo, roda sempre.

A segunda camada existe porque instrução em prompt não é um controle de
segurança confiável: o modelo pode ignorá-la, e o comportamento muda a cada
fine-tuning. Em um contexto clínico, a garantia de que o assistente nunca
prescreve não pode depender de o modelo ter "obedecido".

### Controles

| Função | Momento | O que faz |
|---|---|---|
| `check_input_scope` | pré-LLM | Recusa temas fora de escopo, pedidos de prescrição direta e tentativas de prompt injection. |
| `check_output_safety` | pós-LLM | Bloqueia prescrição direta na resposta gerada. |
| `enforce_human_validation` | pós-LLM | Garante o aviso de validação humana em toda resposta liberada. |

### O que conta como prescrição direta

`check_output_safety` só bloqueia quando há **posologia numérica combinada
com verbo imperativo de administração** ("Administre 500 mg de..."). Citar o
que o protocolo recomenda ("a primeira linha terapêutica é metformina, a
critério médico") ou mencionar um valor de referência diagnóstico ("glicemia
≥ 126 mg/dL") é legítimo e não é bloqueado — um guardrail que barrasse isso
tornaria o assistente inútil para o profissional. Ambos os casos estão
cobertos por testes.

## Logging de auditoria (item 13)

Formato JSON Lines em `logs/audit.jsonl` — uma interação por linha,
diretamente consultável com `jq` ou pandas, sem parsing de texto livre.

Campos registrados, conforme exigido no enunciado:

| Campo | Conteúdo |
|---|---|
| `timestamp` | Momento da interação (UTC, ISO 8601). |
| `session_id` | Identificador da sessão. |
| `pergunta` | Pergunta recebida. |
| `contexto_recuperado` | Trechos de protocolo recuperados, com score. |
| `contexto_paciente` | Dados do prontuário usados como contexto. |
| `resposta` | Resposta gerada. |
| `fontes` / `confianca` | Explainability da resposta. |
| `grafo_nos_executados` | Nós do LangGraph percorridos. |
| `bloqueios_seguranca` | Guardrails acionados, com motivo e padrão casado. |
| `alerta_emitido` | Se houve acionamento da equipe médica. |
| `requer_validacao_humana` | Sempre `true`. |
| `llm_backend` | Qual backend de LLM respondeu. |
| `duracao_ms` / `erro` | Desempenho e falhas. |

### LangSmith (opcional)

Se `LANGCHAIN_TRACING_V2=true` e `LANGCHAIN_API_KEY` estiverem definidas, o
LangChain envia os traces automaticamente, sem configuração adicional neste
código. O log local continua sendo a fonte de auditoria própria do hospital —
rastreabilidade não deve depender de um serviço externo.

## Inspecionar os logs

```bash
python -m security.inspect_logs              # resumo geral
python -m security.inspect_logs --bloqueios  # interações bloqueadas
python -m security.inspect_logs --alertas    # fluxos com alerta de risco
python -m security.inspect_logs --sessao sess-abc123
python -m security.inspect_logs --detalhe -n 3
```

## Explainability (item 14)

Toda resposta liberada carrega um bloco com:

- **Fontes consultadas** — protocolo (identificador + título + relevância) e
  registro de prontuário que embasaram a conduta.
- **Grau de confiança** — score de *recuperação*, não de correção clínica.
  Deriva da similaridade dos trechos recuperados, da presença de dados do
  paciente e de uma penalidade quando há exames pendentes.

A distinção entre "confiança na recuperação" e "correção clínica" é
explicitada junto ao número na própria resposta, para não induzir o
profissional a ler o score como um aval clínico.
