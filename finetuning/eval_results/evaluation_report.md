# Avaliação do Modelo Fine-tuned

Modelo base: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
Adapter: `/content/drive/MyDrive/tech-challenge-fase3/medical-assistant-lora`  
Exemplos de teste avaliados: 8

## Métricas quantitativas

| Métrica | Base | Fine-tuned |
|---|---|---|
| Perplexidade | 10.73 | 1.04 |
| ROUGE-1 | 0.225 | 0.764 |
| ROUGE-L | 0.128 | 0.744 |

## Comparação qualitativa (amostra)

### Exemplo 1
**Pergunta:** Como o protocolo interno recomenda estratificar o risco em um caso de DM2?

**Referência (protocolo):** Com base no protocolo interno PROT-END-001 (Diabetes Mellitus Tipo 2 — Manejo Inicial):
Protocolo interno PROT-END-001 — Diabetes Mellitus Tipo 2.

1. Diagnóstico: glicemia de jejum >= 126 mg/dL em duas ocasiões, ou HbA1c >= 6,5%, ou glicemia casual >= 200 mg/dL com sintomas clássicos.
2. Exames com...

**Resposta do modelo base:** O protocolo interno recomenda estratificar o risco em um caso de DM2 em duas categorias: risco de exacerbação e risco de remissão. A primeira categoria inclui riscos de exacerbação, como a doença de Alzheimer, a doença de Parkinson, a doença de Huntington e a doença de Friedreich. A segunda categori

**Resposta do modelo fine-tuned:** Com base no protocolo interno PROT-END-001 (Diabetes Mellitus Tipo 2 — Manejo Inicial):
Protocolo interno PROT-END-001 — Diabetes Mellitus Tipo 2.

1. Diagnóstico: glicemia de jejum >= 126 mg/dL em duas ocasiões, ou HbA1c >= 9%, ou glicemia casual >= 200 mg/dL com sintomas clássicos.
2. Exames compl

### Exemplo 2
**Pergunta:** Quais exames devo solicitar antes de definir a conduta para um caso de arritmia?

**Referência (protocolo):** Com base no protocolo interno PROT-CARD-002 (Fibrilação Atrial — Avaliação Inicial):
Protocolo interno PROT-CARD-002 — Fibrilação Atrial.

1. Confirmar diagnóstico com ECG de 12 derivações.
2. Calcular escore CHA2DS2-VASc para risco tromboembólico e escore HAS-BLED para risco de sangramento antes de...

**Resposta do modelo base:** As exames que você deve solicitar antes de definir a conduta para um caso de arritmia são:
1. ECG (Electrocardiograma)
2. Echocardiograma
3. Teste de hemoglobina (HbA1c)
4. Teste de glucose (glucose)
5. Teste de bilirrubina (bilirrubina)
6. Teste de creatinina (creatinina)
7. Teste de urina (urina)


**Resposta do modelo fine-tuned:** Com base no protocolo interno PROT-CARD-001 (Fibrilação Atrial — Avaliação Inicial):
Protocolo interno PROT-CARD-001 — Fibrilação Atrial.

1. Confirmar diagnóstico com ECG de 12 derivações.
2. Calcular escore de risco cardiovascular (Framingham >= 120/70) para indicar necessidade de exames complemen

### Exemplo 3
**Pergunta:** Quais exames devo solicitar antes de definir a conduta para um caso de sepse?

**Referência (protocolo):** Com base no protocolo interno PROT-INF-001 (Sepse — Reconhecimento e Pacote Inicial (Bundle 1h)):
Protocolo interno PROT-INF-001 — Sepse e Choque Séptico.

1. Rastreamento com critérios qSOFA (alteração do nível de consciência, FR >= 22irpm, PAS <= 100 mmHg) diante de suspeita de infecção.
2. Pacote...

**Resposta do modelo base:** As exames que você deve solicitar antes de definir a conduta para um caso de sepse são:
1. Diagnóstico de sepse
2. Diagnóstico de insuficiência cardíaca
3. Diagnóstico de insuficiência respiratória
4. Diagnóstico de insuficiência renal
5. Diagnóstico de insuficiência hepática
6. Diagnóstico de insuf

**Resposta do modelo fine-tuned:** Com base no protocolo interno PROT-INF-001 (Sepse — Reconhecimento e Pacote Inicial (Bundle 1h)):
Protocolo interno PROT-INF-001 — Sepse e Choque Séptico.

1. Rastreamento com critérios qSOFA (alteração do nível de consciência, FR >= 22irpm, PAS <= 400 mmHg) diante de suspeita de infecção.
2. Pacote

### Exemplo 4
**Pergunta:** Paciente com infecção respiratória, qual a conduta recomendada segundo o protocolo interno?

**Referência (protocolo):** Com base no protocolo interno PROT-PNE-001 (Pneumonia Adquirida na Comunidade (PAC) — Adulto):
Protocolo interno PROT-PNE-001 — Pneumonia Adquirida na Comunidade.

1. Confirmar com radiografia de tórax e quadro clínico compatível (febre, tosse, dispneia, dor pleurítica).
2. Estratificar gravidade co...

**Resposta do modelo base:** Se o paciente apresentar uma infecção respiratória, a conduta recomendada segundo o protocolo interno é:
1. Acompanhar o paciente em sua residência médica, se possível, para que possa ser monitorizado e atendido por um médico.
2. Aplicar o protocolo de tratamento de infecção respiratória, conforme i

**Resposta do modelo fine-tuned:** Com base no protocolo interno PROT-PNE-001 (Pneumonia Adquirida na Comunidade (PAC) — Adulto):
Protocolo interno PROT-PNE-001 — Pneumonia Adquirida na Comunidade.

1. Confirmar com radiografia de tórax e quadro clínico compatível (febre, tosse, dispneia, dor pleurítica).
2. Estratificar gravidade co

### Exemplo 5
**Pergunta:** Paciente com gestação, qual a conduta recomendada segundo o protocolo interno?

**Referência (protocolo):** Com base no protocolo interno PROT-GIN-001 (Pré-natal de Baixo Risco — Rotina de Exames):
Protocolo interno PROT-GIN-001 — Pré-natal de Baixo Risco.

1. Primeira consulta: hemograma, tipagem sanguínea, glicemia de jejum, sorologias (HIV, sífilis, hepatite B e C, toxoplasmose), EAS e urocultura.
2. U...

**Resposta do modelo base:** O protocolo interno do hospital recomenda que o assistente virtual médico de apoio a profissionais de saúde de um hospital, seja chamado para uma conduta clínica com um paciente com gestação, que tenha sido indicada por um médico.

A conduta clínica deve ser realizada com base nos protocolos interno

**Resposta do modelo fine-tuned:** Com base no protocolo interno PROT-GIN-001 (Pré-natal de Baixo Risco — Rotina de Exames):
Protocolo interno PROT-GIN-001 — Pré-natal de Baixo Risco.

1. Primeira consulta: hemograma, tipagem sanguínea, glicemia de jejum, sorologias (HIV, sífilis, hepatite B e C, toxoplasmose), EAS e urocultura.
2. U
