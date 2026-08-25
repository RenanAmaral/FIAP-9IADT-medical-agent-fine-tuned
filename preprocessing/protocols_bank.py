"""Banco de protocolos clínicos internos sintéticos usados para gerar o dataset
de fine-tuning e para indexação no RAG do assistente (Etapa 3).

Todo o conteúdo aqui é fictício, criado para fins didáticos do Tech Challenge,
e não deve ser usado como orientação clínica real.
"""

PROTOCOLS = [
    {
        "id": "PROT-CARD-001",
        "especialidade": "cardiologia",
        "titulo": "Hipertensão Arterial Sistêmica (HAS) — Conduta Ambulatorial",
        "condicoes": ["hipertensão", "pressão alta", "HAS"],
        "texto": (
            "Protocolo interno PROT-CARD-001 — Hipertensão Arterial Sistêmica.\n\n"
            "1. Confirmação diagnóstica: PA >= 140/90 mmHg em pelo menos duas aferições "
            "em consultas distintas, ou MAPA/MRPA compatível.\n"
            "2. Exames iniciais obrigatórios: creatinina, potássio, glicemia de jejum, "
            "perfil lipídico, EAS e ECG de repouso.\n"
            "3. Estratificação de risco cardiovascular conforme escore de Framingham.\n"
            "4. Primeira linha medicamentosa: IECA/BRA ou bloqueador de canal de cálcio, "
            "conforme perfil do paciente e comorbidades.\n"
            "5. Reavaliação em 4 a 6 semanas após início ou ajuste de dose.\n"
            "6. Encaminhar para avaliação presencial imediata se PA >= 180/120 mmHg "
            "com sinais de lesão de órgão-alvo (emergência hipertensiva).\n"
            "7. Toda alteração de conduta medicamentosa depende de validação do médico "
            "responsável; o assistente não deve prescrever diretamente."
        ),
    },
    {
        "id": "PROT-CARD-002",
        "especialidade": "cardiologia",
        "titulo": "Fibrilação Atrial — Avaliação Inicial",
        "condicoes": ["fibrilação atrial", "arritmia", "FA"],
        "texto": (
            "Protocolo interno PROT-CARD-002 — Fibrilação Atrial.\n\n"
            "1. Confirmar diagnóstico com ECG de 12 derivações.\n"
            "2. Calcular escore CHA2DS2-VASc para risco tromboembólico e escore "
            "HAS-BLED para risco de sangramento antes de indicar anticoagulação.\n"
            "3. Avaliar necessidade de controle de frequência (betabloqueador ou "
            "bloqueador de canal de cálcio não di-hidropiridínico) versus controle de "
            "ritmo, conforme sintomas e tempo de instalação.\n"
            "4. Pacientes com CHA2DS2-VASc >= 2 (homens) ou >= 3 (mulheres) são "
            "candidatos a anticoagulação oral, a critério médico.\n"
            "5. Encaminhar para avaliação de cardiologista em até 7 dias em casos novos "
            "sem instabilidade hemodinâmica; encaminhamento IMEDIATO se houver "
            "instabilidade hemodinâmica.\n"
            "6. Este protocolo não substitui avaliação médica individualizada."
        ),
    },
    {
        "id": "PROT-END-001",
        "especialidade": "endocrinologia",
        "titulo": "Diabetes Mellitus Tipo 2 — Manejo Inicial",
        "condicoes": ["diabetes", "DM2", "glicemia alta"],
        "texto": (
            "Protocolo interno PROT-END-001 — Diabetes Mellitus Tipo 2.\n\n"
            "1. Diagnóstico: glicemia de jejum >= 126 mg/dL em duas ocasiões, ou "
            "HbA1c >= 6,5%, ou glicemia casual >= 200 mg/dL com sintomas clássicos.\n"
            "2. Exames complementares: função renal, perfil lipídico, "
            "microalbuminúria e fundoscopia anual.\n"
            "3. Primeira linha terapêutica: metformina associada a mudança de estilo "
            "de vida, salvo contraindicação (TFG < 30 mL/min/1,73m²).\n"
            "4. Meta geral de HbA1c < 7%, individualizada conforme idade, comorbidades "
            "e risco de hipoglicemia.\n"
            "5. Encaminhar para endocrinologia se HbA1c permanecer > 9% após 6 meses "
            "de tratamento otimizado, ou em caso de complicações micro/macrovasculares.\n"
            "6. Ajustes de esquema terapêutico exigem validação médica presencial."
        ),
    },
    {
        "id": "PROT-PNE-001",
        "especialidade": "pneumologia",
        "titulo": "Pneumonia Adquirida na Comunidade (PAC) — Adulto",
        "condicoes": ["pneumonia", "PAC", "infecção respiratória"],
        "texto": (
            "Protocolo interno PROT-PNE-001 — Pneumonia Adquirida na Comunidade.\n\n"
            "1. Confirmar com radiografia de tórax e quadro clínico compatível "
            "(febre, tosse, dispneia, dor pleurítica).\n"
            "2. Estratificar gravidade com escore CURB-65.\n"
            "3. CURB-65 0-1: tratamento ambulatorial com antibiótico oral conforme "
            "diretriz institucional de antimicrobianos.\n"
            "4. CURB-65 2: considerar observação hospitalar breve.\n"
            "5. CURB-65 >= 3: internação, considerar UTI se houver critérios de "
            "gravidade adicionais (sepse, insuficiência respiratória).\n"
            "6. Reavaliar resposta clínica em 48-72h; ausência de melhora exige "
            "reavaliação de diagnóstico e cobertura antimicrobiana.\n"
            "7. A escolha e prescrição do antimicrobiano específico é sempre "
            "responsabilidade do médico assistente."
        ),
    },
    {
        "id": "PROT-PNE-002",
        "especialidade": "pneumologia",
        "titulo": "DPOC — Exacerbação Aguda",
        "condicoes": ["DPOC", "exacerbação", "dispneia crônica"],
        "texto": (
            "Protocolo interno PROT-PNE-002 — Exacerbação Aguda de DPOC.\n\n"
            "1. Avaliar gravidade: dispneia, uso de musculatura acessória, "
            "saturação de O2, nível de consciência.\n"
            "2. Oxigenoterapia titulada para SpO2 alvo de 88-92%.\n"
            "3. Broncodilatadores de curta ação (beta-2 agonista +/- anticolinérgico) "
            "em doses repetidas.\n"
            "4. Considerar corticoide sistêmico por curto período e antibiótico se "
            "houver sinais de infecção bacteriana (aumento de purulência do escarro).\n"
            "5. Critérios de internação: falha de resposta ambulatorial, hipoxemia "
            "grave, confusão mental, comorbidades descompensadas.\n"
            "6. Sinais de alerta (uso importante de musculatura acessória, "
            "rebaixamento do nível de consciência, SpO2 persistentemente baixa) "
            "exigem acionamento imediato da equipe médica."
        ),
    },
    {
        "id": "PROT-NEU-001",
        "especialidade": "neurologia",
        "titulo": "Suspeita de Acidente Vascular Cerebral (AVC) Isquêmico",
        "condicoes": ["AVC", "derrame", "déficit neurológico agudo"],
        "texto": (
            "Protocolo interno PROT-NEU-001 — Suspeita de AVC Isquêmico Agudo.\n\n"
            "1. Aplicar escala de reconhecimento pré-hospitalar (ex.: FAST) diante de "
            "déficit neurológico agudo.\n"
            "2. Registrar horário exato do início dos sintomas (ou última vez "
            "assintomático) — determinante para elegibilidade de trombólise.\n"
            "3. Tomografia de crânio sem contraste em caráter de URGÊNCIA MÁXIMA.\n"
            "4. Avaliar elegibilidade para trombólise endovenosa (janela de até 4,5h) "
            "e/ou trombectomia mecânica conforme protocolo de AVC vigente.\n"
            "5. Este é um fluxo de tempo crítico: qualquer suspeita deve gerar alerta "
            "imediato à equipe médica e não deve aguardar interação assíncrona com o "
            "assistente virtual.\n"
            "6. O assistente pode apoiar a triagem informativa, mas nunca substitui o "
            "acionamento da linha de cuidado de AVC."
        ),
    },
    {
        "id": "PROT-INF-001",
        "especialidade": "infectologia",
        "titulo": "Sepse — Reconhecimento e Pacote Inicial (Bundle 1h)",
        "condicoes": ["sepse", "infecção grave", "choque séptico"],
        "texto": (
            "Protocolo interno PROT-INF-001 — Sepse e Choque Séptico.\n\n"
            "1. Rastreamento com critérios qSOFA (alteração do nível de consciência, "
            "FR >= 22irpm, PAS <= 100 mmHg) diante de suspeita de infecção.\n"
            "2. Pacote de 1 hora: coleta de lactato sérico, hemoculturas antes do "
            "antibiótico, antibioticoterapia de amplo espectro, reposição volêmica "
            "para hipotensão/lactato >= 4 mmol/L, e vasopressores se necessário para "
            "manter PAM >= 65 mmHg.\n"
            "3. Reavaliar lactato em 2-4 horas.\n"
            "4. Encaminhamento para UTI em casos de choque séptico refratário à "
            "volemia inicial.\n"
            "5. Este é um alerta de alta prioridade: o assistente deve sinalizar a "
            "equipe médica imediatamente diante de critérios compatíveis com sepse."
        ),
    },
    {
        "id": "PROT-PED-001",
        "especialidade": "pediatria",
        "titulo": "Febre sem Sinais de Localização em Lactentes",
        "condicoes": ["febre", "lactente", "criança pequena"],
        "texto": (
            "Protocolo interno PROT-PED-001 — Febre sem Sinais de Localização.\n\n"
            "1. Lactentes menores de 3 meses com febre (Tax >= 38°C) são considerados "
            "de risco e devem ser avaliados presencialmente com prioridade.\n"
            "2. Entre 3 e 36 meses, aplicar critérios de risco (aspecto tóxico, "
            "irritabilidade, hipoatividade) para definir necessidade de exames "
            "complementares (hemograma, PCR, EAS, urocultura).\n"
            "3. Orientar sinais de alarme para retorno imediato: recusa alimentar, "
            "letargia, exantema petequial, dificuldade respiratória.\n"
            "4. Antitérmicos apenas para conforto, sem prescrição automática pelo "
            "assistente — orientação sempre condicionada à validação médica.\n"
            "5. Encaminhar para avaliação presencial urgente em caso de aspecto "
            "tóxico ou sinais de alarme."
        ),
    },
    {
        "id": "PROT-GIN-001",
        "especialidade": "ginecologia",
        "titulo": "Pré-natal de Baixo Risco — Rotina de Exames",
        "condicoes": ["pré-natal", "gestação", "gravidez"],
        "texto": (
            "Protocolo interno PROT-GIN-001 — Pré-natal de Baixo Risco.\n\n"
            "1. Primeira consulta: hemograma, tipagem sanguínea, glicemia de jejum, "
            "sorologias (HIV, sífilis, hepatite B e C, toxoplasmose), EAS e "
            "urocultura.\n"
            "2. Ultrassonografia obstétrica para datação idealmente no primeiro "
            "trimestre.\n"
            "3. Rastreamento de diabetes gestacional entre 24 e 28 semanas (TOTG).\n"
            "4. Sinais de alarme que exigem avaliação imediata: sangramento vaginal, "
            "cefaleia intensa, edema súbito, diminuição de movimentos fetais.\n"
            "5. Classificação de risco deve ser revista a cada consulta; mudança "
            "para alto risco implica encaminhamento ao pré-natal especializado.\n"
            "6. Este protocolo cobre apenas gestações classificadas como baixo risco."
        ),
    },
    {
        "id": "PROT-ORT-001",
        "especialidade": "ortopedia",
        "titulo": "Lombalgia Aguda Inespecífica",
        "condicoes": ["lombalgia", "dor lombar", "dor nas costas"],
        "texto": (
            "Protocolo interno PROT-ORT-001 — Lombalgia Aguda Inespecífica.\n\n"
            "1. Investigar sinais de alarme (red flags): trauma significativo, "
            "perda de peso inexplicada, febre, déficit neurológico progressivo, "
            "história de câncer, uso de corticoide crônico, idade > 70 anos.\n"
            "2. Na ausência de sinais de alarme, exames de imagem NÃO são "
            "recomendados rotineiramente nas primeiras 4-6 semanas.\n"
            "3. Orientar manutenção de atividade conforme tolerado, analgesia "
            "escalonada e evitar repouso prolongado no leito.\n"
            "4. Reavaliar em 4-6 semanas se não houver melhora significativa.\n"
            "5. Síndrome da cauda equina (retenção urinária, anestesia em sela, "
            "fraqueza progressiva) é emergência cirúrgica — acionar equipe "
            "imediatamente."
        ),
    },
]


def get_protocol_by_condition(keyword: str):
    keyword = keyword.lower()
    for protocol in PROTOCOLS:
        if any(keyword in c.lower() for c in protocol["condicoes"]):
            return protocol
    return None
