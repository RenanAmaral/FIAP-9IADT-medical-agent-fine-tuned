from assistant.evaluate_rag import evaluate_retrieval
from assistant.explainability import build_explanation
from assistant.rag import RetrievedChunk
from security.audit import read_audit_log


# --------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------


def test_retriever_finds_relevant_protocol(retriever):
    chunks = retriever.retrieve("escore CURB-65 para pneumonia", k=3)
    assert chunks
    assert chunks[0].protocol_id == "PROT-PNE-001"


def test_retrieved_text_excludes_indexing_header(retriever):
    """O cabeçalho (título/condições) é usado só para embedding; o texto
    exibido deve ser o corpo do protocolo."""
    chunk = retriever.retrieve("escore CURB-65 para pneumonia", k=1)[0]
    assert not chunk.trecho.startswith("Pneumonia Adquirida na Comunidade (PAC)")
    assert "Protocolo interno PROT-PNE-001" in chunk.trecho


def test_specialty_augments_instead_of_restricting(retriever):
    """Um quadro que atravessa especialidades deve recuperar o protocolo
    crítico mesmo que o paciente esteja catalogado em outra área."""
    chunks = retriever.retrieve(
        "paciente com suspeita de sepse pneumonia adquirida na comunidade",
        k=3,
        especialidade="pneumologia",
    )
    assert "PROT-INF-001" in {c.protocol_id for c in chunks}


def test_specialty_protocol_is_appended_when_missing(retriever):
    chunks = retriever.retrieve("escore CHA2DS2-VASc fibrilação atrial", k=3, especialidade="ortopedia")
    assert "PROT-ORT-001" in {c.protocol_id for c in chunks}


def test_retrieval_benchmark_meets_quality_bar(retriever):
    results = evaluate_retrieval(retriever, k=3)
    assert results["recall_at_3"] >= 0.9
    assert results["top1_accuracy"] >= 0.8


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------


def _chunk(protocol_id="PROT-CARD-001", score=0.5):
    return RetrievedChunk(
        protocol_id=protocol_id,
        titulo="Título",
        especialidade="cardiologia",
        trecho="conteúdo",
        score=score,
    )


def test_explanation_lists_protocol_and_patient_sources():
    exp = build_explanation([_chunk()], codigo_paciente="PAC-0001", tem_dados_paciente=True)
    tipos = {s.tipo for s in exp.sources}
    assert tipos == {"protocolo", "prontuario"}


def test_explanation_deduplicates_same_protocol():
    chunks = [_chunk(score=0.3), _chunk(score=0.6)]
    exp = build_explanation(chunks)
    protocolos = [s for s in exp.sources if s.tipo == "protocolo"]
    assert len(protocolos) == 1
    assert protocolos[0].score == 0.6


def test_confidence_higher_with_patient_data():
    sem = build_explanation([_chunk()], tem_dados_paciente=False)
    com = build_explanation([_chunk()], codigo_paciente="PAC-0001", tem_dados_paciente=True)
    assert com.confidence > sem.confidence


def test_pending_exams_reduce_confidence_and_add_note():
    sem = build_explanation([_chunk()], codigo_paciente="P", tem_dados_paciente=True)
    com = build_explanation(
        [_chunk()], codigo_paciente="P", tem_dados_paciente=True, exames_pendentes=["HbA1c"]
    )
    assert com.confidence < sem.confidence
    assert any("exames pendentes" in n for n in com.notes)


def test_no_chunks_gives_low_confidence_and_warning():
    exp = build_explanation([])
    assert exp.confidence < 0.4
    assert exp.confidence_label == "baixa"
    assert any("Nenhum protocolo" in n for n in exp.notes)


def test_explanation_block_mentions_sources_and_confidence():
    block = build_explanation([_chunk()], codigo_paciente="PAC-0001", tem_dados_paciente=True).to_block()
    assert "Fontes consultadas" in block
    assert "PROT-CARD-001" in block
    assert "Grau de confiança" in block


# --------------------------------------------------------------------------
# Chain completa
# --------------------------------------------------------------------------


def test_chain_answers_clinical_question_with_sources(assistant):
    response = assistant.run("Quais exames iniciais para hipertensão?", codigo_paciente="PAC-0001")
    assert not response.bloqueado
    assert response.explanation is not None
    assert response.explanation.sources
    assert "PROT-CARD-001" in response.texto_completo


def test_chain_always_includes_human_validation_notice(assistant):
    response = assistant.run("Qual a conduta para pneumonia?", codigo_paciente="PAC-0003")
    assert "VALIDAÇÃO HUMANA OBRIGATÓRIA" in response.resposta


def test_chain_blocks_out_of_scope_question(assistant):
    response = assistant.run("Qual a receita de bolo de chocolate?")
    assert response.bloqueado
    assert response.motivo_bloqueio == "out_of_scope"
    assert response.explanation is None


def test_chain_flags_pending_exams(assistant):
    response = assistant.run("Posso ajustar o tratamento?", codigo_paciente="PAC-0002")
    assert response.exames_pendentes
    assert any("exames pendentes" in n for n in response.explanation.notes)


def test_chain_works_without_patient_context(assistant):
    response = assistant.run("O que o protocolo diz sobre estratificar gravidade em pneumonia?")
    assert not response.bloqueado
    assert response.codigo_paciente is None
    assert response.explanation.sources


def test_chain_handles_unknown_patient_gracefully(assistant):
    response = assistant.run("Qual a conduta para hipertensão?", codigo_paciente="PAC-9999")
    assert not response.bloqueado
    assert response.exames_pendentes == []


# --------------------------------------------------------------------------
# Auditoria
# --------------------------------------------------------------------------


def test_interaction_is_logged_with_required_fields(assistant, log_path):
    assistant.run("Quais exames iniciais para hipertensão?", codigo_paciente="PAC-0001")
    entries = read_audit_log(log_path)

    assert len(entries) == 1
    entry = entries[0]
    for field in (
        "timestamp",
        "session_id",
        "pergunta",
        "contexto_recuperado",
        "resposta",
        "grafo_nos_executados",
        "bloqueios_seguranca",
    ):
        assert field in entry
    assert entry["codigo_paciente"] == "PAC-0001"
    assert entry["contexto_recuperado"]
    assert entry["confianca"] is not None


def test_blocked_interaction_is_logged_with_reason(assistant, log_path):
    assistant.run("Quem ganhou o jogo de futebol?")
    entries = read_audit_log(log_path)

    assert len(entries) == 1
    bloqueios = entries[0]["bloqueios_seguranca"]
    assert bloqueios
    assert bloqueios[0]["reason"] == "out_of_scope"


def test_each_interaction_appends_one_line(assistant, log_path):
    assistant.run("Qual a conduta para hipertensão?", codigo_paciente="PAC-0001")
    assistant.run("Qual a conduta para pneumonia?", codigo_paciente="PAC-0003")
    assert len(read_audit_log(log_path)) == 2


def test_session_id_is_preserved_when_provided(assistant, log_path):
    assistant.run("Qual a conduta para hipertensão?", session_id="sess-fixa-123")
    assert read_audit_log(log_path)[0]["session_id"] == "sess-fixa-123"


# --------------------------------------------------------------------------
# Comparação entre backends
# --------------------------------------------------------------------------


def test_comparison_gives_both_backends_identical_context(retriever, db_path, tmp_path):
    """O ponto da comparação: os backends precisam receber exatamente o mesmo
    contexto recuperado, senão a diferença observada não é atribuível ao
    modelo."""
    from assistant.compare_backends import compare

    comparisons = compare(
        backends=["template", "template"],
        cases=[("PAC-0003", "Qual a conduta?")],
        db_path=db_path,
        protocols_dir="data/protocols",
        log_path=tmp_path / "cmp.jsonl",
        base_model="irrelevante-para-o-stub",
        adapter_dir=None,
    )

    assert len(comparisons) == 1
    case = comparisons[0]
    assert case.contexto_protocolos
    assert len(case.respostas) == 2


def test_comparison_writes_markdown_with_both_answers(retriever, db_path, tmp_path):
    from assistant.compare_backends import compare, write_markdown

    comparisons = compare(
        backends=["template", "template"],
        cases=[(None, "Quais critérios do qSOFA indicam sepse?")],
        db_path=db_path,
        protocols_dir="data/protocols",
        log_path=tmp_path / "cmp.jsonl",
        base_model="x",
        adapter_dir=None,
    )
    out = tmp_path / "comparacao.md"
    write_markdown(comparisons, out)

    text = out.read_text(encoding="utf-8")
    assert "qSOFA" in text
    assert text.count("### `template`") == 2
    assert "mesmo contexto recuperado" in text
