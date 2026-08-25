import pytest

from assistant.chains import MedicalAssistantChain
from assistant.database import build_database
from assistant.llm import TemplateClinicalLLM
from assistant.rag import ProtocolRetriever


@pytest.fixture(scope="session")
def retriever() -> ProtocolRetriever:
    """Índice FAISS dos protocolos. Escopo de sessão: construir o índice é a
    parte cara do setup e ele é somente-leitura nos testes.
    """
    return ProtocolRetriever.from_protocols("data/protocols")


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "hospital_test.db"
    build_database(path, seed=42)
    return path


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "audit_test.jsonl"


@pytest.fixture
def assistant(retriever, db_path, log_path) -> MedicalAssistantChain:
    return MedicalAssistantChain(
        llm=TemplateClinicalLLM(),
        retriever=retriever,
        db_path=db_path,
        log_path=log_path,
        llm_backend_name="template",
    )
