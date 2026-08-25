"""Etapa 3, item 10 — Contextualização com RAG sobre os protocolos internos.

Indexa os protocolos internos (`data/protocols/*.md`, gerados na Etapa 1) em
um vector store FAISS e recupera os trechos relevantes para compor o contexto
da resposta.

Sobre a escolha do modelo de embeddings
---------------------------------------
Há duas implementações de embeddings aqui:

- `TfidfEmbeddings` (padrão): TF-IDF via scikit-learn, ajustado sobre o
  próprio corpus de protocolos. É determinístico, não exige GPU e — o ponto
  decisivo neste projeto — **não depende de download de modelo**, então o
  RAG roda e é testável no ambiente de desenvolvimento sem acesso à
  Hugging Face Hub. Para um corpus pequeno e de vocabulário técnico
  bastante específico (nomes de protocolo, siglas clínicas), TF-IDF já
  recupera bem.
- `load_huggingface_embeddings()`: embeddings densos
  (`sentence-transformers/all-MiniLM-L6-v2` por padrão), recomendados em
  produção por capturarem similaridade semântica além da lexical. Exige
  rede na primeira execução para baixar o modelo.

A chain aceita qualquer um dos dois via injeção, então trocar é uma linha.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_PROTOCOLS_DIR = Path("data/protocols")
DEFAULT_INDEX_DIR = Path("assistant/vector_store")

#: Stopwords em português. O scikit-learn só embute lista de stopwords para
#: inglês; sem elas, termos como "de", "para" e "que" dominam os vetores TF-IDF
#: e achatam a discriminação entre protocolos (medido: top-1 sobe de 73% para
#: 93% no benchmark de `assistant/evaluate_rag.py`).
#: Escrita sem acentos porque o vetorizador usa `strip_accents="unicode"`: os
#: tokens chegam à filtragem já normalizados, então stopwords acentuadas
#: nunca casariam.
PT_STOPWORDS = [
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "deve", "devem",
    "do", "dos", "e", "em", "essa", "esse", "esta", "este", "eu", "foi", "ha",
    "isso", "ja", "mais", "mas", "me", "na", "nas", "no", "nos", "nao", "num",
    "numa", "o", "os", "ou", "para", "pela", "pelo", "por", "qual", "quando",
    "que", "se", "sem", "ser", "seu", "sua", "sao", "tambem", "um", "uma",
]


class TfidfEmbeddings(Embeddings):
    """Embeddings TF-IDF ajustados sobre o corpus de protocolos.

    Implementa a interface `Embeddings` do LangChain, então funciona com
    qualquer vector store da biblioteca.
    """

    def __init__(self, corpus: list[str], max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            stop_words=PT_STOPWORDS,
        )
        self._vectorizer.fit(corpus)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._vectorizer.transform(texts).toarray().astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vectorizer.transform([text]).toarray().astype("float32")[0].tolist()


def load_huggingface_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Embeddings densos via sentence-transformers (requer rede na primeira
    execução para baixar o modelo). Import tardio para não tornar
    `sentence-transformers` uma dependência obrigatória do módulo.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


@dataclass
class RetrievedChunk:
    """Trecho recuperado, já com os metadados necessários para a
    explainability (Etapa 5, item 14).
    """

    protocol_id: str
    titulo: str
    especialidade: str
    trecho: str
    score: float

    def to_context_string(self) -> str:
        return f"[{self.protocol_id} — {self.titulo}]\n{self.trecho}"


def _parse_protocol_file(path: Path) -> tuple[str, str]:
    """Retorna (titulo, corpo) de um arquivo de protocolo em Markdown."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    titulo = lines[0].lstrip("# ").strip() if lines else path.stem
    corpo = "\n".join(lines[1:]).strip()
    return titulo, corpo


def _split_into_chunks(body: str, max_chars: int = 700) -> list[str]:
    """Split por itens numerados do protocolo, agrupando até `max_chars`.

    Os protocolos são escritos como listas numeradas de condutas, então
    quebrar nos itens preserva unidades semânticas completas — muito melhor
    que um split cego por número de caracteres, que cortaria uma conduta ao
    meio e produziria contexto truncado no prompt.
    """
    parts = re.split(r"\n(?=\d+\.\s)", body)
    chunks: list[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if current and len(current) + len(part) + 1 > max_chars:
            chunks.append(current.strip())
            current = part
        else:
            current = f"{current}\n{part}" if current else part

    if current.strip():
        chunks.append(current.strip())
    return chunks


def load_protocol_documents(protocols_dir: str | Path = DEFAULT_PROTOCOLS_DIR) -> list[Document]:
    """Carrega os protocolos internos como `Document`s do LangChain, já
    fatiados em chunks e com metadados de proveniência.
    """
    protocols_dir = Path(protocols_dir)
    if not protocols_dir.exists():
        raise FileNotFoundError(
            f"Diretório de protocolos não encontrado: {protocols_dir}. "
            "Rode `python -m preprocessing.run_pipeline` primeiro."
        )

    # Metadados vêm do banco de protocolos da Etapa 1, mantendo uma única
    # fonte de verdade.
    from preprocessing.protocols_bank import PROTOCOLS

    meta_by_id = {p["id"]: p for p in PROTOCOLS}

    documents: list[Document] = []
    for path in sorted(protocols_dir.glob("*.md")):
        protocol_id = path.stem
        titulo, corpo = _parse_protocol_file(path)
        protocol_meta = meta_by_id.get(protocol_id, {})
        especialidade = protocol_meta.get("especialidade", "geral")
        condicoes = protocol_meta.get("condicoes", [])

        # Cada chunk é indexado com um cabeçalho contendo título, condições e
        # especialidade. Sem isso, um chunk que fala só de "escore CURB-65"
        # não casa com uma pergunta que diz "pneumonia", porque a palavra não
        # aparece naquele trecho — o cabeçalho devolve ao chunk o contexto que
        # o split removeu. O texto exibido ao usuário continua sendo apenas o
        # corpo (`trecho_original`), sem o cabeçalho.
        header = (
            f"{titulo}. Condições: {', '.join(condicoes)}. "
            f"Especialidade: {especialidade}."
        )

        for i, chunk in enumerate(_split_into_chunks(corpo)):
            documents.append(
                Document(
                    page_content=f"{header}\n{chunk}",
                    metadata={
                        "protocol_id": protocol_id,
                        "titulo": titulo,
                        "especialidade": especialidade,
                        "chunk_index": i,
                        "source_path": str(path),
                        "trecho_original": chunk,
                    },
                )
            )
    return documents


class ProtocolRetriever:
    """Wrapper fino sobre o FAISS que devolve `RetrievedChunk`s prontos para
    o prompt e para o bloco de fontes da resposta.
    """

    def __init__(self, vector_store, documents: list[Document]):
        self._vector_store = vector_store
        self.documents = documents

    @classmethod
    def from_protocols(
        cls,
        protocols_dir: str | Path = DEFAULT_PROTOCOLS_DIR,
        embeddings: Embeddings | None = None,
    ) -> "ProtocolRetriever":
        from langchain_community.vectorstores import FAISS

        documents = load_protocol_documents(protocols_dir)
        if embeddings is None:
            embeddings = TfidfEmbeddings([d.page_content for d in documents])

        vector_store = FAISS.from_documents(documents, embeddings)
        return cls(vector_store, documents)

    @staticmethod
    def _to_chunk(doc: Document, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            protocol_id=doc.metadata["protocol_id"],
            titulo=doc.metadata["titulo"],
            especialidade=doc.metadata["especialidade"],
            # Exibe o corpo do protocolo, sem o cabeçalho auxiliar de indexação.
            trecho=doc.metadata.get("trecho_original", doc.page_content),
            # FAISS devolve distância L2 (menor = mais similar); convertemos
            # para uma similaridade em (0, 1] para uso no grau de confiança.
            score=float(1.0 / (1.0 + score)),
        )

    def retrieve(
        self, query: str, k: int = 3, especialidade: str | None = None
    ) -> list[RetrievedChunk]:
        """Busca os `k` trechos mais relevantes para a consulta.

        A `especialidade` do paciente **acrescenta** contexto, não restringe:
        a busca principal é global e, só se nenhum dos resultados for da
        especialidade do paciente, o melhor trecho dessa especialidade é
        anexado.

        Filtrar rigidamente pela especialidade do prontuário seria a escolha
        óbvia, mas é perigosa aqui: quadros graves atravessam especialidades.
        Um paciente catalogado em pneumologia que evolui com sepse precisa do
        protocolo de infectologia — sob filtro rígido esse protocolo nunca
        seria recuperado. Em triagem clínica, deixar de trazer um protocolo
        crítico é um erro muito mais caro que trazer um protocolo a mais.
        """
        results = self._vector_store.similarity_search_with_score(query, k=k)
        chunks = [self._to_chunk(doc, score) for doc, score in results]

        if especialidade and not any(c.especialidade == especialidade for c in chunks):
            extra = self._vector_store.similarity_search_with_score(
                query, k=1, filter={"especialidade": especialidade}
            )
            chunks.extend(self._to_chunk(doc, score) for doc, score in extra)

        # Deduplica trechos repetidos preservando a ordem de relevância.
        seen: set[tuple[str, str]] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            key = (chunk.protocol_id, chunk.trecho)
            if key in seen:
                continue
            seen.add(key)
            unique.append(chunk)
        return unique

    def save(self, index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self._vector_store.save_local(str(index_dir))


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "Nenhum protocolo interno relevante foi encontrado para esta consulta."
    return "\n\n".join(chunk.to_context_string() for chunk in chunks)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--protocols-dir", type=Path, default=DEFAULT_PROTOCOLS_DIR)
    args = parser.parse_args()

    retriever = ProtocolRetriever.from_protocols(args.protocols_dir)
    for chunk in retriever.retrieve(args.query, k=args.k):
        print(f"--- {chunk.protocol_id} (score={chunk.score:.3f}) ---")
        print(chunk.trecho[:300])
        print()


if __name__ == "__main__":
    main()
