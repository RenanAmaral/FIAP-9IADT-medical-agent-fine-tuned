"""Autenticação na Hugging Face Hub.

Sem um token, a Hub aplica limites de taxa por IP — o que no Colab significa
compartilhar a cota com todos os outros notebooks que saem pelo mesmo IP.
Daí o aviso "You are sending unauthenticated requests to the HF Hub".
O download funciona, mas fica mais lento e pode falhar com HTTP 429 no meio
de um arquivo de 2 GB.

Um token de leitura (gratuito, em https://huggingface.co/settings/tokens)
resolve. Ele também é **obrigatório** para modelos com licença restrita
(Llama 3, Gemma, Mistral em alguns casos), que exigem aceitar os termos na
página do modelo antes do download.

Ordem de busca:
1. Variável de ambiente `HF_TOKEN` (ou `HUGGING_FACE_HUB_TOKEN`).
2. Secrets do Colab (`google.colab.userdata`), se estivermos no Colab.
3. Login já persistido em `~/.cache/huggingface/token`.
"""

from __future__ import annotations

import os


def _token_from_env() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(var)
        if token:
            return token.strip()
    return None


def _token_from_colab_secrets() -> str | None:
    """Lê o secret `HF_TOKEN` do Colab, se disponível.

    `userdata.get` levanta exceções específicas quando o secret não existe ou
    não foi autorizado para o notebook; nenhuma delas deve interromper o
    treino, então tratamos qualquer falha como "sem token".
    """
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        token = userdata.get("HF_TOKEN")
    except Exception:
        return None

    return token.strip() if token else None


def _token_already_persisted() -> bool:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return False

    try:
        return bool(get_token())
    except Exception:
        return False


def ensure_hf_login(verbose: bool = True) -> bool:
    """Autentica na Hub se houver token disponível.

    Devolve `True` se a sessão está autenticada. Nunca levanta exceção nem
    interrompe o fluxo: sem token, o download de modelos públicos continua
    funcionando, apenas sujeito a limite de taxa.
    """
    if _token_already_persisted():
        if verbose:
            print("[hf] Sessão já autenticada na Hugging Face Hub.")
        return True

    token = _token_from_env() or _token_from_colab_secrets()

    if not token:
        if verbose:
            print(
                "[hf] Sem HF_TOKEN — os downloads serão anônimos e sujeitos a "
                "limite de taxa por IP (mais lentos, e podem falhar com HTTP 429).\n"
                "     Para autenticar, crie um token de leitura em "
                "https://huggingface.co/settings/tokens e:\n"
                "       • no Colab: adicione-o em Secrets (🔑) com o nome HF_TOKEN\n"
                "                   e ative 'Notebook access';\n"
                "       • localmente: export HF_TOKEN=hf_xxx"
            )
        return False

    # Propaga para as libs que leem a variável diretamente.
    os.environ.setdefault("HF_TOKEN", token)

    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except ImportError:
        if verbose:
            print("[hf] huggingface_hub não instalado; usando apenas a variável de ambiente.")
        return True
    except Exception as exc:
        if verbose:
            print(f"[hf] Falha ao autenticar ({type(exc).__name__}: {exc}). Seguindo sem login.")
        return False

    if verbose:
        print("[hf] Autenticado na Hugging Face Hub.")
    return True


def dtype_kwarg(torch_dtype) -> dict:
    """Nome do parâmetro de dtype aceito pelo `transformers` instalado.

    O parâmetro passou de `torch_dtype` para `dtype` no transformers 5.0.
    Como `from_pretrained` recebe `**kwargs`, um nome errado não levanta erro:
    em 4.x um `dtype=` seria simplesmente ignorado e o modelo carregaria em
    fp32 — mais lento e ocupando o dobro da memória, sem nenhum sinal de que
    algo deu errado. Por isso escolhemos pelo número de versão.
    """
    try:
        import transformers

        major = int(transformers.__version__.split(".")[0])
    except Exception:
        major = 4

    return {"dtype": torch_dtype} if major >= 5 else {"torch_dtype": torch_dtype}


__all__ = ["dtype_kwarg", "ensure_hf_login"]
