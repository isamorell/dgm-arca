import threading
from functools import lru_cache

_lock = threading.Lock()
_llms: dict = {}


def get_llm(model_id: str, adapter: str | None = None):
    """(tokenizer, model) cacheado por (model_id, adapter). Seguro entre hilos."""
    key = (model_id, adapter)
    with _lock:
        if key not in _llms:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter)
            _llms[key] = (tok, model.eval())
        return _llms[key]


_embedder_lock = threading.Lock()

@lru_cache(maxsize=1)
def _make_embedder():
    from rag.embed import Embedder
    return Embedder()

def get_embedder():
    with _embedder_lock:
        return _make_embedder()