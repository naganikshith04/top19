import time
from typing import Optional, Tuple, List
import google.generativeai as genai
import logging

log = logging.getLogger("connectors.gemini")

class GeminiConnector:
    def __init__(self, api_key: Optional[str], max_new_tokens: int = 128):
        if api_key:
            genai.configure(api_key=api_key)
            self.ready = True
        else:
            self.ready = False
        self.max_new_tokens = max_new_tokens

    def available(self):
        return self.ready

    def invoke(self, prompt: str, model: str) -> Tuple[str, float]:
        if not self.ready:
            raise RuntimeError("Gemini client not initialized")
        model_id = model.split("/")[-1] if "/" in model else model
        m = genai.GenerativeModel(model_id)
        t0 = time.time()
        out = m.generate_content(prompt)
        txt = getattr(out, "text", None)
        if not txt and getattr(out, "candidates", None):
            chunks = []
            for c in out.candidates:
                if getattr(c, "content", None) and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        if hasattr(p, "text"):
                            chunks.append(p.text)
            txt = "\n".join(chunks) if chunks else ""
        return txt or "", time.time() - t0

    def list_models(self) -> List[str]:
        if not self.ready:
            raise RuntimeError("Gemini client not initialized")
        resp = genai.list_models()
        model_ids = [m.name for m in resp]
        log.info(f"Available Gemini models: {model_ids}")
        return model_ids
