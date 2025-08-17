import time
from typing import Optional, Tuple, List
from together import Together
import logging

log = logging.getLogger("connectors.together")

class TogetherConnector:
    def __init__(self, api_key: Optional[str], max_new_tokens: int = 128):
        if api_key:
            self.client = Together(api_key=api_key)
        else:
            self.client = None
        self.max_new_tokens = max_new_tokens

    def available(self):
        return self.client is not None

    def invoke(self, prompt: str, model: str) -> Tuple[str, float]:
        if not self.client:
            raise RuntimeError("Together client not initialized")
        t0 = time.time()
        out = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.7,
        )
        txt = out.choices[0].message.content
        return txt, time.time() - t0

    def list_models(self) -> List[str]:
        if not self.client:
            raise RuntimeError("Together client not initialized")
        resp = self.client.models.list()
        # Handle both list-of-strings and dict-with-'models' format
        if isinstance(resp, list):
            model_ids = [str(m) for m in resp]
        elif isinstance(resp, dict) and 'models' in resp:
            model_ids = [str(m) for m in resp['models']]
        else:
            raise RuntimeError(f"Unexpected Together models response format: {resp!r}")
        log.info(f"Available Together models: {model_ids}")
        return model_ids
