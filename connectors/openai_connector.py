import time
from typing import Optional, Tuple, List
from openai import OpenAI
import logging

log = logging.getLogger("connectors.openai")

class OpenAIConnector:
    def __init__(self, api_key: Optional[str], organization: Optional[str] = None, project: Optional[str] = None, max_new_tokens: int = 128):
        if api_key:
            self.client = OpenAI(api_key=api_key, organization=organization, project=project)
        else:
            self.client = None
        self.max_new_tokens = max_new_tokens

    def available(self):
        return self.client is not None

    def invoke(self, prompt: str, model: str) -> Tuple[str, float]:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
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
            raise RuntimeError("OpenAI client not initialized")
        resp = self.client.models.list()
        model_ids = [m.id for m in resp.data]
        log.info(f"Available OpenAI models: {model_ids}")
        return model_ids
