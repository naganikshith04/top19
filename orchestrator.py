import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from connectors.bedrock_connector import BedrockConnector, BedrockOnDemandNotSupported
from connectors.jumpstart_connector import JumpStartConnector
from connectors.openai_connector import OpenAIConnector
from connectors.together_connector import TogetherConnector
from connectors.gemini_connector import GeminiConnector

log = logging.getLogger("orchestrator")

@dataclass
class CatalogEntry:
    canonical_name: str
    provider: str
    provider_model_id: str
    meta: Dict[str, Any]

@dataclass
class RunRecord:
    provider: str
    model_id: str
    canonical_name: str
    query: str
    ok: bool
    response_time: Optional[float] = None
    response_preview: Optional[str] = None
    error: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class MultiProviderOrchestrator:
    def __init__(
        self,
        region_bedrock: str,
        region_jumpstart: str,
        jumpstart_exec_role_arn: Optional[str],
        default_js_instance: str = "ml.g5.2xlarge",
        max_new_tokens: int = 128,
    ):
        self.bedrock = BedrockConnector(region=region_bedrock, max_new_tokens=max_new_tokens)
        self.jumpstart = JumpStartConnector(
            region=region_jumpstart,
            role_arn=jumpstart_exec_role_arn,
            default_instance=default_js_instance,
        )
        self.openai = OpenAIConnector(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORG_ID"),
            project=os.getenv("OPENAI_PROJECT"),
            max_new_tokens=max_new_tokens,
        )
        self.together = TogetherConnector(
            api_key=os.getenv("TOGETHER_API_KEY"),
            max_new_tokens=max_new_tokens,
        )
        self.gemini = GeminiConnector(
            api_key=os.getenv("GOOGLE_API_KEY"),
            max_new_tokens=max_new_tokens,
        )

    def init_clients(self):
        # Clients initialized in connectors' __init__, so nothing extra here
        log.info("Provider clients initialized.")

    def find_jumpstart_equivalent(self, canonical_name: str) -> Optional[str]:
        mapping = {
            "claude 3 sonnet": "meta.llama3-8b-instruct-v1:0",
            "claude 3 haiku": "meta.llama3-70b-instruct-v1:0",
            "claude 3 opus": "mistral.mistral-7b-instruct-v0:2",
        }
        return mapping.get(canonical_name.lower())

    def run_bedrock(self, prompt: str, model_id: str) -> Tuple[str, float]:
        return self.bedrock.invoke(prompt, model_id)

    def run_jumpstart(self, prompt: str, model_id: str) -> Tuple[str, float]:
        return self.jumpstart.deploy_invoke_cleanup(prompt, model_id)

    def run_openai(self, prompt: str, model_id: str) -> Tuple[str, float]:
        return self.openai.invoke(prompt, model_id)

    def run_together(self, prompt: str, model_id: str) -> Tuple[str, float]:
        return self.together.invoke(prompt, model_id)

    def run_gemini(self, prompt: str, model_id: str) -> Tuple[str, float]:
        return self.gemini.invoke(prompt, model_id)

    def run_entry(self, entry: CatalogEntry, prompt: str) -> RunRecord:
        try:
            if entry.provider == "bedrock":
                try:
                    text, rt = self.run_bedrock(prompt, entry.provider_model_id)
                except BedrockOnDemandNotSupported:
                    jumpstart_id = self.find_jumpstart_equivalent(entry.canonical_name)
                    if jumpstart_id:
                        log.info(f"Falling back to JumpStart for model {entry.canonical_name}: {jumpstart_id}")
                        text, rt = self.run_jumpstart(prompt, jumpstart_id)
                    else:
                        raise RuntimeError(f"No JumpStart equivalent found for {entry.canonical_name}")
            elif entry.provider == "jumpstart":
                text, rt = self.run_jumpstart(prompt, entry.provider_model_id)
            elif entry.provider == "openai":
                text, rt = self.run_openai(prompt, entry.provider_model_id)
            elif entry.provider == "together":
                text, rt = self.run_together(prompt, entry.provider_model_id)
            elif entry.provider == "gemini":
                text, rt = self.run_gemini(prompt, entry.provider_model_id)
            else:
                raise RuntimeError(f"Unknown provider: {entry.provider}")

            preview = (text or "").strip().replace("\r", " ").replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + "…"

            return RunRecord(
                provider=entry.provider,
                model_id=entry.provider_model_id,
                canonical_name=entry.canonical_name,
                query=prompt,
                ok=True,
                response_time=rt,
                response_preview=preview,
            )
        except Exception as e:
            return RunRecord(
                provider=entry.provider,
                model_id=entry.provider_model_id,
                canonical_name=entry.canonical_name,
                query=prompt,
                ok=False,
                error=str(e),
            )
