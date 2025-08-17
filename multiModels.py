import os
import time
import json
import uuid
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from together import Together
except Exception:
    Together = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

import boto3
from botocore.exceptions import ClientError

try:
    from sagemaker import Session
    from sagemaker.jumpstart.model import JumpStartModel
except Exception:
    Session = None
    JumpStartModel = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

class BedrockOnDemandNotSupported(Exception):
    pass

class MultiProviderOrchestrator:
    def __init__(
        self,
        region_bedrock: str = "us-east-1",
        region_jumpstart: str = "us-east-2",
        jumpstart_exec_role_arn: Optional[str] = None,
        default_js_instance: str = "ml.g5.2xlarge",
        max_new_tokens: int = 128,
    ):
        self.region_bedrock = region_bedrock
        self.region_jumpstart = region_jumpstart
        self.jumpstart_exec_role_arn = jumpstart_exec_role_arn
        self.default_js_instance = default_js_instance
        self.max_new_tokens = max_new_tokens

        self.bedrock_model_lookup: Dict[str, Dict[str, str]] = {}

        self.openai_client: Optional[OpenAI] = None
        self.together_client: Optional[Together] = None
        self.gemini_ready: bool = False
        self.bedrock_client = None
        self.bedrock_runtime = None
        self.sm_client = None
        self.sm_runtime = None
        self.sm_session = None

        self.discovered: Dict[str, List[str]] = {p: [] for p in ["openai", "together", "gemini", "bedrock", "jumpstart"]}
        self._ephemeral_endpoints: List[str] = []

    def init_clients(self):
        log.info("Initializing provider clients...")

        boto3.setup_default_session(region_name=self.region_bedrock)

        self.bedrock_client = boto3.client("bedrock", region_name=self.region_bedrock)
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=self.region_bedrock)

        # Load Bedrock model mappings after Bedrock client init
        self.load_bedrock_model_mappings()

        js = boto3.session.Session(region_name=self.region_jumpstart)
        self.sm_client = js.client("sagemaker")
        self.sm_runtime = js.client("sagemaker-runtime")
        self.sm_session = Session(boto_session=js) if Session else None

        log.info("✅ AWS clients ready")

        if OpenAI and os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                organization=os.getenv("OPENAI_ORG_ID"),
                project=os.getenv("OPENAI_PROJECT"),
            )
            log.info("✅ OpenAI ready")
        else:
            log.info("Skipping OpenAI (no SDK or API key)")

        if Together and os.getenv("TOGETHER_API_KEY"):
            self.together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
            log.info("✅ Together ready")
        else:
            log.info("Skipping Together (no SDK or API key)")

        if genai and os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.gemini_ready = True
            log.info("✅ Gemini ready")
        else:
            log.info("Skipping Gemini (no SDK or API key)")

    def load_bedrock_model_mappings(self):
        """Map both user nicknames, modelName, and modelId to the correct modelId and modelArn."""
        self.bedrock_model_lookup = {}
        if not self.bedrock_client:
            log.warning("Bedrock client not initialized; skipping Bedrock model ID mapping.")
            return
        try:
            resp = self.bedrock_client.list_foundation_models()
            for model_summary in resp.get("modelSummaries", []):
                keys = []
                if "modelName" in model_summary:
                    keys.append(model_summary["modelName"].lower())
                if "modelId" in model_summary:
                    keys.append(model_summary["modelId"].lower())
                for k in keys:
                    self.bedrock_model_lookup[k] = {
                        "modelId": model_summary["modelId"],
                        "modelArn": model_summary.get("modelArn")
                    }
            log.info(f"Loaded {len(self.bedrock_model_lookup)} Bedrock model IDs/ARNs.")
            # Print all model names for reference
            for model_summary in resp.get("modelSummaries", []):
                log.info(f"Bedrock modelName: '{model_summary.get('modelName')}', modelId: '{model_summary.get('modelId')}'")
        except Exception as e:
            log.warning(f"Bedrock model mapping failed: {e}")

    def resolve_bedrock_model_id_by_pretty_name(self, nickname: str) -> Optional[str]:
        if not self.bedrock_client:
            log.warning("Bedrock client not initialized; cannot resolve nickname.")
            return None
        try:
            resp = self.bedrock_client.list_foundation_models()
            for summary in resp.get('modelSummaries', []):
                name = summary.get('modelName', '').replace(" ", "").lower()
                if nickname.replace(" ", "").lower() in name:
                    mid = summary['modelId']
                    # Prefer base modelId (no :12k, :28k suffix)
                    if ':' in mid and mid.count(':') > 1:
                        base_id = ':'.join(mid.split(':')[:-1])
                        log.info(f"Resolved Bedrock nickname '{nickname}' to base modelId: {base_id} (was {mid}, modelName: {summary.get('modelName')})")
                        return base_id
                    log.info(f"Resolved Bedrock nickname '{nickname}' to modelId: {mid} (modelName: {summary.get('modelName')})")
                    return mid
            log.warning(f"No Bedrock model found for nickname '{nickname}'.")
            return None
        except Exception as e:
            log.warning(f"Failed to resolve Bedrock nickname '{nickname}': {e}")
            return None

    def run_openai(self, prompt: str, model: str) -> Tuple[str, float]:
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        t0 = time.time()
        out = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.7,
        )
        txt = out.choices[0].message.content
        return txt, time.time() - t0

    def run_together(self, prompt: str, model: str) -> Tuple[str, float]:
        if not self.together_client:
            raise RuntimeError("Together client not initialized")
        t0 = time.time()
        out = self.together_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.7,
        )
        txt = out.choices[0].message.content
        return txt, time.time() - t0

    def run_gemini(self, prompt: str, model: str) -> Tuple[str, float]:
        if not self.gemini_ready:
            raise RuntimeError("Gemini not initialized")
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

    def run_bedrock(self, prompt: str, model_id: str) -> Tuple[str, float]:
        t0 = time.time()

        lookup = self.bedrock_model_lookup or {}
        model_key = model_id.lower()
        resolved = lookup.get(model_key)
        resolved_model_id = model_id
        model_arn = None  # initialize

        if resolved:
            resolved_model_id = resolved["modelId"]
            model_arn = resolved.get("modelArn")

        family = resolved_model_id.split(".")[0]
        body: Dict[str, Any]

        if family == "anthropic":
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_new_tokens,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            }
        else:
            body = {"inputText": prompt}

        if model_arn:
            log.info(f"Using Bedrock model ARN: {model_arn} for model_id: {resolved_model_id}")
        else:
            log.info(f"No ARN found for Bedrock model_id: {resolved_model_id}")

        try:
            resp = self.bedrock_runtime.invoke_model(
                modelId=resolved_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            raw = resp["body"].read().decode("utf-8")
            if family == "anthropic":
                parsed = json.loads(raw)
                msg = parsed.get("output", {}).get("message") or parsed.get("content") or parsed.get("messages")
                if isinstance(msg, list):
                    txt = []
                    for m in msg:
                        if isinstance(m, dict) and "content" in m:
                            for part in m["content"]:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    txt.append(part.get("text", ""))
                    response_text = "\n".join(txt).strip()
                else:
                    response_text = str(msg)
            else:
                response_text = raw
            return response_text, time.time() - t0
        except Exception as e:
            if "on-demand throughput isn’t supported" in str(e):
                raise BedrockOnDemandNotSupported(f"Bedrock on-demand not supported for {resolved_model_id}")
            raise

    def find_jumpstart_equivalent(self, canonical_name: str) -> Optional[str]:
        """
        A simple lookup to find a jumpstart model ID similar to the canonical_name.
        Extend this list as needed.
        """
        available_models = [
            "meta.llama3-8b-instruct-v1:0",
            "meta.llama3-70b-instruct-v1:0",
            "mistral.mistral-7b-instruct-v0:2",
            "mistral.mixtral-8x7b-instruct-v0:1",
        ]
        for mid in available_models:
            if canonical_name.replace(' ', '').lower() in mid.replace('-', '').replace(' ', '').lower():
                return mid
        return None

    def run_jumpstart(self, prompt: str, model_id: str) -> Tuple[str, float]:
        if not JumpStartModel or not self.sm_session:
            raise RuntimeError("SageMaker SDK not available for JumpStart")

        accept_eula = "llama" in model_id.lower()
        endpoint_name = self._js_endpoint_name("js-ep")
        log.info(f"🚀 [JumpStart] Deploying {model_id} -> {endpoint_name} ({self.default_js_instance}) in {self.region_jumpstart}")

        model = JumpStartModel(
            model_id=model_id,
            role=self.jumpstart_exec_role_arn,
            sagemaker_session=self.sm_session,
        )

        _t0 = time.time()
        try:
            predictor = model.deploy(
                initial_instance_count=1,
                instance_type=self.default_js_instance,
                endpoint_name=endpoint_name,
                accept_eula=accept_eula,
            )
        except ClientError as e:
            raise RuntimeError(f"JumpStart deploy failed: {e}")

        self._ephemeral_endpoints.append(endpoint_name)
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": self.max_new_tokens}}

        try:
            resp = self.sm_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload).encode("utf-8"),
            )
            body = resp["Body"].read().decode("utf-8", errors="ignore")
        except Exception as e:
            body = f"[invoke error] {e}"

        self.cleanup_jumpstart_endpoint(endpoint_name)
        return body, time.time() - _t0

    def cleanup_jumpstart_endpoint(self, endpoint_name: str):
        sm = self.sm_client
        try:
            sm.delete_endpoint(EndpointName=endpoint_name)
            sm.get_waiter("endpoint_deleted").wait(EndpointName=endpoint_name)
        except Exception as e:
            log.warning(f"(warn) delete_endpoint: {e}")

        try:
            epc_name = None
            try:
                ep = sm.describe_endpoint(EndpointName=endpoint_name)
                epc_name = ep.get("EndpointConfigName")
            except Exception:
                pass

            if not epc_name:
                cfgs = sm.list_endpoint_configs(NameContains=endpoint_name).get("EndpointConfigs", [])
                if cfgs:
                    epc_name = cfgs[0]["EndpointConfigName"]

            if epc_name:
                epc = sm.describe_endpoint_config(EndpointConfigName=epc_name)
                for v in epc.get("ProductionVariants", []):
                    mname = v.get("ModelName")
                    if mname:
                        try:
                            sm.delete_model(ModelName=mname)
                        except Exception as e:
                            log.warning(f"(warn) delete_model {mname}: {e}")
                try:
                    sm.delete_endpoint_config(EndpointConfigName=epc_name)
                except Exception as e:
                    log.warning(f"(warn) delete_endpoint_config: {e}")
        except Exception as e:
            log.warning(f"(warn) post-delete describe: {e}")

        try:
            sm.describe_endpoint(EndpointName=endpoint_name)
            log.warning(f"(verify) endpoint still describable: {endpoint_name}")
        except Exception:
            log.info(f"✅ [JumpStart] Endpoint {endpoint_name} deleted")

    def _js_endpoint_name(self, prefix: str = "js-ep") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8]}-{int(time.time())}"

    def run_entry(self, entry: CatalogEntry, prompt: str) -> RunRecord:
        try:
            if entry.provider == "openai":
                text, rt = self.run_openai(prompt, entry.provider_model_id)
            elif entry.provider == "together":
                text, rt = self.run_together(prompt, entry.provider_model_id)
            elif entry.provider == "gemini":
                text, rt = self.run_gemini(prompt, entry.provider_model_id)
            elif entry.provider == "bedrock":
                try:
                    text, rt = self.run_bedrock(prompt, entry.provider_model_id)
                except BedrockOnDemandNotSupported:
                    jumpstart_model_id = self.find_jumpstart_equivalent(entry.canonical_name)
                    if jumpstart_model_id:
                        log.info(f"Falling back to JumpStart for model {entry.canonical_name}: {jumpstart_model_id}")
                        text, rt = self.run_jumpstart(prompt, jumpstart_model_id)
                    else:
                        raise RuntimeError(f"No JumpStart equivalent found for {entry.canonical_name}")
            elif entry.provider == "jumpstart":
                text, rt = self.run_jumpstart(prompt, entry.provider_model_id)
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

def main():
    ORCH = MultiProviderOrchestrator(
        region_bedrock="us-east-1",
        region_jumpstart="us-east-2",
        jumpstart_exec_role_arn=os.getenv("JUMPSTART_EXEC_ROLE_ARN"),
        default_js_instance="ml.g5.2xlarge",
        max_new_tokens=128,
    )

    log.info("="*50)
    ORCH.init_clients()

    with open("top10_models.json", "r") as f:
        data = json.load(f)

    model_list = data.get("models", [])

    top10: List[CatalogEntry] = []
    for model_id in model_list:
        if model_id.startswith("gpt"):
            provider = "openai"
            family = "gpt"
            provider_model_id = model_id
        elif model_id.lower().startswith("claude") or model_id in [
            "Opus 4", "Sonnet 4", "Haiku 3.5", "Opus 3",
            "Sonnet 3.7", "Sonnet 3.6", "Sonnet 3.5"]:
            provider = "bedrock"
            family = "claude"
            resolved_id = ORCH.resolve_bedrock_model_id_by_pretty_name(model_id)
            if not resolved_id:
                log.warning(f"Skipping unknown Bedrock nickname: {model_id}")
                continue
            provider_model_id = resolved_id
        elif model_id.startswith("llama"):
            provider = "jumpstart"
            family = "llama"
            provider_model_id = model_id
        elif model_id.startswith("mistral"):
            provider = "jumpstart"
            family = "mistral"
            provider_model_id = model_id
        elif model_id.startswith("gemini"):
            provider = "gemini"
            family = "gemini"
            provider_model_id = model_id
        else:
            provider = "unknown"
            family = "unknown"
            provider_model_id = model_id

        top10.append(
            CatalogEntry(
                canonical_name=model_id.lower(),
                provider=provider,
                provider_model_id=provider_model_id,
                meta={"family": family},
            )
        )

    log.info("=== Top 10 Loaded from JSON ===")
    for i, e in enumerate(top10, 1):
        log.info(f"{i:2d}. [{e.provider}] {e.canonical_name} -> {e.provider_model_id}")

    queries = [
        "In one sentence, explain why caching improves performance.",
        "Give three concise bullet points on vector databases.",
    ]

    results: List[RunRecord] = []
    for e in top10:
        for q in queries:
            log.info(f"▶ Running [{e.provider}] {e.provider_model_id} :: {q[:48]}…")
            rec = ORCH.run_entry(e, q)
            results.append(rec)
            if rec.ok:
                log.info(f"✓ {e.provider}:{e.canonical_name} [{rec.response_time:.2f}s] {rec.response_preview}")
            else:
                log.warning(f"✗ {e.provider}:{e.canonical_name} ERROR: {rec.error}")
            time.sleep(0.25)

    ok = sum(1 for r in results if r.ok)
    log.info("="*50)
    log.info(f"Done. {ok}/{len(results)} successful.")

    by_provider: Dict[str, Tuple[int, int]] = {}
    for r in results:
        sc, tot = by_provider.get(r.provider, (0, 0))
        by_provider[r.provider] = (sc + (1 if r.ok else 0), tot + 1)

    for p, (sc, tot) in by_provider.items():
        log.info(f" {p:10s}: {sc}/{tot} ok")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
