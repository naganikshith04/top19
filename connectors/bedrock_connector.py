import boto3
import json
import logging
import time
from typing import Optional, Dict, Tuple, List

log = logging.getLogger("connectors.bedrock")

class BedrockOnDemandNotSupported(Exception):
    pass

class BedrockConnector:
    def __init__(self, region: str = "us-east-1", max_new_tokens: int = 128):
        self.region = region
        self.max_new_tokens = max_new_tokens
        self.client = boto3.client("bedrock", region_name=region)
        self.runtime = boto3.client("bedrock-runtime", region_name=region)
        self.model_lookup: Dict[str, Dict[str, str]] = {}
        self.load_model_mappings()

    def load_model_mappings(self):
        self.model_lookup = {}
        try:
            resp = self.client.list_foundation_models()
            for model_summary in resp.get("modelSummaries", []):
                keys = []
                if "modelName" in model_summary:
                    keys.append(model_summary["modelName"].lower())
                if "modelId" in model_summary:
                    keys.append(model_summary["modelId"].lower())
                for k in keys:
                    self.model_lookup[k] = {
                        "modelId": model_summary["modelId"],
                        "modelArn": model_summary.get("modelArn"),
                    }
            log.info(f"Loaded {len(self.model_lookup)} Bedrock model IDs/ARNs.")
            for model_summary in resp.get("modelSummaries", []):
                log.info(f"Bedrock modelName: '{model_summary.get('modelName')}', modelId: '{model_summary.get('modelId')}'")
        except Exception as e:
            log.warning(f"Bedrock model mapping failed: {e}")

    def list_models(self) -> List[str]:
        try:
            resp = self.client.list_foundation_models()
            names = [model_summary["modelName"] for model_summary in resp.get("modelSummaries", [])]
            log.info(f"Available Bedrock models: {names}")
            return names
        except Exception as e:
            log.warning(f"Could not list Bedrock models: {e}")
            return []

    def resolve_model_id(self, nickname: str) -> Optional[str]:
        try:
            resp = self.client.list_foundation_models()
            for summary in resp.get('modelSummaries', []):
                name = summary.get('modelName', '').replace(" ", "").lower()
                if nickname.replace(" ", "").lower() in name:
                    mid = summary['modelId']
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

    # bedrock_connector.py - replace invoke() up to the runtime.invoke_model() call

def invoke(self, prompt: str, model_id: str) -> Tuple[str, float]:
    t0 = time.time()
    lookup = self.model_lookup or {}
    model_key = model_id.lower()
    resolved = lookup.get(model_key)
    resolved_model_id = model_id
    model_arn = None
    if resolved:
        resolved_model_id = resolved["modelId"]
        model_arn = resolved.get("modelArn")

    # Model family router
    family = resolved_model_id.split(".")[0].lower()

    if family == "anthropic":
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_new_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
    elif family in ("mistral", "meta"):
        # Text-generation style (prompt)
        body = {
            "prompt": prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.7
        }
    elif family in ("cohere", "ai21", "amazon", "stability"):
        # Keep a generic prompt contract; many accept 'prompt', some accept 'inputText'.
        # Try 'prompt' first; Bedrock will reject invalid fields with ValidationException.
        body = {
            "prompt": prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.7
        }
    else:
        # Last-resort legacy shape
        body = {"inputText": prompt}

    try:
        resp = self.runtime.invoke_model(
            modelId=resolved_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        raw = resp["body"].read().decode("utf-8")
        return raw, time.time() - t0
    except Exception as e:
        msg = str(e)
        if "on-demand throughput isn’t supported" in msg or "Inference profile" in msg:
            raise BedrockOnDemandNotSupported(f"Bedrock on-demand not supported for {resolved_model_id}")
        # Some providers demand a different shape; retry once with inputText
        if "ValidationException" in msg and "prompt" in body:
            try:
                fallback_body = {"inputText": prompt}
                resp = self.runtime.invoke_model(
                    modelId=resolved_model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(fallback_body),
                )
                raw = resp["body"].read().decode("utf-8")
                return raw, time.time() - t0
            except Exception:
                pass
        raise
