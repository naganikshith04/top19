import boto3
import json
import time
import logging
from typing import Optional, Tuple, List
from botocore.exceptions import ClientError
from sagemaker import Session
from sagemaker.jumpstart.model import JumpStartModel

log = logging.getLogger("connectors.jumpstart")

class JumpStartConnector:
    def __init__(self, region: str, role_arn: str, default_instance: str, max_new_tokens: int = 128):
        self.region = region
        self.role_arn = role_arn
        self.default_instance = default_instance
        self.max_new_tokens = max_new_tokens
        js = boto3.session.Session(region_name=region)
        self.sm_client = js.client("sagemaker")
        self.sm_runtime = js.client("sagemaker-runtime")
        self.sm_session = Session(boto_session=js)

    def deploy_invoke_cleanup(self, prompt: str, model_id: str) -> Tuple[str, float]:
        endpoint_name = f"js-ep-{str(int(time.time()))}"
        model = JumpStartModel(
            model_id=model_id,
            role=self.role_arn,
            sagemaker_session=self.sm_session,
        )
        start_time = time.time()
        try:
            model.deploy(
                initial_instance_count=1,
                instance_type=self.default_instance,
                endpoint_name=endpoint_name,
                accept_eula=True,
            )
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": self.max_new_tokens}}
            resp = self.sm_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload).encode("utf-8"),
            )
            body = resp["Body"].read().decode("utf-8", errors="ignore")
        except Exception as e:
            body = f"[invoke error] {e}"
        finally:
            self.cleanup(endpoint_name)
        elapsed = time.time() - start_time
        return body, elapsed

    def cleanup(self, endpoint_name: str):
        sm = self.sm_client
        try:
            sm.delete_endpoint(EndpointName=endpoint_name)
            sm.get_waiter("endpoint_deleted").wait(EndpointName=endpoint_name)
        except Exception as e:
            log.warning(f"Could not delete endpoint: {e}")
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
                            log.warning(f"Could not delete model {mname}: {e}")
                try:
                    sm.delete_endpoint_config(EndpointConfigName=epc_name)
                except Exception as e:
                    log.warning(f"Could not delete endpoint config: {e}")
        except Exception as e:
            log.warning(f"Error during cleanup: {e}")

    def list_models(self) -> List[str]:
        """
        Optionally list available JumpStart models.
        Logic will depend on your SageMaker JumpStart setup or Marketplace access.
        """
        try:
            resp = self.sm_client.list_models()
            model_names = [m['ModelName'] for m in resp.get('Models', [])]
            log.info(f"Available JumpStart models: {model_names}")
            return model_names
        except Exception as e:
            log.warning(f"Could not list JumpStart models: {e}")
            return []
    