# index-models.py (deploy → invoke → cleanup → verify-deleted)
import argparse, time, sys
import boto3
from botocore.exceptions import ClientError
from sagemaker import Session
from sagemaker.jumpstart.model import JumpStartModel

def make_session(region: str):
    sess = boto3.session.Session(region_name=region)
    return sess, sess.client("sagemaker"), sess.client("sagemaker-runtime")

def pick_fallback_instance(model_id: str):
    mid = model_id.lower()
    if "120b" in mid:
        return "ml.p5.48xlarge"
    if "20b" in mid:
        return "ml.g5.2xlarge"
    if "llama-3-1-8b" in mid or "8b" in mid:
        return "ml.g5.2xlarge"
    return "ml.g5.2xlarge"

def cleanup_all(sm_client, endpoint_name: str):
    """Delete endpoint, endpoint-config, and model(s). Caches names first to avoid race conditions."""
    epc_name = None
    model_names = []
    try:
        ep = sm_client.describe_endpoint(EndpointName=endpoint_name)
        epc_name = ep.get("EndpointConfigName")
        if epc_name:
            epc = sm_client.describe_endpoint_config(EndpointConfigName=epc_name)
            model_names = [v.get("ModelName") for v in epc.get("ProductionVariants", []) if v.get("ModelName")]
    except ClientError as e:
        print(f"(warn) pre-delete describe failed: {e}", file=sys.stderr)

    # Delete endpoint
    try:
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        sm_client.get_waiter("endpoint_deleted").wait(EndpointName=endpoint_name)
    except ClientError as e:
        print(f"(warn) delete_endpoint: {e}", file=sys.stderr)

    # Delete endpoint config
    if epc_name:
        try:
            sm_client.delete_endpoint_config(EndpointConfigName=epc_name)
        except ClientError as e:
            print(f"(warn) delete_endpoint_config {epc_name}: {e}", file=sys.stderr)

    # Delete models
    for m in model_names:
        try:
            sm_client.delete_model(ModelName=m)
        except ClientError as e:
            print(f"(warn) delete_model {m}: {e}", file=sys.stderr)

def verify_deleted(sm_client, rt_client, endpoint_name: str, strict: bool = True):
    """Verify endpoint is really gone:
       1) describe-endpoint should fail
       2) invoke to the same name should fail
       3) (optional) list checks
       Returns True if all signals say it's deleted; False otherwise.
    """
    ok = True

    # 1) Describe should fail
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print("⚠️  describe-endpoint unexpectedly succeeded (endpoint may still exist).", file=sys.stderr)
        ok = False
    except ClientError as e:
        print(f"✅ describe-endpoint failed as expected: {e.response['Error'].get('Code', 'Error')}")

    # 2) Invoke should fail
    try:
        rt_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=b'{"inputs":"ping"}'
        )
        print("⚠️  invoke unexpectedly succeeded (endpoint may still exist).", file=sys.stderr)
        ok = False
    except ClientError as e:
        print(f"✅ invoke failed as expected: {e.response['Error'].get('Code', 'Error')}")

    # 3) Optional list checks
    try:
        eps = sm_client.list_endpoints(NameContains=endpoint_name).get("Endpoints", [])
        if any(ep.get("EndpointName") == endpoint_name for ep in eps):
            print("⚠️  list-endpoints still shows the endpoint.", file=sys.stderr)
            ok = False
        else:
            print("✅ list-endpoints does not show the endpoint.")
    except ClientError as e:
        print(f"(warn) list_endpoints: {e}", file=sys.stderr)

    if strict and not ok:
        print("❌ Verification detected leftover endpoint resources.", file=sys.stderr)
    else:
        print("🧾 Verification complete.")

    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--exec-role-arn", required=True)
    ap.add_argument("--endpoint-name-prefix", default="jumpstart-ephemeral")
    ap.add_argument("--instance-type", default="ml.g5.2xlarge")
    ap.add_argument("--model-id", default="meta-textgeneration-llama-3-1-8b-instruct")
    ap.add_argument("--model-version", default="*",
                    help="Optionally pin e.g. 2.10.0 for stability; * uses latest")
    ap.add_argument("--payload", default='{"inputs":"Hello from JumpStart!","parameters":{"max_new_tokens":64}}')
    ap.add_argument("--content-type", default="application/json")
    ap.add_argument("--accept", default="application/json")
    ap.add_argument("--accept-eula", action="store_true",
                    help="Required for some JumpStart models (e.g., Llama).")
    ap.add_argument("--strict-verify", action="store_true",
                    help="Exit non-zero if post-cleanup verification detects leftovers.")
    args = ap.parse_args()

    # Region-scoped clients/sessions
    boto_sess, sm, rt = make_session(args.region)
    sm_session = Session(boto_session=boto_sess)

    # Construct JumpStart model
    model = JumpStartModel(
        model_id=args.model_id,
        model_version=args.model_version,
        role=args.exec_role_arn,
        sagemaker_session=sm_session
    )

    endpoint_name = f"{args.endpoint_name_prefix}-{int(time.time())}"
    instance_type = args.instance_type
    print(f"Using model '{args.model_id}' with version '{args.model_version}'.")
    print(f"🚀 Deploying {model.model_id} to {endpoint_name} in {args.region} (instance={instance_type})")

    # Deploy with one-shot fallback for “unsupported instance type”
    try:
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            accept_eula=args.accept_eula,
        )
    except ClientError as e:
        msg = str(e)
        if "failed to satisfy constraint: Member must satisfy enum value set" in msg or \
           ("Value '" in msg and "failed to satisfy constraint" in msg):
            fb = pick_fallback_instance(args.model_id)
            print(f"⚠️ Instance '{instance_type}' not allowed here. Retrying with fallback '{fb}' …")
            predictor = model.deploy(
                initial_instance_count=1,
                instance_type=fb,
                endpoint_name=endpoint_name,
                accept_eula=args.accept_eula,
            )
        else:
            raise

    print("✅ Endpoint InService")

    # Invoke once
    print("▶️ Invoking …")
    resp = rt.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=args.content_type,
        Accept=args.accept,
        Body=args.payload.encode("utf-8")
    )
    out = resp["Body"].read()
    try:
        print(out.decode("utf-8"))
    except UnicodeDecodeError:
        sys.stdout.buffer.write(out)

    # Cleanup
    print("🧹 Cleanup …")
    cleanup_all(sm, endpoint_name)

    # Verify deletion
    print("🔍 Verifying deletion …")
    ok = verify_deleted(sm, rt, endpoint_name, strict=args.strict_verify)

    print("✅ Done" if ok else "⚠️ Done with warnings")
    if args.strict_verify and not ok:
        sys.exit(2)

if __name__ == "__main__":
    main()
