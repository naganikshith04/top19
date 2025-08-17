from orchestrator import MultiProviderOrchestrator, CatalogEntry
import json

def main():
    ORCH = MultiProviderOrchestrator(
        region_bedrock="us-east-1",
        region_jumpstart="us-east-2",
        jumpstart_exec_role_arn="your-jumpstart-role-arn",
        default_js_instance="ml.g5.2xlarge",
        max_new_tokens=128,
    )

    ORCH.init_clients()
    with open("top10_models.json") as f:
        data = json.load(f)
    model_list = data.get("models", [])

    # Get available models from providers
    openai_models = set(m.lower() for m in ORCH.openai.list_models()) if ORCH.openai.available() else set()
    together_models = set(m.lower() for m in ORCH.together.list_models()) if ORCH.together.available() else set()
    gemini_models = set(m.lower() for m in ORCH.gemini.list_models()) if ORCH.gemini.available() else set()
    bedrock_models = set(m.lower() for m in ORCH.bedrock.list_models())

    catalogs = []
    for orig_id in model_list:
        model_id = orig_id.lower()
        if model_id in openai_models:
            catalogs.append(CatalogEntry(model_id, "openai", orig_id, {}))
        elif model_id in together_models:
            catalogs.append(CatalogEntry(model_id, "together", orig_id, {}))
        elif model_id in gemini_models:
            catalogs.append(CatalogEntry(model_id, "gemini", orig_id, {}))
        elif model_id in bedrock_models:
            resolved_id = ORCH.bedrock.resolve_model_id(orig_id)
            catalogs.append(CatalogEntry(model_id, "bedrock", resolved_id or orig_id, {}))
        else:
            print(f"⚠️ Model '{orig_id}' not found in any provider, skipping.")
            continue
    
    queries = [
        "In one sentence, explain why caching improves performance.",
        "Give three concise bullet points on vector databases.",
    ]

    for entry in catalogs:
        for query in queries:
            print(f"\n[{entry.provider}] {entry.provider_model_id} -> {query[:48]}...")
            rec = ORCH.run_entry(entry, query)
            if rec.ok:
                print(f"✅ OK [{rec.response_time:.2f}s]: {rec.response_preview}")
            else:
                print(f"❌ FAIL: {rec.error}")

if __name__ == "__main__":
    main()
