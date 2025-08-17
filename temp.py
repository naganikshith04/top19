import os

def mask(v): 
    return v[:4] + "…" + v[-3:] if v and len(v) > 10 else v

print("OPENAI_API_KEY:", mask(os.getenv("OPENAI_API_KEY")))
print("TOGETHER_API_KEY:", mask(os.getenv("TOGETHER_API_KEY")))
print("GOOGLE_API_KEY:", mask(os.getenv("GOOGLE_API_KEY")))

# --- OpenAI ---
try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                        organization=os.getenv("OPENAI_ORG_ID"),
                        project=os.getenv("OPENAI_PROJECT"))
        resp = client.models.list()
        print("[OpenAI] OK, models:", len(resp.data))
    else:
        print("[OpenAI] key not set")
except Exception as e:
    print("[OpenAI] ERROR:", e)

# --- Together ---
try:
    from together import Together
    if os.getenv("TOGETHER_API_KEY"):
        t = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        models = t.models.list()
        print("[Together] OK, models:", len(models))
    else:
        print("[Together] key not set")
except Exception as e:
    print("[Together] ERROR:", e)

# --- Gemini ---
try:
    import google.generativeai as genai
    if os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        models = list(genai.list_models())
        print("[Gemini] OK, models:", len(models))
    else:
        print("[Gemini] key not set")
except Exception as e:
    print("[Gemini] ERROR:", e)
