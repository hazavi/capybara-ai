"""
Local AI ChatGPT API - FastAPI Backend
Change MODEL_NAME below to switch models ⬅️
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gpt4all import GPT4All

# ===== CONFIGURATION ⬅️ CHANGE SETTINGS HERE =====
# Find other models in README.md under "Available Models"
MODEL_NAME = "orca-mini-3b-gguf2-q4_0.gguf"  # (1.9GB, fast)


SERVER_PORT = 8000        # Change if port in use
MAX_TOKENS = 200          # Response length
TEMPERATURE = 0.7         # Creativity (0.1-1.0)
# ==================================================

# Initialize FastAPI
app = FastAPI(title="Local AI ChatGPT API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
                   allow_methods=["*"], allow_headers=["*"])

# Model cache for dynamic loading
model_cache = {}

# Load default AI model
print(f"🤖 Loading default model {MODEL_NAME}...")
try:
    model_cache[MODEL_NAME] = GPT4All(MODEL_NAME)
    print(f"✓ Ready! Server: http://localhost:{SERVER_PORT}\n")
except Exception as e:
    print(f"❌ Error: {e}\nTip: Check internet connection on first run")
    exit(1)

# Data models
class ChatRequest(BaseModel):
    prompt: str
    model_name: str = MODEL_NAME
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE

class ChatResponse(BaseModel):
    response: str
    model: str = MODEL_NAME

# ===== API ENDPOINTS =====

@app.get("/")
def root():
    return {"message": "Local AI ChatGPT API", "model": MODEL_NAME, 
            "endpoints": {"/chat": "POST", "/health": "GET", "/docs": "GET"}}

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt cannot be empty")
    
    # Load model if not cached
    if req.model_name not in model_cache:
        print(f"🤖 Loading new model: {req.model_name}...")
        try:
            model_cache[req.model_name] = GPT4All(req.model_name)
            print(f"✓ Model {req.model_name} loaded successfully")
        except Exception as e:
            raise HTTPException(500, f"Failed to load model: {str(e)}")
    
    model = model_cache[req.model_name]
    response = model.generate(req.prompt, max_tokens=req.max_tokens, 
                            temp=req.temperature, top_k=40, top_p=0.9)
    return ChatResponse(response=response.strip(), model=req.model_name)

# ===== START SERVER =====
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Server: http://localhost:{SERVER_PORT}")
    print(f"📚 Docs: http://localhost:{SERVER_PORT}/docs")
    print("💡 Open index.html in browser for web interface\n")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
