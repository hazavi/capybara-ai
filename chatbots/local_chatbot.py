"""
Local AI Chatbot - Terminal Interface
Change MODEL_NAME below to switch models ⬅️
"""
from gpt4all import GPT4All

# ===== CONFIGURATION ⬅️ CHANGE SETTINGS HERE =====
# Find other models in README.md under "Available Models"
MODEL_NAME = "orca-mini-3b-gguf2-q4_0.gguf"  # (1.9GB, fast)

MAX_TOKENS = 200       # Response length
TEMPERATURE = 0.7      # Creativity (0.1-1.0)
# ==================================================

# Load model
print(f"🤖 Loading {MODEL_NAME}...")
try:
    model = GPT4All(MODEL_NAME)
    print("✓ Ready!\n")
except Exception as e:
    print(f"❌ Error: {e}\nTip: Check internet on first run")
    exit(1)

def chat():
    print("="*60)
    print("   LOCAL AI CHATBOT")
    print("="*60)
    print("Commands: 'exit' to quit, 'clear' to reset\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            if prompt.lower() == 'clear':
                print("🔄 Conversation cleared!\n")
                continue
            
            with model.chat_session():
                print("AI: ", end="", flush=True)
                response = model.generate(prompt, max_tokens=MAX_TOKENS, temp=TEMPERATURE)
                print(response + "\n")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    chat()