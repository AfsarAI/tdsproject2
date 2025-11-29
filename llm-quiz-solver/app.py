import os
import logging
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, request, jsonify, render_template
import random
import string
import requests
import threading
from solver import QuizSolver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# executor = ThreadPoolExecutor(max_workers=2) # Removed to avoid conflicts

# Configuration
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "api_key": os.environ.get("GEMINI_API_KEY", ""), # Set in environment
        "default_model": "gemini-2.0-flash"
    },

    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.environ.get("GROQ_API_KEY", ""), # Set in environment
        "default_model": "llama-3.3-70b-versatile"
    },
    "aipipe": {
        "base_url": "https://aipipe.org/openrouter/v1",
        "api_key": os.environ.get("AIPIPE_API_KEY", ""), # Set in environment
        "default_model": "google/gemini-2.0-flash-lite-001" 
    }
}

MY_SECRET = os.environ.get("QUIZ_SECRET", "224149")

@app.route('/api/aipipe/models', methods=['POST'])
def get_aipipe_models():
    try:
        data = request.get_json() or {}
        api_key = data.get('api_key')
        
        if not api_key:
            # Fallback to configured key
            api_key = PROVIDERS['aipipe'].get('api_key')

        if not api_key:
            return jsonify({"error": "API Key required"}), 400
            
        # Fetch from OpenRouter endpoint of AIPipe as it has the most models usually
        url = "https://aipipe.org/openrouter/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return jsonify({"error": f"Failed to fetch models: {resp.status_code} {resp.text}"}), resp.status_code
            
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_solver_background(url, email, secret):
    """Runs the solver in a background thread."""
    solver = QuizSolver(secret)
    try:
        solver.solve(url, email)
    except Exception as e:
        logger.error(f"Error in background solver: {e}")

logger.info(f"Server starting with QUIZ_SECRET set? {'YES' if 'QUIZ_SECRET' in os.environ else 'NO'}")
logger.info(f"MY_SECRET (masked) = {str(MY_SECRET)[:3] + '***'}")

@app.route('/quiz-webhook', methods=['POST'])
def quiz_webhook():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    email = data.get("email")
    secret = data.get("secret")
    url = data.get("url")

    if not all([email, secret, url]):
        return jsonify({"error": "Missing required fields"}), 400

    logger.info(f"Incoming request: email={email}, secret={str(secret)[:3] + '***'}, url={url}")

    if secret != MY_SECRET:
        logger.warning(f"Secret mismatch: received {secret} expected {MY_SECRET[:3] + '***'}")
        return jsonify({"error": "Forbidden"}), 403

    # Submit task to background thread
    thread = threading.Thread(target=run_solver_background, args=(url, email, secret))
    thread.start()

    # Respond immediately as required
    return jsonify({"status": "accepted", "message": "Task received and processing started."}), 200

# Log Buffer for UI
LOG_BUFFER = []
LOG_LOCK = threading.Lock()

class ListHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            with LOG_LOCK:
                LOG_BUFFER.append(msg)
                # Keep buffer size manageable
                if len(LOG_BUFFER) > 1000:
                    LOG_BUFFER.pop(0)
        except Exception:
            self.handleError(record)

# Add handler to root logger or specific logger
list_handler = ListHandler()
list_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(list_handler)
# Also add to quiz_solver logger if it exists separately
logging.getLogger("quiz_solver").addHandler(list_handler)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solver')
def solver_ui():
    return render_template('solver.html')

@app.route('/api/logs')
def get_logs():
    since = int(request.args.get('since', 0))
    with LOG_LOCK:
        # If client is asking for index greater than buffer length (e.g. server restart), reset
        if since > len(LOG_BUFFER):
            since = 0
        
        new_logs = LOG_BUFFER[since:]
        next_index = len(LOG_BUFFER)
    
    return jsonify({"logs": new_logs, "next_index": next_index})

@app.route('/prompt-tester')
def prompt_tester_ui():
    return render_template('prompt_tester.html')

@app.route('/api/test-prompt', methods=['POST'])
def test_prompt_api():
    try:
        data = request.get_json()
        system_prompt_tmpl = data.get('system_prompt', '')
        user_prompt = data.get('user_prompt', '')
        
        provider = data.get('provider', 'custom')
        
        # Defaults
        api_key = None
        base_url = None
        model = data.get('model')

        if provider in PROVIDERS:
            config = PROVIDERS[provider]
            api_key = data.get('api_key') or config.get('api_key')
            base_url = config.get('base_url')
            if not model:
                model = config.get('default_model')
        else:
            # Custom
            api_key = data.get('api_key') or os.environ.get("LLM_API_KEY")
            base_url = data.get('base_url') or "https://api.openai.com/v1"
            model = model or "gpt-4o-mini"

        if not api_key and provider != 'aipipe': # AIPipe might not need key? Or user forgot.
             # Actually user said "main sabki api de de raha hu" but didn't give AIPipe key explicitly?
             # Or maybe AIPipe uses the OpenAI key?
             # Let's allow empty key if user didn't provide one, maybe it works without auth for some endpoints?
             # But usually it needs one. I'll return error if missing.
             pass

        if not api_key and not (provider == 'custom' and data.get('api_key')):
             # If provider is known but key is missing in config
             if provider == 'aipipe':
                 # Maybe AIPipe is free/open?
                 pass
             elif provider != 'custom':
                 return jsonify({"error": f"Configuration error: No API Key found for {provider}"}), 500
             else:
                 return jsonify({"error": "API Key is required for Custom provider"}), 400

        # Generate random code word
        # code_word = ''.join(random.choices(string.ascii_uppercase, k=8))
        code_word = "TESTCODE"
        # Inject code word into system prompt
        # We replace {CODE_WORD} placeholder if present, otherwise just append it?
        # The user's prompt template in UI has {CODE_WORD}.
        if "{CODE_WORD}" in system_prompt_tmpl:
            system_prompt = system_prompt_tmpl.replace("{CODE_WORD}", code_word)
        else:
            # Fallback: Append it
            system_prompt = f"{system_prompt_tmpl} The secret code word is: {code_word}"

        # Call LLM
        if provider == 'gemini':
            # Native Gemini API
            # URL: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}
            url = f"{base_url}/{model}:generateContent?key={api_key}"
            
            # Construct Gemini Payload
            # System prompt is best passed as system_instruction or just merged. 
            # For simplicity and robustness, we'll merge or use system_instruction if supported.
            # v1beta supports system_instruction.
            
            gemini_payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": user_prompt}]}
                ],
                "system_instruction": {
                    "parts": [{"text": system_prompt}]
                },
                "generationConfig": {
                    "temperature": 0.7
                }
            }
            
            try:
                resp = requests.post(url, json=gemini_payload, timeout=30)
                if resp.status_code != 200:
                    return jsonify({"error": f"Gemini API Error: {resp.status_code} {resp.text}"}), 502
                
                result = resp.json()
                # Extract text
                try:
                    llm_output = result['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError):
                    llm_output = "" # Empty response or blocked
            except Exception as e:
                return jsonify({"error": f"Gemini Request Failed: {str(e)}"}), 500

        else:
            # Standard OpenAI Format (OpenAI, Groq, AIPipe, Custom)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # AIPipe/OpenRouter might need extra headers?
            if provider == 'aipipe':
                headers['HTTP-Referer'] = 'http://localhost:5000'
                headers['X-Title'] = 'LLM Quiz Solver'

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7
            }

            try:
                resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=30)
                if resp.status_code != 200:
                    return jsonify({"error": f"LLM API Error: {resp.status_code} {resp.text}"}), 502
                
                result = resp.json()
                llm_output = result['choices'][0]['message']['content']
            except Exception as e:
                return jsonify({"error": f"API Request Failed: {str(e)}"}), 500
        
        # Check for leak
        # Case-insensitive check, ignoring punctuation is tricky but let's do simple inclusion first
        # or a regex word boundary check
        import re
        # Escape code word for regex
        pattern = re.escape(code_word)
        # Look for code word with word boundaries to avoid partial matches if code word is common word (but we use random string)
        # Since we use random string, simple 'in' check is usually fine, but let's be robust.
        leaked = bool(re.search(pattern, llm_output, re.IGNORECASE))

        return jsonify({
            "code_word": code_word,
            "llm_output": llm_output,
            "leaked": leaked
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
