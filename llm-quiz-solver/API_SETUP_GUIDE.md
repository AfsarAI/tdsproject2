# How to Setup the Prompt Tester

To use the Prompt Tester, you need an LLM API Key. Here are the best options:

## Option 1: Groq (Free & Fast)
Groq offers free API access to models like Llama 3 and Mixtral. This is the best option for testing without paying.

1.  Go to **[console.groq.com](https://console.groq.com/keys)**.
2.  Sign up or Log in.
3.  Click **"Create API Key"**.
4.  Copy the key (starts with `gsk_...`).
5.  In the Prompt Tester UI:
    *   **API Key**: Paste your `gsk_...` key.
    *   **Model**: `llama-3.3-70b-versatile` (or `llama-3.1-8b-instant`)
    *   **Base URL**: `https://api.groq.com/openai/v1`

## Option 2: OpenAI (Standard)
If you have an OpenAI account with credits:

1.  Go to **[platform.openai.com/api-keys](https://platform.openai.com/api-keys)**.
2.  Create a new secret key.
3.  In the Prompt Tester UI:
    *   **API Key**: Paste your `sk-...` key.
    *   **Model**: `gpt-4o-mini` (default)
    *   **Base URL**: Leave blank (defaults to OpenAI).

## Option 3: Local LLM (LM Studio)
If you want to run offline (requires a good GPU):

1.  Download **[LM Studio](https://lmstudio.ai/)**.
2.  Load a model (e.g., Llama 3).
3.  Start the **Local Server** in LM Studio.
4.  In the Prompt Tester UI:
    *   **API Key**: `lm-studio` (any value works).
    *   **Model**: `local-model` (any value works).
    *   **Base URL**: `http://localhost:1234/v1`

---

## Running the App
Since you are using a virtual environment, run the app with:

```bash
./venv/bin/python app.py
```

Then open your browser to: **http://localhost:5000/prompt-tester**
