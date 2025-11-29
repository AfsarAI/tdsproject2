# Deployment Guide (Render.com)

This application is configured for easy deployment on [Render](https://render.com).

## Steps to Deploy

1.  **Sign up/Log in** to [Render.com](https://dashboard.render.com/).
2.  Click **"New +"** and select **"Web Service"**.
3.  Connect your GitHub account and select the **`tdsproject2`** repository.
4.  **IMPORTANT**: In the "Root Directory" field, enter: `llm-quiz-solver`.
5.  Render will automatically detect the `render.yaml` file (or you can configure manually).
5.  **Environment Variables**:
    You need to add your secrets in the Render Dashboard (under "Environment"):
    *   `GEMINI_API_KEY`: Your Gemini API Key.
    *   `GROQ_API_KEY`: Your Groq API Key.
    *   `AIPIPE_API_KEY`: Your AIPipe API Key.
    *   `QUIZ_SECRET`: A secret password for your webhook (e.g., `224149` or a random string).

6.  Click **"Create Web Service"**.

## Manual Configuration (if needed)

*   **Runtime**: Python 3
*   **Build Command**: `pip install -r requirements.txt && playwright install chromium`
*   **Start Command**: `gunicorn app:app`
