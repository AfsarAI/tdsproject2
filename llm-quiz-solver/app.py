import os
import logging
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from solver import QuizSolver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=2)

# Configuration
MY_SECRET = os.environ.get("QUIZ_SECRET", "default-secret")

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

    # Submit task to background executor
    executor.submit(run_solver_background, url, email, secret)

    # Respond immediately as required
    return jsonify({"status": "accepted", "message": "Task received and processing started."}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
