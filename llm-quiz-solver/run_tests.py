# run_tests.py — improved version
import requests
import time
import threading
import os
import subprocess
import sys
import signal

# Configuration
SECRET = "224149"
PORT = 5002
URL = f"http://localhost:{PORT}/quiz-webhook"
DEMO_QUIZ_URL = "https://tds-llm-analysis.s-anand.net/demo"
SERVER_SCRIPT = "app.py"  # change if your server filename differs
STARTUP_TIMEOUT = 10      # seconds to wait for server to be ready
POST_TIMEOUT = 30         # seconds for POST to submit

def _drain_stdout(proc):
    """Read subprocess stdout/stderr and print live prefixed lines."""
    try:
        for line in iter(proc.stdout.readline, ''):
            if line:
                print("[server]", line.rstrip())
            else:
                break
    except Exception as e:
        print("[runner] Error while reading server output:", e)

def run_server():
    """Runs the Flask app in a subprocess and wires its stdout/stderr to our console."""
    env = os.environ.copy()
    env["QUIZ_SECRET"] = SECRET
    env["PORT"] = str(PORT)
    
    # Prefer venv python if available
    venv_python = os.path.join(os.getcwd(), "venv", "bin", "python3")
    if os.path.exists(venv_python):
        python_exe = venv_python
    else:
        python_exe = sys.executable
        
    cmd = [python_exe, SERVER_SCRIPT]

    print(f"[runner] Starting server: {' '.join(cmd)} (PORT={PORT})")
    # Capture stdout/stderr and print them live
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Start a background thread to print server output live
    t = threading.Thread(target=_drain_stdout, args=(proc,), daemon=True)
    t.start()
    return proc

def wait_for_server(port, timeout=STARTUP_TIMEOUT):
    """Poll localhost:port until it responds (or timeout). Returns True if ready."""
    base = f"http://localhost:{port}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            # If your app defines a root endpoint (/) this will be quickest.
            r = requests.get(base, timeout=1)
            # If we get any response (200/404/whatever), consider server up.
            print(f"[runner] Server responded to GET {base} with status {r.status_code}")
            return True
        except requests.exceptions.ConnectionError:
            # Not up yet
            time.sleep(0.5)
            continue
        except Exception as e:
            # Some other network error — but server may still be starting
            print(f"[runner] Waiting for server (got error): {e}")
            time.sleep(0.5)
    return False

def test_endpoint():
    payload = {
        "email": "23f2002023@ds.study.iitm.ac.in",
        "secret": SECRET,
        "url": DEMO_QUIZ_URL
    }

    print(f"[runner] Sending POST request to {URL} ...")
    try:
        response = requests.post(URL, json=payload, timeout=POST_TIMEOUT)
        print(f"[runner] Response Status: {response.status_code}")
        # Try to pretty-print JSON body safely
        try:
            print(f"[runner] Response Body: {response.json()}")
        except Exception:
            print(f"[runner] Response Text: {response.text}")

        if response.status_code == 200:
            print("[runner] SUCCESS: Endpoint accepted the task.")
        elif response.status_code == 403:
            print("[runner] FAILURE: Forbidden (secret mismatch).")
        elif response.status_code == 400:
            print("[runner] FAILURE: Bad request (invalid JSON or missing fields).")
        else:
            print("[runner] FAILURE: Endpoint returned an unexpected status.")
    except Exception as e:
        print(f"[runner] Error sending request: {e}")

if __name__ == "__main__":
    server_proc = None
    try:
        server_proc = run_server()

        print("[runner] Waiting for server to become ready...")
        ready = wait_for_server(PORT, timeout=STARTUP_TIMEOUT)
        if not ready:
            print(f"[runner] Server not ready after {STARTUP_TIMEOUT}s. Check server logs above.")
            # If server not ready, let user inspect logs briefly then exit.
            time.sleep(2)
        else:
            # Run the actual test
            test_endpoint()
            # Keep server running a bit to allow background worker to complete (if any)
            keep_seconds = 30
            print(f"[runner] Keeping server running for {keep_seconds} seconds to allow background tasks...")
            time.sleep(keep_seconds)

    except KeyboardInterrupt:
        print("[runner] Interrupted by user.")
    finally:
        if server_proc:
            print("[runner] Stopping server...")
            try:
                # Try graceful terminate
                server_proc.send_signal(signal.SIGINT)
                time.sleep(1)
                if server_proc.poll() is None:
                    server_proc.terminate()
                    time.sleep(1)
                if server_proc.poll() is None:
                    server_proc.kill()
            except Exception as e:
                print("[runner] Error while terminating server:", e)
            server_proc.wait(timeout=5)
        print("[runner] Done.")