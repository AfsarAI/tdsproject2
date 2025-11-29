import requests
import json

# Configuration
URL = "http://localhost:5000/quiz-webhook"
PAYLOAD = {
    "email": "23f2002023@ds.study.iitm.ac.in",
    "secret": "224149",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
}

print(f"Sending POST request to {URL}...")
print(f"Payload: {json.dumps(PAYLOAD, indent=2)}")

try:
    response = requests.post(URL, json=PAYLOAD, timeout=30)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("\n✅ Success! Check your MAIN terminal (where app.py is running) to see the Agent logs.")
    else:
        print("\n❌ Failed. Check server logs.")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Make sure 'python3 app.py' is running in another terminal!")
