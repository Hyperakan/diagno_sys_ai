import requests
import json
import sys

# Configure Ollama server endpoint
HOST = "http://localhost:11434"
MODEL_NAME = "llama3.1:1b"

def check_version():
    """Check Ollama server version."""
    print("=== Checking Ollama server version... ===")
    try:
        resp = requests.get(f"{HOST}/api/version")
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except requests.RequestException as e:
        print(f"[Error] Checking version failed: {e}", file=sys.stderr)
        sys.exit(1)

def pull_model():
    """Pull the Llama 3.1 1b model if not already present."""
    print(f"\n=== Pulling model '{MODEL_NAME}' if needed... ===")
    try:
        with requests.post(
            f"{HOST}/api/pull",
            json={"model": MODEL_NAME},
            stream=True,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line.strip():
                    print(line)
    except requests.RequestException as e:
        print(f"[Error] Pulling model {MODEL_NAME} failed: {e}", file=sys.stderr)
        sys.exit(1)

def send_prompt(prompt):
    """
    Send a prompt to the model and return the response.
    
    Args:
        prompt (str): The prompt to send to the model
        
    Returns:
        str: The model's response
    """
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        
        resp = requests.post(f"{HOST}/api/generate", json=payload)
        resp.raise_for_status()
        response_data = resp.json()
        
        return response_data['response']
        
    except requests.RequestException as e:
        print(f"[Error] Generate request failed: {e}", file=sys.stderr)
        return None


def main():
    check_version()
    pull_model()
    
    # Example usage
    prompt = "What is the capital of France?"
    response = send_prompt(prompt)
    if response:
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")

if __name__ == "__main__":
    main() 