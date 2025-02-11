#!/usr/bin/env python3

import json
import requests
import sys
import pandas as pd

# Configure your Ollama server endpoint
HOST = "http://localhost:11434"
DATASET_PATH = "/Users/hakandogan/bitirme/test_pipeline/MedQA-USMLE-Partial.csv"
#DATASET_PATH = '/Users/hakandogan/bitirme/test_pipeline/tus_dataset_partial.csv'
def check_version():
    """Check Ollama server version."""
    print("=== Checking Ollama server version... ===")
    try:
        resp = requests.get(f"{HOST}/api/version")
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except requests.RequestException as e:
        print(f"[Error] Checking version failed: {e}", file=sys.stderr)

def list_local_models():
    """List all local models available on the Ollama server."""
    print("\n=== Listing local models... ===")
    try:
        resp = requests.get(f"{HOST}/api/tags")
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except requests.RequestException as e:
        print(f"[Error] Listing models failed: {e}", file=sys.stderr)

def pull_model(model_name):
    """
    Pull (download) a model from the Ollama library.
    If the model is already present, the server will quickly verify or skip.
    """
    print(f"\n=== Pulling model '{model_name}' if needed... ===")
    try:
        # We’ll stream the JSON responses for demonstration
        # If you prefer a single final response, set "stream": false in the JSON body.
        with requests.post(
            f"{HOST}/api/pull",
            json={"model": model_name},  # stream defaults to True
            stream=True,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line.strip():
                    # Each line is a JSON object in string form
                    print(line)
    except requests.RequestException as e:
        print(f"[Error] Pulling model {model_name} failed: {e}", file=sys.stderr)

def test_generate(model_name, prompt="Explain briefly why the sky is blue."):
    """Send a quick test prompt to /api/generate and print the response."""
    print(f"\n=== Testing /api/generate with '{model_name}' ===")
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # retrieve the entire response at once
    }
    try:
        resp = requests.post(
            f"{HOST}/api/generate",
            json=payload
        )
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except requests.RequestException as e:
        print(f"[Error] Generate request failed: {e}", file=sys.stderr)

def prompt_design_tus(prompt):
    # Design a prompt for the model
    prompt = f"""Aşağıdaki soruyu derinlemesine düşünerek cevaplayın ve son seçeneği (A,B,C veya D) 
    aşağıdaki formatta (<<<seçenek>>) döndürün: <<<A>>>\n\nSoru: {prompt}"""
    return prompt

def prompt_design_usmle(prompt):
    # Design a prompt for the model
    prompt = f"""Think deeply about the following question and return the last option (A, B, C, or D)
    in the following format (<<<option>>>) : <<<A>>>\n\nQuestion: {prompt}"""
    return prompt

def test_answer(model_name, prompt="Explain briefly why the sky is blue.", exam_type="tus"):
    if exam_type == "tus":
        prompt = prompt_design_tus(prompt)
    elif exam_type == "usmle":
        prompt = prompt_design_usmle(prompt)
    """Send a quick test prompt to /api/generate and print the response."""
    print(f"\n=== Testing /api/generate with '{model_name}' ===")
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # retrieve the entire response at once
    }
    try:
        resp = requests.post(f"{HOST}/api/generate", json=payload)
        resp.raise_for_status()
        response_data = resp.json()
        # Extract the generated answer
        #generated_answer = response_data.get("choices", [{}])[0].get("text", "").strip()
        generated_answer = response_data['response']
        print(f"Generated answer: {generated_answer}")
        return generated_answer
    except requests.RequestException as e:
        print(f"[Error] Generate request failed: {e}", file=sys.stderr)
        return ""

def test_chat(model_name):
    """
    Send a conversation with one user message:
    “What is the difference between CPU and GPU?”
    to /api/chat and print the response.
    """
    print(f"\n=== Testing /api/chat with '{model_name}' ===")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "What is the difference between CPU and GPU?"}
        ],
        "stream": False  # retrieve the entire chat response at once
    }
    try:
        resp = requests.post(
            f"{HOST}/api/chat",
            json=payload
        )
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except requests.RequestException as e:
        print(f"[Error] Chat request failed: {e}", file=sys.stderr)



def extract_option_from_llm_answer(df):
    def safe_extract_option(answer):
        # Check if both delimiters are present
        if '<<<' in answer and '>>>' in answer:
            try:
                # Attempt to extract text between delimiters
                return answer.split('<<<')[1].split('>>>')[0].strip()
            except IndexError:
                return "cannot extract"
        return "cannot extract"
    df['extracted_option'] = df['llm_answer'].apply(safe_extract_option)
    return df

def calculate_accuracy_tus(df):
    # Calculate the accuracy
    correct_count = 0
    for index, row in df.iterrows():
        if row['correct_option_name'] == row['extracted_option']:
            correct_count += 1

    accuracy = correct_count / len(df) * 100
    print(f"Accuracy: {accuracy:.2f}%")

def calculate_accuracy_usmle(df):
    # Calculate the accuracy
    correct_count = 0
    for index, row in df.iterrows():
        if row['answer_idx'] == row['extracted_option']:
            correct_count += 1

    accuracy = correct_count / len(df) * 100
    print(f"Accuracy: {accuracy:.2f}%")
def test_tus_dataset(model_name):
    df = pd.read_csv('/Users/hakandogan/bitirme/test_pipeline/tus_dataset_partial.csv')

    # Then you can continue with applying the test_answer
    df['llm_answer'] = df.apply(lambda row: test_answer(model_name, f"""{row['Questions']} \n\n Şıklar: {'Options'}\nCevap: ?"""), axis=1)
    df = extract_option_from_llm_answer(df)
    # Save the results to a new CSV file
    df.to_csv('/Users/hakandogan/bitirme/test_pipeline/tus_dataset_with_llm_answers.csv', index=False)
    
    
def test_usmle_dataset(model_name):
    df = pd.read_csv(DATASET_PATH)
    print("DataFrame columns:", df.columns.tolist())  # Debug line to verify headers
    
    # Use the column name that exists in your CSV file.
    df['llm_answer'] = df.apply(
    lambda row: test_answer(
        model_name,
        f"Question: {row['question']}\nOptions: {row['options']}\nAnswer: ?"
    ),
    axis=1
)

    df = extract_option_from_llm_answer(df)
    df.to_csv('/Users/hakandogan/bitirme/test_pipeline/MedQA-USMLE-Partial-with-llm-answers.csv', index=False)



def main():
    #check_version()
    #list_local_models()
    # Change 'llama3.2' to your desired model if different
    #MODEL_NAME = "llama3.2:1b"
    MODEL_NAME = "deepseek-r1:7b"
    pull_model(MODEL_NAME)
    #test_generate(MODEL_NAME)
    #test_answer(MODEL_NAME)
    #test_tus_dataset(MODEL_NAME)
    #df = pd.read_csv('/Users/hakandogan/bitirme/test_pipeline/MedQA-USMLE-Partial-with-llm-answers.csv')
    #calculate_accuracy_tus(df)
    test_usmle_dataset(MODEL_NAME)
    df = pd.read_csv('/Users/hakandogan/bitirme/test_pipeline/MedQA-USMLE-Partial-with-llm-answers.csv')
    calculate_accuracy_usmle(df)
    #test_chat(MODEL_NAME)

    print("\n=== Test pipeline complete. ===")

if __name__ == "__main__":
    main()
