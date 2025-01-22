import requests
import json

# FastAPI sunucusunun URL'si
url = "http://localhost:8501/ask"  # Eğer FastAPI'yi farklı bir portta çalıştırıyorsanız URL'yi değiştirin

# Soru
query = "What are the symptoms of a cold?"

# JSON veri
data = {
    "query": query
}

# API'ye POST isteği gönderme
response = requests.post(url, json=data)

# Cevabı yazdırma
if response.status_code == 200:
    result = response.json()
    print("Soru: ", result['query'])
    print("Cevap: ", result['response'])
else:
    print(f"Error {response.status_code}: {response.text}")
