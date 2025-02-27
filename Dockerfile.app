# Temel PyTorch imajını kullan
FROM pytorch/pytorch:latest

# Çalışma dizinini oluştur
WORKDIR /app

# Gerekli bağımlılıkları yükle
COPY app/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY app/ .

# Uygulamayı başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8501", "--reload"]
