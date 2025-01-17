# Base image olarak Python'ı kullanıyoruz
FROM python:3.9-slim

# Çalışma dizinini belirliyoruz
WORKDIR /app

# Gerekli Python bağımlılıklarını kopyalıyoruz ve yükliyoruz
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyasını kopyalıyoruz
COPY . /app/

# Flask uygulamasını başlatıyoruz
CMD ["python", "app.py"]

