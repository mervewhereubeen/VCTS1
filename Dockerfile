FROM python:3.11-slim

# OpenCV ve benzeri paketler için küçük sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python bağımlılıkları
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY . .

# Render (veya benzeri) platformlar PORT env veriyor; 10000'i sabitliyoruz
ENV PORT=10000
EXPOSE 10000

# Uygulamayı başlat
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
