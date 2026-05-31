FROM m.daocloud.io/docker.io/library/python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 先装 numpy 1.26.4（Python 3.9 兼容的最新 1.x 版本）
RUN pip install --no-cache-dir numpy==1.26.4

# 再装其他依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py .
COPY app.py .
COPY recommendation_pipeline.pkl .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]