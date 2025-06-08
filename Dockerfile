# ------------------------------------------------------------
# Streamlit RAG demo (CPU build, Python 3.12 slim)
# ------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --- system deps (faiss needs libgomp) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy only runtime code/assets
COPY data ./data
COPY main.py .

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]
