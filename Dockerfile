FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE ${PORT:-10000}

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-10000}/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=${PORT:-10000}", "--server.address=0.0.0.0"] 