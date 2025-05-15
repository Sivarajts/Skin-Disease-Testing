FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE ${PORT:-10000}

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-10000}/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=${PORT:-10000}", "--server.address=0.0.0.0"] 