FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY ui ./ui
COPY data ./data

EXPOSE 8000 8501

# Overridden per-service in docker-compose.yml (api vs ui).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
