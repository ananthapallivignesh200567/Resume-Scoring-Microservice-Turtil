FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything correctly
COPY app /app/app
COPY config.json /app/config.json
COPY schema.json /app/schema.json
COPY data /app/data

# ✅ Ensure it's a package
RUN touch /app/app/__init__.py

# Expose FastAPI port
EXPOSE 8000

# ✅ Start from app.main, not just main
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
