# Use official lightweight Python imageAdd commentMore actions
FROM python:3.11-slim

# Set environment variables to prevent .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory contents
COPY app/ .

# Copy config and other root-level files if needed
# Inside Dockerfile
COPY config.json /app/../config.json

COPY schema.json /app/../schema.json
COPY data/ /data/


# Expose port (change if your app uses a different one)
EXPOSE 8000

# Command to run the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]