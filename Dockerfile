FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Build the vector index from data/ (also downloads the embedding model,
# so the first request doesn't have to)
RUN python ingest.py

# Expose port
EXPOSE 10000

# Run the application
CMD ["python", "app.py"]
