FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create directory for vector store if it doesn't exist
RUN mkdir -p vector_store data .nicegui

# Set permissions (Hugging Face runs as user 1000)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app && \
    chmod -R 755 /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Expose the plotting port
EXPOSE 7860

# Run the app
CMD ["python", "main.py"]
