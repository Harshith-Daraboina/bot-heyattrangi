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

# Create directories for data and cache
RUN mkdir -p vector_store data .nicegui .cache

# Set permissions (Hugging Face runs as user 1000 but sometimes overrides group)
# We set 777 to ensure the app can write regardless of the user ID running it
RUN chmod -R 777 vector_store data .nicegui .cache

# Create user but we've already opened up permissions
RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Expose the plotting port
EXPOSE 7860

# Run the app
CMD ["python", "main.py"]
