FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed (e.g., for build tools)
# RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create directory for vector store if it doesn't exist
RUN mkdir -p vector_store

# Set permissions (Hugging Face runs as user 1000)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Expose the plotting port
EXPOSE 7860

# Run the app
CMD ["python", "main.py"]
