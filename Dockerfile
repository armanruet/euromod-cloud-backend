# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_cloud.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_cloud.txt

# Copy application code
COPY script_executor_cloud.py .

# Create directory for EUROMOD data
RUN mkdir -p /app/euromod_data/Input

# Copy EUROMOD data files (you'll need to add these)
# COPY euromod_data/ /app/euromod_data/

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONPATH=/app
ENV EUROMOD_MODEL_PATH=/app/euromod_data
ENV EUROMOD_DATA_PATH=/app/euromod_data/Input/LU_training_data.txt
ENV PYTHON_EXECUTABLE=python

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["python", "script_executor_cloud.py"] 