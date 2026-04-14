# Use an optimized Python slim image for edge-compatible deployment
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (e.g., for OpenCV and OR-Tools)
# CHANGED: libgl1-mesa-glx -> libgl1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and deployment scripts into the container
COPY src/ ./src/
COPY deploy/ ./deploy/

# Expose the port the FastAPI app runs on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "deploy.main:app", "--host", "0.0.0.0", "--port", "8000"]