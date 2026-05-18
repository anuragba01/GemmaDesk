# Use an official Python slim image as the base
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for ffmpeg, audio processing, and LiteRT C++ bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    libvulkan1 \
    libegl1 \
    libgles2 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
# Note: We filter out pywebview and PyQt6 dependencies since the GUI is run on the host system,
# while the Docker container solely runs the Streamlit server.
RUN grep -vE "pywebview|PyQt6|qtpy" requirements.txt > req_docker.txt && \
    pip install --no-cache-dir -r req_docker.txt

# Copy the rest of the application files
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
