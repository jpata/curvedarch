# Use Python 3.10 slim image (matching project's local/specified version)
FROM python:3.10-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    # VTK/OCP off-screen rendering
    VTK_DEFAULT_OPENGL_WINDOW=EGL \
    VTK_EGL_DEVICE_INDEX=0

WORKDIR /app

# Install system dependencies
# Combined essential build tools, IPOPT, and visualization/OCP requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    curl \
    git \
    # Optimization (TNA solver)
    coinor-libipopt-dev \
    # CadQuery / OCP / Visualization
    libgl1-mesa-glx \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libice6 \
    libegl1 \
    libgbm1 \
    libosmesa6 \
    xvfb \
    libfontconfig1 \
    libx11-6 \
    libxcursor1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (UID 1000 is standard for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose the port Streamlit will run on
EXPOSE 7860

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
