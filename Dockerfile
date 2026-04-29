# AI4Animation Studio — Docker image
# ====================================
# Build:   docker build -t ai4animation-studio .
# Run:     docker run -p 7860:7860 ai4animation-studio
# Or use:  docker compose up

FROM python:3.12-slim

LABEL maintainer="AI4AnimationPy" \
      description="AI4Animation Studio — browser-based animation AI pipeline"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source for layer caching
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy framework source
COPY ai4animation/ ./ai4animation/
COPY setup.py .

# Install the framework in editable mode (no raylib needed for web mode)
RUN pip install --no-cache-dir -e . --no-deps

# Copy the web app
COPY webapp/ ./webapp/

# Expose the Gradio port
EXPOSE 7860

# Persistent project storage
VOLUME ["/root/.ai4animation"]

# Environment
ENV PORT=7860
ENV GRADIO_SHARE=0
ENV PYTHONUNBUFFERED=1

CMD ["python", "webapp/app.py"]
