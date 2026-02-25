# Inference Containerfile 
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Working directory for runtime and relative imports
WORKDIR /home/aisg/mnist

# Copy dependency manifests first for better caching
COPY pyproject.toml uv.lock ./

# Install ONLY backend dependencies into a venv
# --no-install-project keeps it simple: we run via PYTHONPATH instead of packaging.
RUN uv sync --frozen --no-install-project --group backend

# Ensure the venv binaries are found
ENV PATH="/home/aisg/mnist/.venv/bin:$PATH"

# Copy only what FastAPI service needs at runtime
# - src/mnist is contains datasets.py, model defs, utils
# - src/mnist_fastapi is API app package from P4
# - conf for configs/logging 
COPY conf/ ./conf/
COPY src/mnist/ ./src/mnist/
COPY src/mnist_fastapi/ ./src/mnist_fastapi/

# Make imports work without installing the project as a package
ENV PYTHONPATH=/home/aisg/mnist/src

# Expose Cloud Run port
EXPOSE 8080

# Run as non-root 
# Ensure files under /home/aisg/mnist are readable by this user.
RUN adduser --disabled-password --gecos '' aisg && \
    chown -R aisg:aisg /home/aisg/mnist
USER aisg

# Start FastAPI via uvicorn
CMD ["uvicorn", "mnist_fastapi.main:APP", "--host", "0.0.0.0", "--port", "8080"]