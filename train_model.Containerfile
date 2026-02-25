# Training Containerfile 
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Ensure Python logs print immediately for training instance
ENV PYTHONUNBUFFERED=1

# Set the runtime working directory expected by CI and by relative paths like ./data/...
WORKDIR /home/aisg/mnist

# Copy only dependency manifests first for better layer caching.
COPY pyproject.toml uv.lock ./

# Create the virtualenv and install ONLY the training dependency group.
# --no-install-project run via PYTHONPATH instead of packaging.
RUN uv sync --frozen --no-install-project --group training

# Ensure the venv binaries are found first.
ENV PATH="/home/aisg/mnist/.venv/bin:$PATH"

# Copy only what training needs at runtime.
COPY conf/ ./conf/
COPY src/mnist/ ./src/mnist/
COPY src/train_model.py ./src/train_model.py

# Make imports work without installing the project as a package.
ENV PYTHONPATH=/home/aisg/mnist/src

# Run as non-root. Ensure that workdir is writable so for CI command
# can create ./data symlink 
RUN adduser --disabled-password --gecos '' aisg && \
    chown -R aisg:aisg /home/aisg/mnist
USER aisg

# Default command. CI/Cloud Run can override this with its own command.
CMD ["python", "-u", "src/train_model.py"]