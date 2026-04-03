FROM python:3.12-slim

# Install Rust toolchain for building kos_rust
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build Rust kernel
COPY kos_rust/ kos_rust/
RUN cd kos_rust && maturin develop --release

# Copy application
COPY kos/ kos/
COPY organism_api.py .
COPY static/ static/
COPY test_universal_agi.py test_masking.py ./

# Create cache directory
RUN mkdir -p .cache/organism

EXPOSE 8090

CMD ["uvicorn", "organism_api:app", "--host", "0.0.0.0", "--port", "8090"]
