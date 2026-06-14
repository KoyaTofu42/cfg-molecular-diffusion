# Base image from NVIDIA NGC with PyTorch and CUDA
# Using 26.02-py3 as requested (last version with CUDA 13.1)
FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /workspace

# Install uv by copying it from the official Astral image (Best Practice)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Enable bytecode compilation for performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Create a virtual environment using the Nvidia container's system Python
RUN uv venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the entire project
COPY . /workspace

# Install the project as an editable package using uv cache mount
WORKDIR /workspace/e3_diffusion_for_molecules
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e .

# Return to root workspace
WORKDIR /workspace

# Default command to run the new unified main.py
CMD ["python", "main.py"]
