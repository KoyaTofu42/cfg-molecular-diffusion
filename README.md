# E3 Diffusion for Molecules (Docker Setup)

This guide explains how to run the project using Docker and `docker-compose`, which provides an isolated, CUDA-enabled environment with all dependencies pre-installed.

## Prerequisites

1. **Docker**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine.
2. **NVIDIA GPU Drivers**: Ensure your host machine has NVIDIA drivers installed.
3. **NVIDIA Container Toolkit**: Required to expose the GPU to Docker containers. (If on Windows with WSL2, this is typically included with Docker Desktop).

## Configuration

1. **Create an Environment File**: 
   Create a `.env` file in the root directory to store your Weights & Biases (W&B) API key.
   ```bash
   echo "WANDB_API_KEY=your_api_key_here" > .env
   ```
   *Note: W&B is used for experiment tracking. Make sure you set your API key to log the runs properly.*

## Usage

### 1. Build and Start the Container

To build the Docker image and start the default training script (`main.py`) in the background:
```bash
docker compose up -d --build
```
This mounts your local directory into the container at `/workspace`. Any code changes you make locally on your host machine will be reflected instantly in the container without needing to rebuild the image.

### 2. Viewing Logs

Since the container runs in detached mode (`-d`), you can view the live training output by checking the Docker logs:
```bash
docker compose logs -f
```

### 3. Stopping the Container

To stop the running container:
```bash
docker compose down
```

### 4. Resuming a Crashed Run

If a training run was interrupted, you can resume it by passing the `--resume` flag and your W&B `run_id`:
```bash
docker compose run -d --rm diffusion python main.py --resume outputs/<your_output_dir> --wandb_run_id <run_id>
```
*Replace `<your_output_dir>` with the specific output folder (e.g., `cfg_multi_property`) and `<run_id>` with your W&B run ID.*

### 5. Running Custom Commands

You can execute arbitrary commands (like running evaluation scripts or opening a shell) inside the isolated container environment:

**Run a specific script:**
```bash
docker compose run --rm diffusion python e3_diffusion_for_molecules/eval_sample.py
```

**Open an interactive Bash shell:**
```bash
docker compose run --rm -it diffusion /bin/bash
```