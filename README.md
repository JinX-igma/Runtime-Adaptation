# Runtime-Adaptation

## Usage

This project uses Docker to provide a consistent and reproducible execution environment.
At this stage, the workflow consists of two steps.

1. Build the Docker image

Run the following command inside the project directory:

docker build -t wesad .

This creates the base environment with Python, PyTorch, and required dependencies.

2. Start the container

./run_wesad.sh

The script launches an interactive Docker container and mounts:
	•	src/ → /workspace/src
	•	logs/ → /workspace/logs

This allows source code to be edited on the host while logs persist outside the container.

Once inside the container, any module inside src/ can be executed, for example:

python -m src.train_baseline

This completes the initial setup required to begin development within the project’s Docker environment.