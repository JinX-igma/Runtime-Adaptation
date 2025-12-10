# Runtime-Adaptation

## Usage

This project uses Docker to provide a consistent and reproducible execution environment.  
At this stage, the workflow consists of two steps.

---

### 1. Build the Docker image

Run the following command inside the project directory:

~~~docker build -t wesad .~~~

This creates the base environment with Python, PyTorch, and required dependencies.

---

### 2. Start the container

~~~./run_wesad.sh~~~

The script launches an interactive Docker container and mounts:

- `src/` → `/workspace/src`
- `logs/` → `/workspace/logs`

This allows source code to be edited on the host while logs persist outside the container.

---

### Running code inside the container

Once inside the Docker environment, you may execute any module inside `src/`, for example:

~~~python3 -m src.samplecode~~~

This completes the initial setup required to begin development within the project’s Docker environment.
