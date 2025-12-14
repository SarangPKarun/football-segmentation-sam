# Football Segmentation with SAM2

A Flask-based web application for segmenting football videos using the Segment Anything Model 2 (SAM2) and YOLO. This tool allows users to upload videos, extract frames, and perform interactive segmentation.

## Prerequisites

- **Anaconda**: Recommended for environment management.
- **GPU**: NVIDIA GPU with CUDA support is highly recommended for faster processing.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd football-segmentation-sam
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate footb
    ```

3.  **Install SAM2:**
    This project relies on SAM2. Clone and install it as an editable package:
    ```bash
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    pip install -e .
    ```

4.  **Download SAM2 Checkpoints:**
    Download the necessary model checkpoints:
    ```bash
    cd checkpoints
    ./download_ckpts.sh
    cd ../..
    ```
    *(Note: Ensure you are back in the project root directory).*

## Usage

1.  **Set Flask Environment Variables:**
    ```bash
    export FLASK_APP=app/app.py
    export FLASK_ENV=development
    ```
    *On Windows PowerShell:*
    ```powershell
    $env:FLASK_APP = "app/app.py"
    $env:FLASK_ENV = "development"
    ```

2.  **Run the Application:**
    ```bash
    flask run
    ```
    Or run directly with python:
    ```bash
    python app/app.py
    ```

3.  **Access the Web Interface:**
    Open your browser and navigate to `http://127.0.0.1:5000`.

4.  **Workflow:**
    - **Upload**: Upload a football video file.
    - **Annotate**: Select points on the extracted frames to guide the segmentation.
    - **Segment**: Run the segmentation process to process the video.

## Project Structure

- `app/`: Contains the main Flask application code (`app.py`).
- `utils/`: Helper scripts for video processing and segmentation logic.
- `static/`: Stores uploaded videos, extracted frames, and segmentation results.
- `templates/`: HTML templates for the web interface.
- `environment.yml`: Conda environment specification.
- `samsegment.ipynb` & `footballl segmentation.ipynb`: Jupyter notebooks for experimentation and testing.

## GPU Monitoring

To monitor GPU usage during heavy segmentation tasks, you can use:
```bash
watch -n 1 nvidia-smi
```
