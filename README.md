# Intelligent Scene Analysis System

This project is an end-to-end machine learning service that classifies natural scene images, generates descriptive captions, and provides confidence and uncertainty scores, all served via a containerized REST API. 

## Key Features

-   **Image Classification:** Classifies images into 6 categories (**buildings, forest, glacier, mountain, sea, street**) using a fine-tuned ResNet50 model with **~92.4%** validation accuracy.
-   **Scene Description:** Generates a short, context-aware caption for each image using the `Salesforce/blip-image-captioning-base` model.
-   **Uncertainty Estimation:** Provides a confidence score and an uncertainty estimate for each prediction using the Monte-Carlo Dropout technique.
-   **REST API:** The entire service is exposed via a robust FastAPI server with `/health`, `/model_info`, and `/predict` endpoints.
-   **Containerized Deployment:** The application is fully containerized with Docker for consistent, reproducible deployment.

## Technology Stack

-   **Model Development:** Python, TensorFlow, Keras
-   **Multimodal Features:** Hugging Face Transformers (PyTorch)
-   **API:** FastAPI, Uvicorn
-   **Containerization:** Docker
-   **Testing:** Pytest

---

## Setup and Installation

### Prerequisites

-   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
-   [Git](https://git-scm.com/downloads) installed.
-   The "Intel Image Classification" dataset. Download it from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Gaurav5289/Gaurav5289-ServiceHive-Assignment-MLE-Gaurav-Verma.git](https://github.com/Gaurav5289/Gaurav5289-ServiceHive-Assignment-MLE-Gaurav-Verma.git)
    cd Gaurav5289-ServiceHive-Assignment-MLE-Gaurav-Verma
    ```

2.  **Organize Data:**
    -   Create a directory `SIC/data`.
    -   Unzip the downloaded dataset and place the contents (`seg_train`, `seg_test`, `seg_pred`) inside the `SIC/data` directory.

---

## How to Run the Service

There are two ways to run the application: using Docker (recommended) or running the local server directly.

### Option 1: Run with Docker (Recommended)

1.  **Build the Docker image:**
    ```bash
    docker build -t scene-analysis-service .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 scene-analysis-service
    ```

### Option 2: Run Locally

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Uvicorn server:**
    ```bash
    uvicorn SIC.api.main:app --reload
    ```

## Usage

Once the service is running (either via Docker or Uvicorn), you can interact with it in two ways:

1.  **Web Frontend:**
    -   Open the `SIC/frontend/index.html` file in your web browser.
    -   Use the form to upload an image and see the results.

2.  **API Documentation:**
    -   Navigate to `http://127.0.0.1:8000/docs` in your web browser.
    -   FastAPI provides an interactive interface to test the API endpoints directly.

---

## Running Tests

To ensure the application is working correctly, you can run the automated unit tests.

```bash
pytest
```
