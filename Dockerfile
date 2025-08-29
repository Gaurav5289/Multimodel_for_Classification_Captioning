# -------------------------------
# 1. Base image
# -------------------------------
FROM python:3.10-slim-bullseye

# -------------------------------
# 2. Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# 3. Install system dependencies
# -------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 4. Create non-root user
# -------------------------------
RUN addgroup --system app && adduser --system --group app

# -------------------------------
# 5. Copy requirements and install Python dependencies
# -------------------------------
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 6. Copy the application code
# -------------------------------
COPY --chown=app:app . .

# -------------------------------
# 7. Switch to non-root user
# -------------------------------
USER app

# -------------------------------
# 8. Expose port for FastAPI
# -------------------------------
EXPOSE 8000

# -------------------------------
# 9. Start the FastAPI app with Uvicorn
# -------------------------------
CMD ["uvicorn", "SIC.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
