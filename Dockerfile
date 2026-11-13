
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train.py predict.py app.py model.bin ./

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]