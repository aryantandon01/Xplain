# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Expose default FastAPI port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
