# Use official Python 3.10 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Expose port 8000
EXPOSE 8000

# Run Streamlit app
CMD ["python3", "-m", "streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0", "--server.headless=true"]