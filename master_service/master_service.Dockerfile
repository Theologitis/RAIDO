# Use a Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install flwr (Flower)
RUN pip install flwr

# Copy the project source code
COPY . .

# Expose the API port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]