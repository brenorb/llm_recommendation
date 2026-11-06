# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code. Supply the dataset and credentials at runtime.
COPY main.py .
COPY llm/ ./llm/
COPY utils/ ./utils/

# Command to run the application
CMD ["python", "-m", "flask", "--app", "main", "run", "--host", "0.0.0.0"]

EXPOSE 5000
