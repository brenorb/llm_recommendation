# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

ARG OPENAI_API_KEY

ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Command to run the application
CMD ["python", "main.py"]

EXPOSE 5000
