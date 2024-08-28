# llm_recommendation
App for recommending anime with LLMs

## Quick Start Guide

### Prerequisites
- Docker installed on your machine.
- An OpenAI API key (if using OpenAI services).

### Build the Docker Image
To build the Docker image, run the following command in the project directory:

```
docker build -t llm_rec .
docker run -d -e OPENAI_API_KEY=YOUR_API_KEY -p 5000:5000 llm_rec
```

### Run the Docker Container
To run the application, use the following command, replacing `YOUR_API_KEY` with your actual OpenAI API key:

### API Endpoints
- **GET /**: Returns a welcome message.
- **POST /upload**: Uploads ratings and anime CSV files for processing.
- **GET /recommend/<int:user_id>**: Recommends anime for a specific user based on their ID.

### Example Usage
1. Upload your CSV files to the `/upload` endpoint.
2. Use the `/recommend/<user_id>` endpoint to get anime recommendations.
