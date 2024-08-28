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
- **GET /anime**: Queries anime data based on various filters.

### Example Usage
1. Upload your CSV files to the `/upload` endpoint.
2. Use the `/recommend/<user_id>` endpoint to get anime recommendations.
   - Example: `/recommend/1?k=5` to get the top 5 anime recommendations for user with ID 1.

### Querying Anime Data
You can query anime data using the `/anime` endpoint with various parameters. Here are some examples:

- **Get anime by ID**:
  ```
  GET /anime?anime_id=1
  ```

- **Get anime by name**:
  ```
  GET /anime?name=Naruto
  ```

- **Get anime by genre**:
  ```
  GET /anime?genre=Action
  ```

- **Get anime by type**:
  ```
  GET /anime?type=TV
  ```

- **Get anime by episode count**:
  ```
  GET /anime?episodes=12
  ```

- **Get anime with more than a certain number of episodes**:
  ```
  GET /anime?mt_episodes=24
  ```

- **Get anime with a rating greater than a certain value**:
  ```
  GET /anime?mt_rating=8.0
  ```

- **Get anime with a specific number of members**:
  ```
  GET /anime?members=1000
  ```

You can combine multiple parameters to refine your search. For example:
```
GET /anime?genre=Adventure&type=Movie&mt_rating=7.5
```
This will return anime that are of the Adventure genre, are Movies, and have a rating greater than 7.5.