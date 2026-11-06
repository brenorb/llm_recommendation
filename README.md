# llm_recommendation
App for recommending anime with LLMs

## Quick Start Guide

Install Docker and place the dataset in `data/` under the project directory.

- `data/rating.csv` needs the columns `user_id`, `anime_id`, and `rating`.
- `data/anime.csv` needs the columns `anime_id`, `name`, `genre`, `type`, `episodes`, `rating`, and `members`.

The first run trains the recommendation model and writes `data/nmf_components.pkl`
if that file is missing. Set `OPENAI_API_KEY` in your shell to enable `/talk`.
The other routes work without a key.

```sh
docker build -t llm_rec .
docker run --rm -e OPENAI_API_KEY -p 127.0.0.1:5000:5000 \
  --mount "type=bind,src=$PWD/data,dst=/app/data" llm_rec
```

### API Endpoints
- **GET /**: Returns a welcome message.
- **GET /anime**: Queries the anime dataset based on various filters. E.g. `/anime?mt_episodes=200&lt_episodes=300&mt_rating=7&mt_members=1000&genre=Drama` queries for a Drama anime with 200 to 300 episodes, a rating of more than 7, and more than 1000 members.
- **GET /user**: Queries the user dataset based on various filters. E.g. `/user?user_id=567&mt_rating=5&lt_rating=7` queries for all entries of user 567 with a rating of more than 5 and less than 7.
- **GET /recommend/<int:user_id>**: Provides recommendations for a specified user. The user can specify the number of recommendations to return using the `k` parameter. If not specified, the default value is 5.
- **GET /talk/<int:user_id>/<string:message>**: Uses OpenAI to recommend an anime based on the user's request.

### Querying Anime Data
You can query anime data using the `/anime` endpoint with various parameters. Here are some examples:

**List of anime filters**:

- **anime_id** - Get anime information by ID.
- **name** - Get anime information by name.
- **genre** - Get anime information by genre.
- **type** - Get anime information by type.
- **episodes** - Get anime information by episode count.
- **mt_episodes** - Get anime information by more than a certain episode count.
- **lt_episodes** - Get anime information by less than a certain episode count.
- **rating** - Get anime information by rating.
- **mt_rating** - Get anime information by more than a certain rating.
- **lt_rating** - Get anime information by less than a certain rating.
- **members** - Get anime information by number of members.
- **mt_members** - Get anime information by more than a certain number of members.
- **lt_members** - Get anime information by less than a certain number of members.

You can combine multiple parameters to refine your search. For example:
```
GET /anime?genre=Adventure&type=Movie&mt_rating=7.5
```
This will return anime that are of the Adventure genre, are Movies, and have a rating greater than 7.5.

### Querying User Data
You can query user data using the `/user` endpoint with various parameters. Here are some examples:

**List of user filters**:

- **user_id** - Get user information by ID.
- **anime_id** - Get user information by anime ID.
- **rating** - Get user information by rating.
- **mt_rating** - Get user information by more than a certain rating.
- **lt_rating** - Get user information by less than a certain rating.

You can combine multiple parameters to refine your search. For example:

```
GET /user?anime_id=456&lt_rating=7.5
```

This will return users that watched anime with ID 456 and have rated it less than 7.5.

### Querying Recommendations
You can query recommendations using the `/recommend/<int:user_id>` endpoint with various parameters. Here are some examples:

**List of recommendation filters**:

- **k** - Get recommendations for a specified user. The user can specify the number of recommendations to return using the `k` parameter. If not specified, the default value is 5.

You can combine multiple parameters to refine your search. For example:

```
GET /recommend/123?k=7
```

This will return the top 7 anime recommendations for user with ID 123.

### Querying Natural Language
Send a natural language request to `/talk`. Percent-encode the message in the URL.

```
GET /talk/123/I%20want%20a%20happy%20anime
```

This will return anime recommendations for user with ID 123 that are uplifting and happy.
