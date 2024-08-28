docker build --tag llm_rec .
docker run -d --name my_flask_app -e OPENAI_API_KEY=$OPENAI_API_KEY -p 5000:5000 llm_rec
docker exec -it my_flask_app celery -A llm_recommendation.main_celery worker --loglevel=info