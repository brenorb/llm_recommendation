#!/bin/sh
set -eu

docker build --tag llm_rec .
docker run --rm -e OPENAI_API_KEY -p 127.0.0.1:5000:5000 \
  --mount "type=bind,src=$PWD/data,dst=/app/data" llm_rec
