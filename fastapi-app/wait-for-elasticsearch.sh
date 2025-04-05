#!/bin/sh
# wait-for-elasticsearch.sh

host="elasticsearch"
port=9200
echo "Waiting for Elasticsearch at ${host}:${port}..."
while ! nc -z $host $port; do   
  sleep 1
done
echo "Elasticsearch is up - starting FastAPI."
exec "$@"
