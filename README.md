# Local GenAI RAG with Elasticsearch & Mistral

- This how-to is based on this one from Elasticsearch: https://www.elastic.co/search-labs/blog/rag-with-llamaIndex-and-elasticsearch
- Instead of Elasticsearch cloud we're going to run everything locally
- The simplest way to get this done is to just clone this GitHub Repo for the code and docker setup
- I've tried this on a M1 Mac. Changes for Windows with WSL will come later.
- The biggest problems that I had were actually installing the dependencies rather than the code itself.

## Install Ollama
- Download Ollama from here https://ollama.com/download/mac
- Unzip, drag into applications and install
- do `ollama run mistral` (It's going to download the Mistral 7b model, 4.1GB size)
- Create a new folder in Documents "Elasticsearch-RAG"
- Open that folder in VSCode

## Install Elasticsearch & Kibana (Docker)
- Use the docker-compose file from the Log Monitoring course: https://github.com/team-data-science/GenAI-RAG/blob/main/docker-compose.yml
- Download Docker Desktop from here: https://www.docker.com/products/docker-desktop/
- Install docker desktop and sign in in the app/create a user -> sends you to the browser

### For Windows Users
**Configure WSL2 to use max only 4GB of ram:**
```
wsl --shutdown
notepad "$env:USERPROFILE/.wslconfig"
```
.wslconfig file:
```
[wsl2]
memory=4GB   # Limits VM memory in WSL 2 up to 4GB
```
**Modify the Linux kernel map count in WSL**
Do this before the start because Elasticsearch requires a higher value to work
`sudo sysctl -w vm.max_map_count=262144`

- go to the Elasticsearch-RAG folder and do `docker compose up`
- make sure you have Elasticsearch 8.11 or later (we use 8.16 here in this project)
- if you get this error on a mac then just open the console in the docker app: error getting credentials - err: exec: docker-credential-desktop: executable file not found in $PATH, out: 
- Install xcode command line tools: `xcode-select --install`
- make sure you're at python 3.8.1 or larger -> installed 3.13.0 from https://www.python.org/downloads/

## Setup the virtual Python environment

- setup virtual python environment - go to the Elasticsearch-RAG folder and do 
`python3 -m venv .elkrag`

Install required libraries:
```
source .elkrag/bin/activate
pip install llama-index (optional python3 -m pip install package name)
pip install llama-index-embeddings-ollama
pip install llama-index-llms-ollama
pip install llama-index-vector-stores-elasticsearch
pip install python-dotenv
```

## Write the data to Elasticsearch
- create / copy in the index.py file
- download the conversations.json file from https://github.com/srikanthmanvi/RAG-InsuranceCompany/blob/main/conversations.json
- if you get an error with the execution then check if pedantic version is <2.0 `pip show pydantic` if not do this: `pip install "pydantic<2.0`
- run the program index.py

## Check the data in Elasticsearch
- go to kibana http://localhost:5601/app/management/data/index_management/indices and see the new index called calls
- go to dev tools and try out this query `GET calls/_search?size=1` http://localhost:5601/app/dev_tools#/console/shell

## Query data from elasticsearch and create an output with Mistral
- if everything is good then run the query.py file
- try a few queries :)