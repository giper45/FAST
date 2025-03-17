# make ENABLE_GPU=false run for disabling
ENABLE_GPU ?= true

GPU_FLAG = $(if $(filter true,$(ENABLE_GPU)),--rm --runtime=nvidia --gpus all,)
BIND_VOLUMES= -v ~/.allennlp:/home/fast/.allennlp -v ~/nltk_data:/home/fast/nltk_data -v ~/.cache:/home/fast/.cache -v `pwd`/data:/home/fast/FAST/data

build-preprocess:
	docker build --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -t docker-fast-preprocess -f Dockerfile.allennlp . 
run-preprocess: build-preprocess
	docker run --name fast-docker -it $(GPU_FLAG) $(BIND_VOLUMES) --rm docker-fast-preprocess 

build:
#docker build --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -t docker-fast -f Dockerfile-nvidia .
	docker build --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -t docker-fast .

stop:
	docker stop fast-docker

run: build
	docker run --name fast-docker -it $(GPU_FLAG) $(BIND_VOLUMES) --rm docker-fast bash

run-bash: build
	docker run --name fast-docker --entrypoint /bin/bash -it $(GPU_FLAG) $(BIND_VOLUMES) --rm docker-fast 
# docker cp data fast-docker:/home/fast/FAST/data

run-example: 
	docker run --name fast-docker -it $(GPU_FLAG) $(BIND_VOLUMES) -v `pwd`/FAST/run_fast_example.py:/home/fast/FAST/FAST/run_fast_example.py --rm docker-fast bash


go:
	docker exec -it fast-docker bash