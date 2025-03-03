# make ENABLE_GPU=false run for disabling
ENABLE_GPU ?= true

GPU_FLAG = $(if $(filter true,$(ENABLE_GPU)),--rm --runtime=nvidia --gpus all,)
BIND_VOLUMES= -v ~/.allennlp:/home/fast/.allennlp -v ~/nltk_data:/home/fast/nltk_data -v ~/.cache:/home/fast/.cache -v `pwd`/data:/home/fast/FAST/data

build:
	docker build --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -t docker-fast .

run: build
	docker run -it $(GPU_FLAG) $(BIND_VOLUMES) --rm docker-fast 
