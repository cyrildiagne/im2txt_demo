name = im2txt_demo

build:
	docker build . -t $(name)

stop:
	docker rm -f $(name) || true

run: stop
	docker run --rm=true -p 50051:50051 --name=im2txt_demo im2txt_demo python im2txt/im2txt/run_inference_service.py
start: stop
	docker run -it --rm=true -p 8888:8888 -v $(shell pwd):/root --name=$(name) $(name) run_jupyter.sh -l
