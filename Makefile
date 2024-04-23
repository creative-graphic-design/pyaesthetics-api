IMAGE=pyaesthetics-api
PORT=7860

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE) .

.PHONY: docker-run
docker-run:
	docker run --rm -p $(PORT):$(PORT) $(IMAGE)

.PHONY: docker
docker: docker-build docker-run

.PHONY: run
run:
	uvicorn app.main:app --port $(PORT) --reload
