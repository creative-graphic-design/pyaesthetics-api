IMAGE=pyaesthetics-api
PORT=7860

#
# Docker
#

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE) .

.PHONY: docker-run
docker-run:
	docker run --rm -p $(PORT):$(PORT) $(IMAGE)

.PHONY: docker
docker: docker-build docker-run

#
# Development
#

.PHONY: run
run:
	uvicorn app.main:app --port $(PORT) --reload

#
# Linter/Formatter/TypeCheck
#

.PHONY: lint
lint:
	poetry run ruff check --output-format=github .

.PHONY: format
format:
	poetry run ruff format --check --diff .

.PHONY: typecheck
typecheck:
	poetry run mypy --cache-dir=/dev/null .
