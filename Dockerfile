FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libglib2.0-0 \
    tesseract-ocr-all \
    && rm -rf /var/lib/apt/lists/*

COPY ./README.md ./pyproject.toml ./poetry.lock /code/
COPY ./app/ /code/app/

RUN pip install poetry && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip install -r requirements.txt

EXPOSE 7860
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860" ]
