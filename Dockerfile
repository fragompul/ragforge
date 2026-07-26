FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY examples/ examples/

RUN pip install --no-cache-dir .

CMD ["python", "examples/basic_pipeline.py"]
