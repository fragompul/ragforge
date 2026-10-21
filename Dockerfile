FROM python:3.12-slim

LABEL org.opencontainers.image.title="ragforge" \
      org.opencontainers.image.description="Zero-dependency hybrid RAG toolkit" \
      org.opencontainers.image.source="https://github.com/fragompul/ragforge" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY examples/ examples/

RUN pip install --no-cache-dir . && useradd --create-home ragforge
USER ragforge

ENTRYPOINT ["ragforge"]
CMD ["demo", "--no-color"]
