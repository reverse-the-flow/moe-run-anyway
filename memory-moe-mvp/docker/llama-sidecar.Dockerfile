FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/memory-moe

COPY llama_sidecar.py /opt/memory-moe/llama_sidecar.py
COPY docker/llama-with-sidecar-entrypoint.sh /opt/memory-moe/llama-with-sidecar-entrypoint.sh

RUN chmod +x /opt/memory-moe/llama-with-sidecar-entrypoint.sh

ENV LLAMA_SERVER_BIN=/app/llama-server
ENV LD_LIBRARY_PATH=/app
ENV LLAMA_SERVER_INTERNAL_HOST=127.0.0.1
ENV LLAMA_SERVER_INTERNAL_PORT=18080
ENV SIDECAR_LISTEN_HOST=0.0.0.0
ENV SIDECAR_LISTEN_PORT=8080
ENV SIDECAR_RUNS_DIR=/var/log/memory-moe-sidecar
ENV SIDECAR_LABEL=llama-sidecar
ENV SIDECAR_CAPTURE_GPU=1

VOLUME ["/var/log/memory-moe-sidecar"]
EXPOSE 8080

ENTRYPOINT ["/opt/memory-moe/llama-with-sidecar-entrypoint.sh"]
