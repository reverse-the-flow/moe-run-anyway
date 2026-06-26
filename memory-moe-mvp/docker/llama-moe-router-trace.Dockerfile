FROM debian:12-slim AS build

ARG LLAMA_CPP_REF=beac5309f1bc67534f509bf29420abf58fff063c

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        git \
        make \
        patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git init llama.cpp \
    && cd llama.cpp \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch --depth 1 origin "${LLAMA_CPP_REF}" \
    && git checkout FETCH_HEAD

COPY patches/llama-cpp-moe-router-trace-example.patch /tmp/llama-cpp-moe-router-trace-example.patch

RUN cd /src/llama.cpp \
    && git apply /tmp/llama-cpp-moe-router-trace-example.patch \
    && cmake -S . -B build-moe-trace \
        -DGGML_CUDA=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build-moe-trace --target llama-moe-router-trace -j 8

FROM debian:12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/llama.cpp/build-moe-trace/bin/ /app/

ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["/app/llama-moe-router-trace"]
