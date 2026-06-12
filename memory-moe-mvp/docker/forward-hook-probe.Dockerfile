FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git procps \
    && rm -rf /var/lib/apt/lists/*

COPY docker/forward-hook-probe.requirements.txt /tmp/forward-hook-probe.requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/forward-hook-probe.requirements.txt

WORKDIR /workspace

CMD ["bash"]
