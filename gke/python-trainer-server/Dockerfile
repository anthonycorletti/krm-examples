FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-dev && \
    apt-get clean && \
    python3 -m pip install --upgrade pip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/*

WORKDIR /app

COPY requirements.txt .
RUN uv --python 3.12 pip install --no-cache -r requirements.txt --system

COPY ./app/ .

CMD ["uv", "run", "main.py"]
