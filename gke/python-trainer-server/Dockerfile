FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-dev build-essential python3-pip && \
    apt-get clean && \
    python3 -m pip install --upgrade pip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache -r requirements.txt

COPY ./app/ .

CMD ["python3", "main.py"]
