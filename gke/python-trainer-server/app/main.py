import logging
import os
import re
import time
import uuid
from collections import Counter
from typing import List

import httpx
import requests

# --- PyTorch & ML Imports ---
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from kubernetes import client, config
from torch.utils.data import DataLoader, Dataset

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
APP_MODE = os.getenv("APP_MODE", "api")  # 'api', 'train', or 'infer'
MODELS_DIR = "/models"
MODEL_NAME = "sentiment_model.pth"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
INFERENCE_SERVICE_URL = "http://inference-server.trainer-server.svc.cluster.local:8000"
FILE_SERVER_URL = "http://file-server.trainer-server.svc.cluster.local:8080"


# --- Modern Text Processing (replacing TorchText) ---
class SimpleTokenizer:
    def __init__(self):
        self.pattern = re.compile(r"\b\w+\b")

    def __call__(self, text: str) -> List[str]:
        return self.pattern.findall(text.lower())


class Vocabulary:
    def __init__(self, tokens: List[str], min_freq: int = 1):
        self.unk_token = "<unk>"
        self.token_to_idx = {self.unk_token: 0}
        self.idx_to_token = {0: self.unk_token}

        # Count token frequencies
        token_freqs = Counter(tokens)

        # Build vocabulary
        idx = 1
        for token, freq in token_freqs.items():
            if freq >= min_freq and token != self.unk_token:
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
                idx += 1

    def __len__(self):
        return len(self.token_to_idx)

    def __call__(self, tokens: List[str]) -> List[int]:
        return [self.token_to_idx.get(token, 0) for token in tokens]

    def get_unk_idx(self):
        return 0


class SST2Dataset(Dataset):
    """Simple SST-2 dataset implementation"""

    def __init__(self, split="train"):
        # For demo purposes, using a simple hardcoded dataset
        # In production, you'd load from actual SST-2 data files
        if split == "train":
            self.data = [
                ("This movie is great!", 1),
                ("I love this film", 1),
                ("Amazing performance", 1),
                ("Excellent cinematography", 1),
                ("Brilliant acting", 1),
                ("This movie is terrible", 0),
                ("I hate this film", 0),
                ("Awful performance", 0),
                ("Poor cinematography", 0),
                ("Bad acting", 0),
                ("The story is compelling and well-written", 1),
                ("Great special effects and soundtrack", 1),
                ("Disappointing and boring plot", 0),
                ("Waste of time and money", 0),
                ("Fantastic direction and editing", 1),
                ("Mediocre script and pacing", 0),
                ("Outstanding character development", 1),
                ("Confusing and poorly executed", 0),
                ("Visually stunning masterpiece", 1),
                ("Completely forgettable movie", 0),
            ]
        else:
            self.data = [("This is a good movie", 1), ("This is a bad movie", 0)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        return label, text  # Return in torchtext format (label, text)


# --- Kubernetes Configuration ---
def configure_kubernetes_client():
    try:
        config.load_incluster_config()
        logging.info("Loaded in-cluster Kubernetes config.")
    except config.ConfigException:
        logging.warning(
            "Could not load in-cluster config. Falling back to kube_config."
        )
        config.load_kube_config()
    return client.BatchV1Api()


# --- Model Definition (Text Classification) ---
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


# --- Shared Utilities ---
tokenizer = SimpleTokenizer()


def upload_model_to_file_server(model_path, filename="sentiment_model.pth"):
    """Upload model to HTTP file server"""
    try:
        with open(model_path, "rb") as f:
            response = requests.put(f"{FILE_SERVER_URL}/{filename}", data=f)
            response.raise_for_status()
            logging.info(f"Model uploaded successfully to file server: {filename}")
            return True
    except Exception as e:
        logging.error(f"Failed to upload model: {e}")
        return False


def download_model_from_file_server(local_path, filename="sentiment_model.pth"):
    """Download model from HTTP file server"""
    try:
        response = requests.get(f"{FILE_SERVER_URL}/{filename}")
        response.raise_for_status()

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(response.content)
        logging.info(f"Model downloaded successfully from file server: {filename}")
        return True
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        return False


# ==============================================================================
#  MODE 1: API Server (Controller) - Runs on CPU nodes
# ==============================================================================
def create_api_app():
    app = FastAPI(title="AI Model Controller API")
    k8s_batch_v1 = configure_kubernetes_client()

    @app.get("/health", status_code=200)
    def health_check():
        return {"status": "API Server is running"}

    @app.post("/train", status_code=202)
    async def trigger_training_job():
        job_name = f"sentiment-training-job-{uuid.uuid4().hex[:6]}"
        namespace = os.getenv("POD_NAMESPACE", "trainer-server")
        image_name = os.getenv(
            "TRAINING_IMAGE",
            "anthonycorletti/krm-examples-python-trainer-server:latest",
        )

        container = client.V1Container(
            name="trainer",
            image=image_name,
            command=["python3", "main.py"],
            env=[client.V1EnvVar(name="APP_MODE", value="train")],
            resources=client.V1ResourceRequirements(
                limits={"nvidia.com/gpu": "1"}, requests={"memory": "4Gi", "cpu": "1"}
            ),
            image_pull_policy="Always",  # Ensure always pull latest image
        )
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": "model-trainer"}),
            spec=client.V1PodSpec(
                restart_policy="Never",
                containers=[container],
                node_selector={"cloud.google.com/gke-accelerator": "nvidia-tesla-p100"},
            ),
        )
        job_spec = client.V1JobSpec(template=template, backoff_limit=1)
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name),
            spec=job_spec,
        )
        try:
            k8s_batch_v1.create_namespaced_job(body=job, namespace=namespace)
            logging.info(f"Created training job: {job_name}")
            return {
                "message": "Training job started successfully",
                "job_name": job_name,
            }
        except client.ApiException as e:
            logging.error(f"Error creating Kubernetes job: {e.body}")
            raise HTTPException(
                status_code=500, detail="Failed to create training job."
            )

    @app.post("/infer")
    async def proxy_inference(request: Request):
        try:
            client_data = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400, detail="Request body must be valid JSON."
            )
        logging.info(f"Forwarding inference request to {INFERENCE_SERVICE_URL}/predict")
        async with httpx.AsyncClient() as client_http:
            try:
                response = await client_http.post(
                    f"{INFERENCE_SERVICE_URL}/predict", json=client_data, timeout=15.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                raise HTTPException(
                    status_code=503, detail=f"Inference service unavailable: {e}"
                )
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code, detail=e.response.json()
                )

    return app


# ==============================================================================
#  MODE 2: Training Script - Runs as a Job on GPU training nodes
# ==============================================================================
def run_training():
    logging.info("--- Starting Sentiment Analysis Model Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on device: {device}")

    # Load dataset and build vocabulary
    logging.info("Loading dataset and building vocabulary...")
    train_dataset = SST2Dataset(split="train")

    # Extract all tokens for vocabulary building
    all_tokens = []
    for _, text in train_dataset:
        tokens = tokenizer(text)
        all_tokens.extend(tokens)

    vocab = Vocabulary(all_tokens, min_freq=1)
    logging.info(f"Vocabulary size: {len(vocab)}")

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for label, text in batch:
            label_list.append(int(label))
            processed_text = torch.tensor(vocab(tokenizer(text)), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch
    )

    # Model setup
    VOCAB_SIZE = len(vocab)
    EMBED_DIM = 64
    NUM_CLASS = 2  # Positive / Negative
    model = TextClassificationModel(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    # Training loop
    num_epochs = 5
    logging.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        total_loss = 0
        for i, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        logging.info(
            f"| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | avg loss: {avg_loss:.4f} |"
        )

    logging.info("Finished Training.")

    # Save model locally first
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(
        {
            "vocab": vocab,
            "model_state_dict": model.state_dict(),
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
            "num_class": NUM_CLASS,
        },
        MODEL_PATH,
    )
    logging.info(f"Model and vocab saved locally to {MODEL_PATH}")

    # Upload to file server
    if upload_model_to_file_server(MODEL_PATH):
        logging.info("Model successfully uploaded to shared storage")
    else:
        logging.error("Failed to upload model to shared storage")


# ==============================================================================
#  MODE 3: Inference Server - Runs as a Deployment on GPU inference nodes
# ==============================================================================
def create_inference_app():
    app = FastAPI(title="Sentiment Analysis Inference Server")
    model_state = {
        "model": None,
        "vocab": None,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    logging.info(f"Inference device configured to: {model_state['device']}")

    def _load_model():
        # First try to download from file server
        if download_model_from_file_server(MODEL_PATH):
            logging.info("Model downloaded from file server")
        elif not os.path.exists(MODEL_PATH):
            logging.warning(
                f"Model file not found at {MODEL_PATH} and download failed."
            )
            return

        try:
            checkpoint = torch.load(MODEL_PATH, map_location=model_state["device"])
            model_state["vocab"] = checkpoint["vocab"]

            # Get model parameters from checkpoint
            vocab_size = checkpoint.get("vocab_size", len(model_state["vocab"]))
            embed_dim = checkpoint.get("embed_dim", 64)
            num_class = checkpoint.get("num_class", 2)

            model = TextClassificationModel(vocab_size, embed_dim, num_class)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(model_state["device"])
            model.eval()
            model_state["model"] = model
            logging.info("Successfully loaded model and vocab.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}", exc_info=True)

    @app.on_event("startup")
    def startup_event():
        _load_model()

    @app.get("/health", status_code=200)
    def health_check():
        return {
            "status": "Inference Server is running",
            "model_loaded": model_state["model"] is not None,
        }

    @app.post("/predict")
    def predict(payload: dict):
        if not model_state["model"]:
            raise HTTPException(status_code=404, detail="Model not loaded.")
        if "text" not in payload or not isinstance(payload["text"], str):
            raise HTTPException(
                status_code=400,
                detail="Payload must contain a 'text' field as a string.",
            )

        text = payload["text"]
        vocab = model_state["vocab"]
        model = model_state["model"]

        with torch.no_grad():
            tokens = tokenizer(text)
            processed_text = torch.tensor(vocab(tokens), dtype=torch.int64).to(
                model_state["device"]
            )
            offsets = torch.tensor([0]).to(model_state["device"])
            output = model(processed_text, offsets)

            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, 1)

            sentiment = "positive" if predicted_class_idx.item() == 1 else "negative"

            return {"sentiment": sentiment, "confidence": confidence.item()}

    return app


# --- Main Application Runner ---
if __name__ == "__main__":
    if APP_MODE == "train":
        run_training()
    elif APP_MODE == "infer":
        import uvicorn

        uvicorn.run(
            "main:create_inference_app", host="0.0.0.0", port=8000, factory=True
        )
    else:  # Default to API mode
        import uvicorn

        uvicorn.run("main:create_api_app", host="0.0.0.0", port=8080, factory=True)
