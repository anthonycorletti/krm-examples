import logging
import os
import time
import uuid

import httpx

# --- PyTorch & ML Imports ---
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from kubernetes import client, config
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import SST2
from torchtext.vocab import build_vocab_from_iterator

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
APP_MODE = os.getenv("APP_MODE", "api")  # 'api', 'train', or 'infer'
MODELS_DIR = "/models"
MODEL_NAME = "sentiment_model.pth"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
INFERENCE_SERVICE_URL = "http://inference-server.ai-models.svc.cluster.local:8000"


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
tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# ==============================================================================
#  MODE 1: API Server (Controller) - Runs on CPU nodes (UNCHANGED)
# ==============================================================================
def create_api_app():
    # This entire function is identical to the previous version. The API controller
    # is agnostic to the model's task; it only manages the infrastructure.
    app = FastAPI(title="AI Model Controller API")
    k8s_batch_v1 = configure_kubernetes_client()

    @app.get("/health", status_code=200)
    def health_check():
        return {"status": "API Server is running"}

    @app.post("/train", status_code=202)
    async def trigger_training_job():
        job_name = f"sentiment-training-job-{uuid.uuid4().hex[:6]}"
        namespace = os.getenv("POD_NAMESPACE", "ai-models")
        image_name = os.getenv(
            "TRAINING_IMAGE", "anthonycorletti/krm-examples-fastapi-gpu:latest"
        )

        container = client.V1Container(
            name="trainer",
            image=image_name,
            command=["python", "main.py"],
            env=[client.V1EnvVar(name="APP_MODE", value="train")],
            volume_mounts=[
                client.V1VolumeMount(name="model-storage", mount_path=MODELS_DIR)
            ],
            resources=client.V1ResourceRequirements(
                limits={"nvidia.com/gpu": "1"}, requests={"memory": "4Gi", "cpu": "1"}
            ),
        )
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": "model-trainer"}),
            spec=client.V1PodSpec(
                restart_policy="Never",
                containers=[container],
                node_selector={"cloud.google.com/gke-accelerator": "nvidia-tesla-a100"},
                volumes=[
                    client.V1Volume(
                        name="model-storage",
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name="model-pvc"
                        ),
                    )
                ],
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
        client_data = await request.json()
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
    logging.info("Loading SST-2 dataset and building vocabulary...")
    train_iter = SST2(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    train_iter = SST2(split="train")
    dataloader = DataLoader(
        list(train_iter), batch_size=8, shuffle=True, collate_fn=collate_batch
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
        for i, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
        scheduler.step()
        logging.info(
            f"| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s |"
        )

    logging.info("Finished Training.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    # Save both model state and vocabulary
    torch.save(
        {
            "vocab": vocab,
            "model_state_dict": model.state_dict(),
        },
        MODEL_PATH,
    )
    logging.info(f"Model and vocab saved to {MODEL_PATH}")


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
        if not os.path.exists(MODEL_PATH):
            logging.warning(f"Model file not found at {MODEL_PATH}.")
            return
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=model_state["device"])
            model_state["vocab"] = checkpoint["vocab"]

            VOCAB_SIZE = len(model_state["vocab"])
            EMBED_DIM = 64
            NUM_CLASS = 2

            model = TextClassificationModel(VOCAB_SIZE, EMBED_DIM, NUM_CLASS)
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
        text_pipeline = lambda x: vocab(tokenizer(x))

        with torch.no_grad():
            processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64).to(
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
