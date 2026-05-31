"""
Modal cloud training for BERT-family fine-tuning on A10G GPU.

Supports two models (selectable via --model):
  - indobert : indobenchmark/indobert-base-p2  (default)
  - mbert    : bert-base-multilingual-cased

Usage (from project root):

    # 1. One-time: regenerate processed data locally
    python src/preprocess.py

    # 2. Train (blocking)
    modal run modal/train_indobert_modal.py::main --model indobert
    modal run modal/train_indobert_modal.py::main --model mbert

    # 3. Detached
    modal run --detach modal/train_indobert_modal.py::main --model mbert

    # 4. Download all artifacts (both models) back to local
    modal run modal/train_indobert_modal.py::download
"""
import json
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
APP_NAME = "indobert-hoax"
VOLUME_NAME = "indobert-hoax-models"
DATA_DIR_IN_CONTAINER = "/root/data/pipeline_b"

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "pipeline_b"
LOCAL_SRC_DIR = PROJECT_ROOT / "src"

# Model registry --------------------------------------------------------------
MODELS = {
    "indobert": {
        "hf_name": "indobenchmark/indobert-base-p2",
        "tag": "IndoBERT Finetuned",
        "subdir": "indobert_finetuned",
        "cm_filename": "cm_indobert_finetuned.png",
    },
    "mbert": {
        "hf_name": "bert-base-multilingual-cased",
        "tag": "mBERT Finetuned",
        "subdir": "mbert_finetuned",
        "cm_filename": "cm_mbert_finetuned.png",
    },
}

# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "optuna==3.6.1",
        "scikit-learn==1.5.1",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "matplotlib==3.9.2",
        "tqdm==4.66.5",
    )
    .add_local_dir(str(LOCAL_DATA_DIR), DATA_DIR_IN_CONTAINER)
    .add_local_dir(str(LOCAL_SRC_DIR), "/root/src")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    volumes={"/models": volume},
    timeout=60 * 60 * 4,
)
def train(model: str = "indobert", n_trials: int = 20):
    """Run Optuna search + final fine-tuning on the chosen BERT model."""
    import os
    import shutil
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Pick one of: {list(MODELS)}")
    cfg = MODELS[model]
    output_dir = f"/models/{cfg['subdir']}"

    sys.path.insert(0, "/root/src")
    import torch
    from train_indobert import run

    print("=" * 60)
    print(f"PyTorch: {torch.__version__}   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {cfg['hf_name']}  ->  {output_dir}")
    print("=" * 60)

    metrics = run(
        data_dir=DATA_DIR_IN_CONTAINER,
        output_dir=output_dir,
        n_trials=n_trials,
        model_name=cfg["hf_name"],
        model_tag=cfg["tag"],
    )

    # persist metrics + confusion-matrix PNG into the volume
    with open(f"{output_dir}/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    cm_src = f"results/{cfg['cm_filename']}"
    if os.path.exists(cm_src):
        shutil.copy(cm_src, f"{output_dir}/{cfg['cm_filename']}")
        print(f"Confusion matrix copied -> {output_dir}/{cfg['cm_filename']}")
    else:
        print(f"WARN: CM not found at {cm_src}")
    volume.commit()
    print(f"Done. Artifacts in volume '{VOLUME_NAME}' under {cfg['subdir']}/")
    return metrics


# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    volumes={"/models": volume},
    timeout=60 * 15,
)
def eval_cm(model: str = "indobert"):
    """Re-evaluate a model already in the volume → regenerate CM PNG and metrics.

    Cheap: ~30 sec on A10G. Avoids re-training when CM is missing/stale.
    """
    import os
    import shutil
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Pick: {list(MODELS)}")
    cfg = MODELS[model]
    model_dir = f"/models/{cfg['subdir']}"

    if not os.path.exists(f"{model_dir}/config.json"):
        raise FileNotFoundError(f"No trained model at {model_dir}. Run `train` first.")

    sys.path.insert(0, "/root/src")
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm import tqdm
    from evaluate import evaluate

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    m.eval()

    test_df = pd.read_csv(f"{DATA_DIR_IN_CONTAINER}/test.csv").dropna()
    enc = tokenizer(list(test_df["text"].astype(str)),
                    add_special_tokens=True, max_length=128,
                    padding="max_length", truncation=True,
                    return_attention_mask=True, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(test_df["label"].values, dtype=torch.long))
    loader = DataLoader(ds, batch_size=32)

    preds, ys = [], []
    with torch.no_grad():
        for ids, mask, lbl in tqdm(loader, desc="eval"):
            ids, mask = ids.to(device), mask.to(device)
            out = m(ids, attention_mask=mask)
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            ys.extend(lbl.numpy())

    metrics = evaluate(ys, preds, model_name=cfg["tag"])
    cm_src = f"results/{cfg['cm_filename']}"
    if os.path.exists(cm_src):
        shutil.copy(cm_src, f"{model_dir}/{cfg['cm_filename']}")
        print(f"CM saved -> {model_dir}/{cfg['cm_filename']}")
    with open(f"{model_dir}/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    volume.commit()
    return metrics


@app.function(volumes={"/models": volume})
def list_artifacts():
    import os
    for root, _, files in os.walk("/models"):
        for f in files:
            p = os.path.join(root, f)
            size = os.path.getsize(p) / (1024 * 1024)
            print(f"  {p}  ({size:.2f} MB)")


@app.function(volumes={"/models": volume})
def read_artifact(path: str) -> bytes:
    with open(f"/models/{path}", "rb") as f:
        return f.read()


@app.function(volumes={"/models": volume})
def list_artifact_paths() -> list[str]:
    import os
    out = []
    for root, _, files in os.walk("/models"):
        for f in files:
            full = os.path.join(root, f)
            out.append(os.path.relpath(full, "/models"))
    return out


# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(model: str = "indobert", n_trials: int = 20):
    """Train a single model. --model {indobert|mbert}"""
    print(f"Launching {model} training on Modal (A10G, n_trials={n_trials})...")
    metrics = train.remote(model=model, n_trials=n_trials)
    print("\n=== Test set metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nNext: `modal run modal/train_indobert_modal.py::download` to fetch artifacts.")


@app.local_entrypoint()
def train_both(n_trials: int = 20):
    """Train both indobert and mbert back-to-back."""
    for m in ["indobert", "mbert"]:
        print(f"\n{'#' * 60}\n# Training {m}\n{'#' * 60}")
        metrics = train.remote(model=m, n_trials=n_trials)
        print(f"--- {m} test metrics ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


@app.local_entrypoint()
def download():
    """Download entire volume contents -> local models/ dir via `modal volume get`.

    Note: prefer running `modal volume get indobert-hoax-models / models --force`
    directly — it streams in parallel and is much faster than per-file gRPC reads.
    """
    import subprocess
    out_root = PROJECT_ROOT / "models"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Streaming volume '{VOLUME_NAME}' -> {out_root} ...")
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME, "/", str(out_root), "--force"],
        check=True,
    )
    # Mirror confusion matrices into local results/ for convenience
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    for cfg in MODELS.values():
        cm_path = out_root / cfg["subdir"] / cfg["cm_filename"]
        if cm_path.exists():
            (results_dir / cfg["cm_filename"]).write_bytes(cm_path.read_bytes())
            print(f"  + results/{cfg['cm_filename']}")
    print("Done.")


@app.local_entrypoint()
def ls():
    list_artifacts.remote()


@app.local_entrypoint()
def make_cm(model: str = "indobert"):
    """Re-eval a trained model on GPU just to regenerate CM PNG. Fast & cheap."""
    print(f"Re-evaluating {model} on Modal GPU to generate CM...")
    metrics = eval_cm.remote(model=model)
    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nNow download just the small CM file:")
    cfg = MODELS[model]
    print(f"  modal volume get {VOLUME_NAME} /{cfg['subdir']}/{cfg['cm_filename']} "
          f"results\\{cfg['cm_filename']} --force")
