"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.

Implement the TODO functions to build a complete fine-tuning pipeline.

Default run: `python lab.py` reads `data/app_reviews_train.csv` (7,472 reviews
across 9 apps with 3 sentiment classes: 0=negative, 1=neutral, 2=positive)
and produces an internal 80/20 train/eval split with seed=42.

CI smoke run: workflow sets DATA_PATH=fixtures/tiny_app_reviews.csv (60 rows).

After training, push the fine-tuned model to your Hugging Face Hub account.
The model directory is local-only (gitignored).
"""

import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import precision_score, recall_score
from transformers import set_seed

import subprocess
import sys

# Upgrade the accelerate library to resolve compatibility issues 
# with the GitHub Actions/Autograder environment.
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "accelerate>=0.27.0"])
except Exception as e:
    print(f"Warning: Failed to upgrade accelerate: {e}")
# 3-class sentiment label mapping (matches the curated dataset's `label` column)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}




def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI uses a smoke CSV); otherwise return
    the default path to the curated app-review training CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the CSV at `data_path` and produce a train/test split.

    The CSV must have at least `text` and `label` columns. (The curated
    `data/app_reviews_train.csv` also includes `app`, `app_name`, and `rating`
    columns — these are useful for inspection but not required by the model.)

    Returns a `DatasetDict` with "train" and "test" keys.
    """
    #  read the CSV with pandas
    df=pd.read_csv(data_path)
    # convert with Dataset.from_pandas(df, preserve_index=False)
    data_set=Dataset.from_pandas(df,preserve_index=False)
    # split with .train_test_split(test_size=test_size, seed=seed)
    ds_dict=data_set.train_test_split(test_size=test_size,seed=seed)
    # return the resulting DatasetDict
    return ds_dict


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.

    `tokenizer` is a loaded HuggingFace tokenizer (callable) — load it once
    in `main()` via `AutoTokenizer.from_pretrained(...)` and pass it in.
    Use truncation=True and max_length=max_length. Do not pad here — padding is
    applied dynamically by DataCollatorWithPadding at training time.

    Note: this signature differs from the drill (`tokenize_dataset(ds, name)`)
    by accepting the loaded tokenizer object so `main()` doesn't re-load it.
    """
    #  define tokenize_fn(batch) calling the passed-in tokenizer with truncation + max_length
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation = True , 
            max_length=max_length
        )
    #  apply ds_dict.map(tokenize_fn, batched=True)
    tokenized = ds_dict.map(tokenize_fn,batched=True)
    #  return the tokenized DatasetDict
    return tokenized


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Return a TrainingArguments configured for fine-tuning."""
    #  return a TrainingArguments configured with the passed arguments.
    # In addition to wiring the kwargs through, set:
    #   - eval_strategy="epoch"           (renamed from evaluation_strategy in transformers 4.41+)
    #   - save_strategy="epoch"
    #   - logging_steps=50
    # The course pins transformers>=4.41,<5.0 — use the new argument names.
    traning_args= TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=seed,
        logging_steps=50,
        # Disable all experiment tracking logging
        report_to="none",  
        # Disable torch_compile to resolve compatibility issues with transformers and accelerate in the Autograder.
        torch_compile=False
    )
    traning_args.eval_strategy = "epoch"
    traning_args.save_strategy = "epoch"
    return traning_args

def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.

    Use sklearn's accuracy_score and f1_score with average="macro".
    """
    #  unpack eval_pred to logits, labels
    logits,labels= eval_pred
    #  argmax logits over axis 1
    predictions=np.argmax(logits,axis=1)
    #  compute accuracy and macro-F1
    #  return as a dict
    return{
         "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro")
    }


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Construct and train a Trainer.

    Returns the trained Trainer (trainer.model is the fine-tuned model). Pass
    id2label=ID2LABEL and label2id=LABEL2ID to the model so its config records
    the human-readable label names — Integration 7A reads them from
    `model.config.id2label` rather than hard-coding.
    """
    # load model with AutoModelForSequenceClassification.from_pretrained(
    #         model_name, num_labels=num_labels, id2label=ID2LABEL, label2id=LABEL2ID)
    model =AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        id2label=ID2LABEL, 
        label2id=LABEL2ID
    )
    #  build data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #  build Trainer with model, args, train/eval datasets, tokenizer, data_collator, compute_metrics
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
             
    )
    #  call trainer.train()
    trainer.train()
    #  return trainer
    return trainer
    


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.

    Read label names from trainer.model.config.id2label (do not hard-code).

    Returns: {"accuracy": float, "macro_f1": float, "per_class_f1": {label_name: f1, ...}}
    """
    # predict on tokenized_test using trainer.predict
    prediction_output=trainer.predict(tokenized_test)
    logits =prediction_output.predictions
    labels= prediction_output.label_ids
    # argmax predictions to class indices
    preds=np.argmax(logits,axis=1)
    
    # TODO: compute accuracy and macro-F1
    accuracy=accuracy_score(labels,preds)
    macro_f1=f1_score(labels,preds,average="macro")
    #  compute per-class F1 with f1_score(..., average=None)
    per_class_f1=f1_score(labels,preds,average=None)
    per_class_precision=precision_score(labels,preds,average=None,zero_division=0)
    per_class_recall=recall_score(labels,preds,average=None,zero_division=0)
    
    #  build per_class_f1 dict using trainer.model.config.id2label for label names
    id2label=trainer.model.config.id2label
    # return all three
    return{
        "accuracy":float(accuracy),
        "macro_f1":float(macro_f1),
        "per_class_f1":{id2label[i]: float(f) for i, f in enumerate(per_class_f1)},
        "per_class_precision":{id2label[i]: float(p) for i, p in enumerate(per_class_precision)},
        "per_class_recall": {id2label[i]: float(r) for i, r in enumerate(per_class_recall)}
    }


def main() -> None:
    """Orchestrate the full pipeline."""
    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"
    set_seed(42)

    ds = prepare_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = make_training_args(output_dir)
    trainer = train_classifier(tokenized, model_name, training_args, tokenizer, num_labels=3)

    # Save locally (model/ is gitignored)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    metrics = evaluate_classifier(trainer, tokenized["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV
    pred_logits = trainer.predict(tokenized["test"]).predictions
    pred_idx = np.argmax(pred_logits, axis=1)
    pred_probs = _softmax(pred_logits)
    id2label = trainer.model.config.id2label
    df_out = pd.DataFrame({
    "text": ds["test"]["text"],
    "label": [id2label[i] for i in ds["test"]["label"]],
    "predicted_label": [id2label[i] for i in pred_idx],
    "predicted_probability": [
        float(pred_probs[i, pred_idx[i]]) for i in range(len(pred_idx))
    ],
    "prob_negative": pred_probs[:, 0],
    "prob_neutral": pred_probs[:, 1],
    "prob_positive": pred_probs[:, 2],
    })
    df_out.to_csv("predictions.csv", index=False)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    # Confusion matrix (for the evaluation report)
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=list(id2label.values()),
    )
    print(pd.DataFrame(cm, index=list(id2label.values()), columns=list(id2label.values())).to_string())
    
    cm_df = pd.DataFrame(
    cm,
    index=list(id2label.values()),
    columns=list(id2label.values())
)

    cm_df.to_csv("confusion_matrix.csv")

    # Push to Hugging Face Hub.
    # Skipped in CI (DATA_PATH set); requires `huggingface-cli login` locally.
    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to https://huggingface.co/<your-username>/{repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            print("Run `huggingface-cli login` and try again.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    main()
