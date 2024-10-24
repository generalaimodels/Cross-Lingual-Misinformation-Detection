import logging
import sys
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MisinformationClassifier(nn.Module):
    """
    A classifier model for detecting misinformation using a pre-trained transformer.
    """
    def __init__(self, model_name: str, num_labels: int) -> None:
        """
        Initialize the misinformation classifier.

        Args:
            model_name (str): Name of the pre-trained transformer model.
            num_labels (int): Number of classification labels.
        """
        super(MisinformationClassifier, self).__init__()
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
            logger.info(f"Loaded pre-trained model '{model_name}' successfully.")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Token IDs.
            attention_mask (torch.Tensor): Attention masks.

        Returns:
            torch.Tensor: Logits.
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def tokenize_function(
    examples: Dict[str, Any], 
    tokenizer: AutoTokenizer, 
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Tokenize the input texts.

    Args:
        examples (Dict[str, Any]): Batch of examples.
        tokenizer (AutoTokenizer): Tokenizer instance.
        max_length (int, optional): Maximum sequence length. Defaults to 128.

    Returns:
        Dict[str, Any]: Tokenized outputs.
    """
    try:
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise

def prepare_datasets(
    raw_datasets: DatasetDict, 
    tokenizer: AutoTokenizer, 
    max_length: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """
    Tokenize and prepare DataLoaders for training and evaluation.

    Args:
        raw_datasets (DatasetDict): Raw dataset splits.
        tokenizer (AutoTokenizer): Tokenizer instance.
        max_length (int, optional): Maximum sequence length. Defaults to 128.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and evaluation DataLoaders.
    """
    try:
        tokenized_datasets = raw_datasets.map(
            lambda x: tokenize_function(x, tokenizer, max_length),
            batched=True
        )
        tokenized_datasets = tokenized_datasets.remove_columns(['text'])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        logger.info("Dataset tokenization and formatting successful.")
    except Exception as e:
        logger.error(f"Failed to tokenize datasets: {e}")
        raise

    try:
        train_loader = DataLoader(
            tokenized_datasets['train'], 
            batch_size=32, 
            shuffle=True
        )
        eval_loader = DataLoader(
            tokenized_datasets['validation'], 
            batch_size=32
        )
        logger.info("DataLoaders created successfully.")
        return train_loader, eval_loader
    except KeyError as e:
        logger.error(f"Missing dataset split: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {e}")
        raise

def train(
    model: nn.Module, 
    train_loader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device
) -> None:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The classifier model.
        train_loader (DataLoader): Training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device to train on.
    """
    model.train()
    total_loss = 0.0
    try:
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Training Loss: {avg_loss:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def evaluate(
    model: nn.Module, 
    eval_loader: DataLoader, 
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The classifier model.
        eval_loader (DataLoader): Evaluation DataLoader.
        device (torch.device): Device to evaluate on.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    model.eval()
    preds = []
    labels = []
    try:
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.detach().cpu().numpy()
                batch_preds = logits.argmax(axis=-1)
                
                preds.extend(batch_preds)
                labels.extend(batch_labels.cpu().numpy())
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "fpr": fpr,
            "fnr": fnr
        }
        logger.info(f"Evaluation Metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def save_model(model: nn.Module, tokenizer: AutoTokenizer, output_dir: str) -> None:
    """
    Save the trained model and tokenizer.

    Args:
        model (nn.Module): The classifier model.
        tokenizer (AutoTokenizer): Tokenizer instance.
        output_dir (str): Directory to save the model.
    """
    try:
        model.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to '{output_dir}'.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

def load_datasets(dataset_name: str) -> DatasetDict:
    """
    Load datasets from Hugging Face Hub.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        DatasetDict: Loaded datasets.
    """
    from datasets import load_dataset
    try:
        datasets = load_dataset(dataset_name)
        logger.info(f"Dataset '{dataset_name}' loaded successfully.")
        return datasets
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        raise

def main() -> None:
    """
    Main function to execute the misinformation detection pipeline.
    """
    try:
        # Configuration
        MODEL_NAME = "xlm-roberta-base"
        DATASET_NAME = "your_dataset_name"  # Replace with actual dataset name
        NUM_LABELS = 2  # Example: 0 - True, 1 - Misinformation
        OUTPUT_DIR = "./misinfo_model"
        NUM_EPOCHS = 3
        LEARNING_RATE = 2e-5
        EPSILON = 1e-8
        MAX_LENGTH = 128
        BATCH_SIZE = 32

        # Check for CUDA
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Using device: {device}")

        # Load datasets
        raw_datasets = load_datasets(DATASET_NAME)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Prepare DataLoaders
        train_loader, eval_loader = prepare_datasets(raw_datasets, tokenizer, MAX_LENGTH)

        # Initialize model
        model = MisinformationClassifier(MODEL_NAME, NUM_LABELS)
        model.to(device)

        # Set up optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
        total_steps = len(train_loader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Training loop
        for epoch in range(NUM_EPOCHS):
            logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            train(model, train_loader, optimizer, scheduler, device)
            evaluate(model, eval_loader, device)

        # Save the trained model
        save_model(model, tokenizer, OUTPUT_DIR)

        logger.info("Training and evaluation completed successfully.")

    except Exception as e:
        logger.critical(f"An unrecoverable error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()