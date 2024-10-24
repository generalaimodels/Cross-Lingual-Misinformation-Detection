import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the mental health conversational agent."""
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 2e-5
    emotion_classes: int = 6
    hidden_size: int = 768
    dropout_rate: float = 0.1
    warmup_steps: int = 100
    gradient_clip: float = 1.0

class EmotionalDataset(Dataset):
    """Custom dataset for emotional conversation data."""
    
    def __init__(
        self,
        texts: List[str],
        emotions: List[int],
        tokenizer: AutoTokenizer,
        max_length: int
    ) -> None:
        self.texts = texts
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        emotion = self.emotions[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "emotion": torch.tensor(emotion, dtype=torch.long)
        }

class EmotionalAttention(nn.Module):
    """Custom attention mechanism for emotional understanding."""
    
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(hidden_states)
        attended_output = torch.sum(attention_weights * hidden_states, dim=1)
        return attended_output

class MentalHealthAgent(nn.Module):
    """Advanced mental health conversational agent with emotional understanding."""
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        
        # Load base transformer model
        self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Emotional understanding components
        self.emotional_attention = EmotionalAttention(config.hidden_size)
        
        # Multi-task learning heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.emotion_classes)
        )
        
        # Response generation components
        self.response_generator = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, self.transformer.config.vocab_size)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Apply emotional attention
        hidden_states = transformer_outputs.last_hidden_state
        attended_output = self.emotional_attention(hidden_states)
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(attended_output)
        
        # Response generation
        response_context = torch.cat([attended_output, hidden_states[:, 0]], dim=1)
        response_logits = self.response_generator(response_context)
        
        outputs = {
            "emotion_logits": emotion_logits,
            "response_logits": response_logits
        }
        
        if labels is not None:
            # Calculate losses
            emotion_loss = F.cross_entropy(emotion_logits, labels)
            response_loss = F.cross_entropy(
                response_logits.view(-1, response_logits.size(-1)),
                input_ids.view(-1)
            )
            
            # Combined loss with weighted components
            total_loss = emotion_loss + 0.5 * response_loss
            outputs["loss"] = total_loss
            
        return outputs

class MentalHealthTrainer:
    """Trainer class for the mental health agent."""
    
    def __init__(
        self,
        model: MentalHealthAgent,
        config: ModelConfig,
        device: torch.device
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["emotion"]
            )
            
            loss = outputs["loss"]
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        
        for batch in eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            preds = outputs["emotion_logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["emotion"].cpu().numpy())
        
        # Calculate metrics
        metrics = classification_report(
            all_labels,
            all_preds,
            output_dict=True
        )
        
        return metrics

def generate_response(
    self,
    text: str,
    tokenizer: AutoTokenizer
) -> Tuple[str, str]:
    """Generate empathetic response based on input text."""
    self.model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=self.config.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(self.device)
    
    with torch.no_grad():
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        # Get predicted emotion
        emotion_idx = outputs["emotion_logits"].argmax(dim=-1)
        
        # Generate response
        response_ids = torch.argmax(outputs["response_logits"], dim=-1)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        return response, str(emotion_idx.item())

def save_checkpoint(
    self,
    path: Union[str, Path],
    epoch: int,
    metrics: Dict[str, float]
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "scheduler_state_dict": self.scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")

def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=self.device)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint

def main():
    """Main training pipeline."""
    try:
        # Initialize configuration
        config = ModelConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Load emotional conversation dataset
        # Note: Replace with actual dataset loading
        dataset = load_dataset("emotion")
        
        # Prepare datasets
        train_dataset = EmotionalDataset(
            texts=dataset["train"]["text"],
            emotions=dataset["train"]["label"],
            tokenizer=tokenizer,
            max_length=config.max_length
        )
        
        eval_dataset = EmotionalDataset(
            texts=dataset["validation"]["text"],
            emotions=dataset["validation"]["label"],
            tokenizer=tokenizer,
            max_length=config.max_length
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            pin_memory=True
        )

        # Initialize model and trainer
        model = MentalHealthAgent(config).to(device)
        trainer = MentalHealthTrainer(model, config, device)

        # Training loop
        best_f1 = 0.0
        for epoch in range(config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
            
            # Train
            train_loss = trainer.train_epoch(train_loader, epoch)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            metrics = trainer.evaluate(eval_loader)
            current_f1 = metrics["weighted avg"]["f1-score"]
            logger.info(f"Validation F1: {current_f1:.4f}")

            # Save best model
            if current_f1 > best_f1:
                best_f1 = current_f1
                trainer.save_checkpoint(
                    "best_model.pt",
                    epoch,
                    metrics
                )

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()