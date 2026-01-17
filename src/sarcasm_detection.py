import kagglehub

# Download latest version
path = kagglehub.dataset_download("rmisra/news-headlines-dataset-for-sarcasm-detection")

print("Path to dataset files:", path)

"""
Sarcasm Detection using Fine-Tuned BERT
A complete pipeline for binary classification of news headlines as sarcastic or non-sarcastic.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATASET PREPARATION
# ============================================================================

class SarcasmDataset(Dataset):
    """
    Custom PyTorch Dataset for sarcasm detection.
    Handles tokenization and encoding of headlines.
    """
    def __init__(self, headlines, labels, tokenizer, max_len=128):
        self.headlines = headlines
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.headlines)
    
    def __getitem__(self, idx):
        headline = str(self.headlines[idx])
        label = self.labels[idx]
        
        # Tokenize the headline
        encoding = self.tokenizer.encode_plus(
            headline,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_and_prepare_data(filepath, test_size=0.2, val_size=0.1):
    """
    Load dataset from JSON file and prepare train/val/test splits.
    Expected format: Each line is a JSON object with keys: headline, is_sarcastic, article_link
    """
    headlines = []
    labels = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                headlines.append(data['headline'])
                labels.append(data['is_sarcastic'])
    except FileNotFoundError:
        print(f"Dataset file not found at {filepath}")
        print("Please download from: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection")
        return None, None, None, None
    
    # Convert to numpy arrays
    headlines = np.array(headlines)
    labels = np.array(labels)
    
    # Print dataset statistics
    print(f"Total samples: {len(labels)}")
    print(f"Sarcastic (1): {sum(labels)} ({sum(labels)/len(labels)*100:.2f}%)")
    print(f"Non-sarcastic (0): {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.2f}%)")
    
    # Create indices and shuffle
    indices = np.random.permutation(len(labels))
    headlines = headlines[indices]
    labels = labels[indices]
    
    # Split into train, validation, and test sets
    train_size = int(len(labels) * (1 - test_size - val_size))
    val_size_samples = int(len(labels) * val_size)
    
    train_headlines = headlines[:train_size]
    train_labels = labels[:train_size]
    
    val_headlines = headlines[train_size:train_size + val_size_samples]
    val_labels = labels[train_size:train_size + val_size_samples]
    
    test_headlines = headlines[train_size + val_size_samples:]
    test_labels = labels[train_size + val_size_samples:]
    
    print(f"\nTrain set: {len(train_labels)}")
    print(f"Validation set: {len(val_labels)}")
    print(f"Test set: {len(test_labels)}")
    
    return (train_headlines, train_labels), (val_headlines, val_labels), (test_headlines, test_labels), labels


# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

class SarcasmDetectionModel(nn.Module):
    """
    BERT-based model for sarcasm detection.
    Uses pre-trained BERT encoder with a classification head.
    """
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super(SarcasmDetectionModel, self).__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters (we'll fine-tune the last layer)
        # Set to False if you want to fine-tune entire BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze last layer of BERT for fine-tuning
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        
        bert_hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(bert_hidden_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT and classification head.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # Get BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract [CLS] token representation (first token)
        cls_output = outputs.pooler_output  # (batch_size, 768)
        
        # Pass through classification head
        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


# ============================================================================
# 3. TRAINING UTILITIES
# ============================================================================

class Trainer:
    """
    Trainer class for model training and evaluation.
    """
    def __init__(self, model, train_loader, val_loader, test_loader, device, learning_rate=2e-5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self, data_loader, phase='Validation'):
        """Evaluate model on given dataset."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=phase, leave=False)
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'preds': all_preds,
            'labels': all_labels
        }
    
    def train(self, epochs=5, early_stopping_patience=3):
        """
        Train model for specified number of epochs with early stopping.
        
        Args:
            epochs: Number of training epochs
            early_stopping_patience: Number of epochs without improvement before stopping
        """
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_results = self.evaluate(self.val_loader, 'Validation')
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_accuracy'].append(val_results['accuracy'])
            self.history['val_f1'].append(val_results['f1'])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f} | Accuracy: {val_results['accuracy']:.4f} | F1: {val_results['f1']:.4f}")
            
            # Early stopping
            if val_results['f1'] > best_val_f1:
                best_val_f1 = val_results['f1']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_sarcasm_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_sarcasm_model.pt'))
        
        # Test on test set
        print("\n" + "="*50)
        print("FINAL TEST SET RESULTS")
        print("="*50)
        test_results = self.evaluate(self.test_loader, 'Testing')
        print(f"Test Loss: {test_results['loss']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test Precision: {test_results['precision']:.4f}")
        print(f"Test Recall: {test_results['recall']:.4f}")
        print(f"Test F1-Score: {test_results['f1']:.4f}")
        
        return test_results


# ============================================================================
# 4. VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics
    axes[1].plot(history['val_accuracy'], label='Accuracy', marker='o')
    axes[1].plot(history['val_f1'], label='F1-Score', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Saved training history plot to training_history.png")
    plt.show()


def plot_confusion_matrix(labels, preds, title='Confusion Matrix'):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"Saved {title.lower()} plot to {title.lower().replace(' ', '_')}.png")
    plt.show()


# ============================================================================
# 5. INFERENCE
# ============================================================================

class SarcasmPredictor:
    """
    Inference class for making predictions on new headlines.
    """
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = SarcasmDetectionModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def predict(self, headline, return_confidence=True):
        """
        Predict sarcasm for a given headline.
        
        Args:
            headline: News headline string
            return_confidence: Whether to return confidence scores
        
        Returns:
            Dictionary with prediction and confidence
        """
        encoding = self.tokenizer.encode_plus(
            headline,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
        
        result = {
            'headline': headline,
            'prediction': 'SARCASTIC' if pred_class == 1 else 'NON-SARCASTIC',
            'predicted_class': pred_class
        }
        
        if return_confidence:
            result['confidence_non_sarcastic'] = probabilities[0, 0].item()
            result['confidence_sarcastic'] = probabilities[0, 1].item()
        
        return result


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load and prepare data
    print("Loading dataset...")
    data_splits = load_and_prepare_data('/Users/nishantdoss/.cache/kagglehub/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection/versions/2/Sarcasm_Headlines_Dataset_v2.json')
    
    if data_splits[0] is not None:
        (train_headlines, train_labels), (val_headlines, val_labels), (test_headlines, test_labels), all_labels = data_splits
        
        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Create datasets
        print("\nPreparing datasets...")
        train_dataset = SarcasmDataset(train_headlines, train_labels, tokenizer)
        val_dataset = SarcasmDataset(val_headlines, val_labels, tokenizer)
        test_dataset = SarcasmDataset(test_headlines, test_labels, tokenizer)
        
        # Create dataloaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        print("Initializing model...")
        model = SarcasmDetectionModel(dropout=0.3)
        model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train model
        print("\nStarting training...")
        trainer = Trainer(model, train_loader, val_loader, test_loader, device, learning_rate=2e-5)
        test_results = trainer.train(epochs=5, early_stopping_patience=3)
        
        # Visualize results
        print("\nGenerating visualizations...")
        plot_training_history(trainer.history)
        plot_confusion_matrix(test_results['labels'], test_results['preds'], 'Test Set Confusion Matrix')
        
        # Example inference
        print("\n" + "="*50)
        print("EXAMPLE PREDICTIONS")
        print("="*50)
        predictor = SarcasmPredictor('best_sarcasm_model.pt', device)
        
        test_headlines_examples = [
            "Trump Announces He's Running For President Again",
            "New Study Finds That Sleeping More Actually Makes You More Tired",
            "Local Man Successfully Completes Entire Day Without Checking Phone",
            "Congress Passes Bill Requiring Birds To Have Licenses"
        ]
        
        for headline in test_headlines_examples:
            result = predictor.predict(headline)
            print(f"\nHeadline: {result['headline']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence (Sarcastic): {result['confidence_sarcastic']:.4f}")
            print(f"Confidence (Non-Sarcastic): {result['confidence_non_sarcastic']:.4f}")