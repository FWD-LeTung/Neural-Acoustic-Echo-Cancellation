import torch
import numpy as np
import os
from torch import nn
from torch.nn.functional import softmax
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperPreTrainedModel, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

MODEL_PATH = "/content/drive/MyDrive/checkpoint-1000"  # <-- Đảm bảo trỏ đúng thư mục
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

class ResidualBlock(nn.Module):
    """
    Tái tạo lại ResidualBlock dựa trên các lớp phổ biến và log lỗi cũ (net.0, net.1 có weights).
    Cấu trúc dự đoán: Linear -> LayerNorm -> GELU -> Dropout
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)

class LumiSmartTurnV1Model(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        
        self.input_dim = config.d_model
        print(f"Model Input Dimension: {self.input_dim} (Expected: 768)")

        # --- RECONSTRUCT ATTENTION POOLING (Indices 0-6) ---
        # 0 Linear: 768 -> 512
        # 1 LayerNorm
        # 2 Tanh
        # 3 Dropout
        # 4 Linear: 512 -> 256
        # 5 Tanh
        # 6 Linear: 256 -> 1
        self.pool_attention = nn.Sequential(
            nn.Linear(self.input_dim, 512), # 0
            nn.LayerNorm(512),              # 1
            nn.Tanh(),                      # 2
            nn.Dropout(0.1),                # 3
            nn.Linear(512, 256),            # 4
            nn.Tanh(),                      # 5
            nn.Linear(256, 1)               # 6
        )

        # --- RECONSTRUCT CLASSIFIER (Indices 0-12) ---
        # 0 Linear: 768 -> 512
        # 1 LayerNorm
        # 2 GELU
        # 3 Dropout
        # 4 ResidualBlock (512)
        # 5 ResidualBlock (512)
        # 6 Linear: 512 -> 256
        # 7 GELU
        # 8 Dropout
        # 9 Linear: 256 -> 128
        # 10 GELU
        # 11 Dropout
        # 12 Linear: 128 -> 1
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, 512), # 0
            nn.LayerNorm(512),              # 1
            nn.GELU(),                      # 2
            nn.Dropout(0.1),                # 3
            ResidualBlock(512),             # 4
            ResidualBlock(512),             # 5
            nn.Linear(512, 256),            # 6
            nn.GELU(),                      # 7
            nn.Dropout(0.1),                # 8
            nn.Linear(256, 128),            # 9
            nn.GELU(),                      # 10
            nn.Dropout(0.1),                # 11
            nn.Linear(128, 1)               # 12
        )

    def forward(self, input_features, labels=None):
        encoder_outputs = self.encoder(input_features=input_features)
        hidden_states = encoder_outputs.last_hidden_state
        
        # Attention Pooling
        attentions_weights = self.pool_attention(hidden_states)
        attentions_weights = softmax(attentions_weights, dim=1)
        pooled = torch.sum(hidden_states * attentions_weights, dim=1)

        # Classifier
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)
        
        return {"logits": logits, "probs": probs}

def load_qat_checkpoint_clean(model, folder_path):
    safetensors_path = os.path.join(folder_path, "model.safetensors")
    pytorch_bin_path = os.path.join(folder_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(pytorch_bin_path):
        state_dict = torch.load(pytorch_bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model file found in {folder_path}")

    clean_dict = {}
    for k, v in state_dict.items():
        # Lọc rác QAT
        if any(x in k for x in ["fake_quant", "observer", "activation_post_process"]):
            continue
        clean_dict[k] = v

    msg = model.load_state_dict(clean_dict, strict=False)
    print(f"Load Report: {msg}")
    return model

def truncate_audio_to_last_n_seconds(audio_array, n_seconds, sample_rate):
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    return audio_array


def main():
    print("Initializing Model...")
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_PATH)

        if feature_extractor.chunk_length != 8:
            feature_extractor.chunk_length = 8
            feature_extractor.n_samples = 128000 

        config = WhisperConfig.from_pretrained(MODEL_PATH)
        

        if config.d_model != 768:
            print(f"WARNING: Config says d_model={config.d_model}, but training log says 768. Forcing 768.")
            config.d_model = 768
            
        model = LumiSmartTurnV1Model(config)
        model = load_qat_checkpoint_clean(model, MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    print("Loading dataset...")
    dataset = load_dataset("PandaLT/Neural-AEC-Test-No-Noise", split="train")
    dataset = dataset.cast_column("mic", Audio(sampling_rate=16000))

    true_labels = []
    pred_labels = []

    print("Starting Inference Loop...")
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset)):
            audio = sample["mic"]["array"]
            audio = truncate_audio_to_last_n_seconds(audio, 8, 16000)
            
            inputs = feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=128000, 
                truncation=True
            )
            
            input_features = inputs.input_features.to(DEVICE)
            outputs = model(input_features)
            prob = outputs["probs"].item()
            
            # Ground Truth
            label = 1 if sample["endpoint_bool"] else 0
            
            # Prediction
            pred = 1 if prob > 0.5 else 0
            
            true_labels.append(label)
            pred_labels.append(pred)

    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels)

    print("\n" + "="*40)
    print("FINAL VALIDATION RESULTS")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 20)
    print(f"Confusion Matrix:\n{cm}")
    print("="*40)

main()