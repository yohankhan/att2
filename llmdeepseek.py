# ===============================
# INSTALL DEPENDENCIES
# ===============================
# !pip install torch tqdm sentencepiece wandb ninja flash-attn --no-build-isolation
# !pip install datasets accelerate deepspeed

# ===============================
# IMPORTS
# ===============================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import math
from tqdm import tqdm
import numpy as np
import wandb
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import warnings

# Try to import optimized dependencies
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("FlashAttention not available. Using standard attention.")

try:
    import torch._dynamo
    TORCH_COMPILE_AVAILABLE = True
except ImportError:
    TORCH_COMPILE_AVAILABLE = False

# ===============================
# CONFIGURATION (Class-based for better organization)
# ===============================
class TrainingConfig:
    # Data
    TEXT_FILE = "shakespeare_clean.txt"
    TOKENIZER_MODEL = "sp_unigram_25k.model"
    
    # Model Architecture
    VOCAB_SIZE = 25000 + 3  # +3 for PAD=0, BOS=1, EOS=2
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    BLOCK_SIZE = 2048  # Increased context length
    
    # Model Dimensions
    EMBED_DIM = 1024  # Increased embedding size
    NUM_HEADS = 16    # Increased number of heads
    NUM_LAYERS = 12   # More layers for deeper model
    FF_DIM = 4096     # Larger feedforward dimension
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4  # Simulate larger batch size
    NUM_EPOCHS = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 1000
    MAX_GRAD_NORM = 1.0  # Gradient clipping
    
    # Optimization
    USE_FUSED_ADAM = True
    USE_FLASH_ATTENTION = FLASH_ATTN_AVAILABLE
    USE_TORCH_COMPILE = TORCH_COMPILE_AVAILABLE
    
    # Checkpointing
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_EVERY = 1000  # Save checkpoint every N steps
    
    # Evaluation
    EVAL_EVERY = 500  # Evaluate every N steps
    GENERATION_EVERY = 1000  # Generate samples every N steps
    
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PRECISION = "bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    USE_DEEPSPEED = False  # Set to True if using DeepSpeed
    
    # Logging
    USE_WANDB = True
    WANDB_PROJECT = "llm-training"
    WANDB_RUN_NAME = f"llm-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

config = TrainingConfig()

# ===============================
# SETUP & LOGGING
# ===============================
# Create checkpoint directory
Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)

# Initialize wandb
if config.USE_WANDB:
    wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config=vars(config)
    )

# ===============================
# STEP 1: TOKENIZER
# ===============================
sp_tokenizer = spm.SentencePieceProcessor(model_file=config.TOKENIZER_MODEL)

def encode_line_safe(line):
    """Encode text with proper special token handling"""
    ids = sp_tokenizer.encode(line, out_type=int)
    # Ensure we don't exceed vocabulary size
    ids = [min(id + 3, config.VOCAB_SIZE - 1) for id in ids]
    return [config.BOS_ID] + ids + [config.EOS_ID]

# ===============================
# STEP 2: OPTIMIZED DATASET & DATALOADER
# ===============================
class EfficientTextDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.block_size = block_size
        self.data = []
        
        # Memory-efficient line reading
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading dataset"):
                line = line.strip()
                if line:
                    token_ids = encode_line_safe(line)
                    if 1 < len(token_ids) <= block_size:
                        self.data.append(token_ids)
        
        # Pre-allocate tensors for efficiency
        self.tensors = []
        for tokens in self.data:
            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)
            self.tensors.append((x, y))
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return self.tensors[idx]

# Dynamic padding collate function
def dynamic_padding_collate(batch):
    x_batch, y_batch = zip(*batch)
    
    # Find max length in this batch (more efficient than fixed padding)
    max_len = max(len(x) for x in x_batch)
    
    # Pad sequences to max length in batch
    x_padded = torch.full((len(batch), max_len), config.PAD_ID, dtype=torch.long)
    y_padded = torch.full((len(batch), max_len), config.PAD_ID, dtype=torch.long)
    
    for i, (x, y) in enumerate(zip(x_batch, y_batch)):
        x_padded[i, :len(x)] = x
        y_padded[i, :len(y)] = y
    
    return x_padded, y_padded

# Create datasets and dataloaders
dataset = EfficientTextDataset(config.TEXT_FILE, config.BLOCK_SIZE)
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=config.BATCH_SIZE, 
    shuffle=True, 
    collate_fn=dynamic_padding_collate,
    pin_memory=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=config.BATCH_SIZE, 
    collate_fn=dynamic_padding_collate,
    pin_memory=True,
    num_workers=2
)

# ===============================
# STEP 3: ADVANCED TRANSFORMER MODEL
# ===============================
class RMSNorm(nn.Module):
    """Root Mean Square Normalization - more stable than LayerNorm"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    """SwiGLU activation function - better than GELU for LLMs"""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) - better positional encoding"""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x, seq_len: int):
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class EfficientAttention(nn.Module):
    """Optimized attention with FlashAttention if available"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary positional embeddings
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Use FlashAttention if available, otherwise fallback
        if config.USE_FLASH_ATTENTION and FLASH_ATTN_AVAILABLE:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask, dropout_p=config.DROPOUT if self.training else 0.0
                )
        else:
            # Standard attention
            attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = attn_weights @ v
        
        # Combine heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.o_proj(attn_output)

class TransformerBlock(nn.Module):
    """Modern Transformer block with RMSNorm and SwiGLU"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = EfficientAttention(embed_dim, num_heads)
        
        self.ff_norm = RMSNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim * 2),  # *2 for SwiGLU
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, embed_dim)  # SwiGLU reduces dimension
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        # Attention sub-layer
        attn_output = self.attn(self.attn_norm(x), attention_mask)
        x = x + self.dropout(attn_output)
        
        # Feedforward sub-layer
        ff_output = self.ff(self.ff_norm(x))
        x = x + self.dropout(ff_output)
        
        return x

class EnterpriseTransformerLM(nn.Module):
    """Modern Transformer with industry best practices"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, block_size, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        
        # Token and positional embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization and output head
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights (input and output embeddings)
        self.head.weight = self.token_emb.weight
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
        # Apply scaled initialization to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, attention_mask=None):
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (x != self.pad_id).unsqueeze(1).unsqueeze(2)
        
        # Convert to attention mask format
        attention_mask = attention_mask.to(dtype=x.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min
        
        # Forward pass through embeddings and layers
        x = self.token_emb(x)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        logits = self.head(x)
        
        return logits

# Create model
model = EnterpriseTransformerLM(
    config.VOCAB_SIZE, config.EMBED_DIM, config.NUM_HEADS, 
    config.NUM_LAYERS, config.FF_DIM, config.BLOCK_SIZE, 
    config.DROPOUT, config.PAD_ID
).to(config.DEVICE)

# Compile model if available
if config.USE_TORCH_COMPILE:
    model = torch.compile(model)

# Print model stats
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params / 1e6:.2f}M")

# ===============================
# STEP 4: OPTIMIZED TRAINING SETUP
# ===============================
# Optimizer with weight decay filtering
def get_optimizer(model, lr, weight_decay):
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Don't decay 1D parameters (biases, layernorm weights)
        if param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    # Use fused AdamW if available
    extra_args = {}
    if config.USE_FUSED_ADAM and config.DEVICE == "cuda":
        try:
            from torch.optim import AdamW
            extra_args["fused"] = True
        except ImportError:
            pass
    
    return AdamW(optim_groups, lr=lr, **extra_args)

optimizer = get_optimizer(model, config.LEARNING_RATE, config.WEIGHT_DECAY)

# Learning rate scheduler
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Calculate total training steps
total_steps = len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
scheduler = get_lr_scheduler(optimizer, config.WARMUP_STEPS, total_steps)

# Loss function
criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID)

# Mixed precision setup
scaler = torch.amp.GradScaler() if config.DEVICE == "cuda" and config.PRECISION == "fp16" else None

# ===============================
# STEP 5: TRAINING LOOP WITH ADVANCED FEATURES
# ===============================
def evaluate(model, val_loader):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc="Evaluating"):
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            
            # Create attention mask
            attention_mask = (xb != config.PAD_ID).unsqueeze(1).unsqueeze(2)
            
            with torch.amp.autocast(device_type=config.DEVICE, enabled=config.PRECISION != "fp32"):
                logits = model(xb, attention_mask)
                loss = criterion(logits.view(-1, config.VOCAB_SIZE), yb.view(-1))
            
            # Calculate perplexity
            total_loss += loss.item() * (yb != config.PAD_ID).sum().item()
            total_tokens += (yb != config.PAD_ID).sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity

def save_checkpoint(step, model, optimizer, scheduler, scaler, loss):
    """Save training checkpoint"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Keep only the latest 3 checkpoints
    checkpoints = sorted(Path(config.CHECKPOINT_DIR).glob("checkpoint_step_*.pt"))
    for old_checkpoint in checkpoints[:-3]:
        old_checkpoint.unlink()

def generate_text(model, tokenizer, prompt, max_len=100, temperature=0.8, top_k=50, top_p=0.9):
    """Advanced text generation with multiple sampling strategies"""
    model.eval()
    ids = encode_line_safe(prompt)
    
    # Truncate if too long
    if len(ids) > config.BLOCK_SIZE:
        ids = ids[-config.BLOCK_SIZE:]
    
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        for _ in range(max_len):
            # Create attention mask
            attention_mask = (x != config.PAD_ID).unsqueeze(1).unsqueeze(2)
            
            # Get model predictions
            logits = model(x, attention_mask)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            x = torch.cat([x, next_id], dim=1)
            
            # Stop if we generate EOS
            if next_id.item() == config.EOS_ID:
                break
    
    # Decode and return
    decoded = tokenizer.decode([i-3 for i in x[0].cpu().numpy() if i > 2])
    return decoded

# Training loop
global_step = 0
best_val_loss = float('inf')

for epoch in range(config.NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_tokens = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    
    for batch_idx, (xb, yb) in enumerate(progress_bar):
        xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
        
        # Create attention mask
        attention_mask = (xb != config.PAD_ID).unsqueeze(1).unsqueeze(2)
        
        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=config.DEVICE, enabled=config.PRECISION != "fp32"):
            logits = model(xb, attention_mask)
            loss = criterion(logits.view(-1, config.VOCAB_SIZE), yb.view(-1))
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights only after accumulating gradients
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            
            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Log metrics
            current_lr = scheduler.get_last_lr()[0]
            if config.USE_WANDB:
                wandb.log({
                    "train_loss": loss.item() * config.GRADIENT_ACCUMULATION_STEPS,
                    "learning_rate": current_lr,
                    "step": global_step,
                    "epoch": epoch
                })
        
        # Update progress bar
        epoch_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS * (yb != config.PAD_ID).sum().item()
        epoch_tokens += (yb != config.PAD_ID).sum().item()
        
        progress_bar.set_postfix({
            "loss": f"{loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}",
            "lr": f"{current_lr:.2e}"
        })
        
        # Evaluation
        if global_step % config.EVAL_EVERY == 0:
            val_loss, perplexity = evaluate(model, val_loader)
            
            if config.USE_WANDB:
                wandb.log({
                    "val_loss": val_loss,
                    "perplexity": perplexity,
                    "step": global_step
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(global_step, model, optimizer, scheduler, scaler, val_loss)
            
            model.train()
        
        # Text generation
        if global_step % config.GENERATION_EVERY == 0:
            prompt = "To be, or not to be"
            generated = generate_text(model, sp_tokenizer, prompt)
            
            if config.USE_WANDB:
                wandb.log({
                    "generated_text": wandb.Html(f"<pre>{generated}</pre>"),
                    "step": global_step
                })
            
            print(f"\nGenerated text (step {global_step}):\n{generated}\n")
        
        # Checkpoint saving
        if global_step % config.CHECKPOINT_EVERY == 0:
            save_checkpoint(global_step, model, optimizer, scheduler, scaler, loss.item())
    
    # End of epoch
    avg_epoch_loss = epoch_loss / epoch_tokens
    print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

# Final evaluation
print("Training completed. Running final evaluation...")
final_val_loss, final_perplexity = evaluate(model, val_loader)
print(f"Final validation loss: {final_val_loss:.4f}, Perplexity: {final_perplexity:.2f}")

# Save final model
save_checkpoint(global_step, model, optimizer, scheduler, scaler, final_val_loss)

# Test generation
prompt = "To be, or not to be"
print("\nFinal generated text:\n", generate_text(model, sp_tokenizer, prompt))

if config.USE_WANDB:
    wandb.finish()
