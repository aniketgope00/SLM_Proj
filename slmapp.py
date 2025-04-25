import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import glob
import random
import os
import json
import pickle
from datetime import datetime

# Global variables - declare at the module level
n_embd = None
n_head = None
n_layer = None
dropout = None
block_size = None
batch_size = None

# Device config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def update_global_params(params):
    """Update global parameters safely"""
    global n_embd, n_head, n_layer, dropout, block_size, batch_size
    n_embd = params['n_embd']
    n_head = params['n_head']
    n_layer = params['n_layer']
    dropout = params['dropout']
    block_size = params['block_size']
    batch_size = params['batch_size']

# Initialize with default hyperparameters
def get_hyperparams():
    return {
        'batch_size': 32,
        'block_size': 128,
        'learning_rate': 3e-4,
        'eval_iters': 50,
        'n_embd': 384,
        'n_head': 4,
        'n_layer': 4,
        'dropout': 0.2
    }

# Initialize global parameters with defaults
update_global_params(get_hyperparams())

# Data loading and processing
def read_files(pattern):
    all_text = ""
    for filename in glob.glob(pattern):
        with open(filename, 'r', encoding='utf-8') as f:
            all_text += f.read()
    return all_text

def prepare_data(text, encode):
    if len(text) < 1000:  # Minimum text length check
        raise ValueError("Not enough text data. Please provide more text (minimum 1000 characters).")
        
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.8 * len(data))  # 80% for training
    return data[:n], data[n:]

def get_batch(data, block_size, batch_size):
    # Ensure we have enough data for the block size
    if len(data) <= block_size:
        raise ValueError(f"Data length ({len(data)}) must be greater than block size ({block_size})")
    
    # Calculate valid range for random sampling
    max_start_idx = len(data) - block_size
    if max_start_idx < 1:
        raise ValueError("Not enough data for generating batches")
        
    ix = torch.randint(0, max_start_idx, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Model definition classes
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens, temperature=1.0):
        # Fixed function with proper temperature handling
        for _ in range(max_new_tokens):
            # Get the sequence limited by block size
            index_cond = index[:, -self.block_size:]
            # Forward pass
            logits, _ = self(index_cond)
            # Focus on the last time step
            logits = logits[:, -1, :] / temperature  # Apply temperature
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)
        return index

def create_model(vocab_size, params):
    """Create a model instance with the given parameters"""
    return GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=params['n_embd'],
        n_head=params['n_head'],
        n_layer=params['n_layer'],
        block_size=params['block_size'],
        dropout=params['dropout']
    ).to(device)

# Fixed encode/decode functions that handle errors better
def create_encode_decode_functions(string_to_int, int_to_string):
    """Create encode and decode functions with proper error handling"""
    # Handle unknown characters by using a default value (last index in vocab)
    default_idx = len(string_to_int)
    
    def encode(s):
        return [string_to_int.get(c, default_idx-1) for c in s]
    
    def decode(l):
        return ''.join([int_to_string.get(i, '') for i in l])
    
    # Attach vocabulary dictionaries for reference
    encode.keywords = {'string_to_int': string_to_int}
    decode.keywords = {'int_to_string': int_to_string}
    
    return encode, decode

def process_uploaded_files(uploaded_files):
    """Process uploaded markdown files and combine their content"""
    combined_text = ""
    for uploaded_file in uploaded_files:
        try:
            # Read the content of the uploaded file
            content = uploaded_file.read().decode('utf-8')
            combined_text += content + "\n\n"
            st.success(f"Successfully processed {uploaded_file.name}")
        except Exception as e:
            st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
    return combined_text

def generate_with_model(model, prompt, encode_fn, decode_fn, max_tokens, temperature, block_size):
    """Fixed generate_with_model function"""
    try:
        if not model:
            raise ValueError("Model not properly initialized")
            
        # Ensure model is in eval mode
        model.eval()
        
        # Encode the prompt
        try:
            encoded_prompt = encode_fn(prompt)
            if not encoded_prompt:
                raise ValueError("Failed to encode prompt - check vocabulary")
                
            # Truncate if necessary to fit block size
            if len(encoded_prompt) > block_size:
                encoded_prompt = encoded_prompt[-block_size:]
            
            # Create tensor and move to device
            context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate text
            with torch.no_grad():
                output = model.generate(
                    context, 
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
            
            # Decode the generated text
            full_output = output[0].tolist()
            
            # Split into prompt and generated text
            prompt_length = len(encoded_prompt)
            generated_tokens = full_output[prompt_length:]
            
            # Decode
            original_prompt = decode_fn(full_output[:prompt_length])
            generated_text = decode_fn(generated_tokens)
            
            return original_prompt, generated_text
            
        except KeyError:
            raise ValueError("Encoding error: vocabulary mismatch. The prompt contains characters not seen during training.")
    
    except Exception as e:
        raise Exception(f"Generation error: {str(e)}")

def save_model_state(model, vocab_data, hyperparams, save_dir="saved_models"):
    """Save model state, vocabulary, and hyperparameters"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create timestamp for the save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_dir, f"model_{timestamp}.pt")
        vocab_path = os.path.join(save_dir, f"vocab_{timestamp}.pkl")
        params_path = os.path.join(save_dir, f"params_{timestamp}.json")
        
        # Save model state
        torch.save(model.state_dict(), model_path)
        
        # Save vocabulary data
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
            
        # Save hyperparameters
        with open(params_path, 'w') as f:
            json.dump(hyperparams, f)
            
        return timestamp
        
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}")

def load_model_state(timestamp, save_dir="saved_models"):
    """Load model state, vocabulary, and hyperparameters"""
    try:
        model_path = os.path.join(save_dir, f"model_{timestamp}.pt")
        vocab_path = os.path.join(save_dir, f"vocab_{timestamp}.pkl")
        params_path = os.path.join(save_dir, f"params_{timestamp}.json")
        
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(vocab_path) or not os.path.exists(params_path):
            raise FileNotFoundError(f"One or more required files for model {timestamp} not found")
        
        # Load hyperparameters
        with open(params_path, 'r') as f:
            hyperparams = json.load(f)
            
        # Load vocabulary data
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            
        # Update global parameters
        update_global_params(hyperparams)
            
        # Create and load model
        model = create_model(len(vocab_data['string_to_int']), hyperparams)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set model to evaluation mode
        
        return model, vocab_data, hyperparams
        
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def get_saved_models(save_dir="saved_models"):
    """Get list of saved model timestamps"""
    try:
        if not os.path.exists(save_dir):
            return []
            
        timestamps = []
        for file in os.listdir(save_dir):
            if file.startswith("model_") and file.endswith(".pt"):
                timestamps.append(file[6:-3])  # Extract timestamp from filename
        return sorted(timestamps, reverse=True)  # Most recent first
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []

def main():
    st.title("Train Your Own TinyGPT Model")
    
    # Initialize session state with more comprehensive defaults
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    if 'string_to_int' not in st.session_state:
        st.session_state.string_to_int = {}
    if 'int_to_string' not in st.session_state:
        st.session_state.int_to_string = {}
    if 'encode_fn' not in st.session_state:
        st.session_state.encode_fn = None
    if 'decode_fn' not in st.session_state:
        st.session_state.decode_fn = None
    if 'current_model_timestamp' not in st.session_state:
        st.session_state.current_model_timestamp = None
    if 'block_size' not in st.session_state:
        st.session_state.block_size = block_size  # Use global block_size
    if 'vocab_size' not in st.session_state:
        st.session_state.vocab_size = 0
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Model Selection")
        
        # Load saved model section
        saved_models = get_saved_models()
        if saved_models:
            selected_model = st.selectbox(
                "Load saved model",
                ["None"] + saved_models,
                format_func=lambda x: f"Model from {x}" if x != "None" else "Train new model"
            )
            
            if selected_model != "None" and selected_model != st.session_state.current_model_timestamp:
                try:
                    with st.spinner("Loading saved model..."):
                        model, vocab_data, loaded_hp = load_model_state(selected_model)
                        
                        # Update global parameters
                        update_global_params(loaded_hp)
                        
                        # Update session state
                        st.session_state.model = model
                        st.session_state.string_to_int = vocab_data['string_to_int']
                        st.session_state.int_to_string = vocab_data['int_to_string']
                        st.session_state.vocab_size = len(vocab_data['string_to_int'])
                        st.session_state.is_trained = True
                        st.session_state.current_model_timestamp = selected_model
                        st.session_state.block_size = loaded_hp['block_size']
                        
                        # Create encode/decode functions
                        encode_fn, decode_fn = create_encode_decode_functions(
                            st.session_state.string_to_int,
                            st.session_state.int_to_string
                        )
                        
                        st.session_state.encode_fn = encode_fn
                        st.session_state.decode_fn = decode_fn
                        
                        st.success(f"Model loaded successfully! Block size: {st.session_state.block_size}")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
            elif selected_model == "None":
                # Reset to default parameters when "None" is selected
                update_global_params(get_hyperparams())
                st.session_state.model = None
                st.session_state.is_trained = False
                st.session_state.current_model_timestamp = None
                st.session_state.block_size = block_size
                st.session_state.string_to_int = {}
                st.session_state.int_to_string = {}
                st.session_state.vocab_size = 0
                st.session_state.encode_fn = None
                st.session_state.decode_fn = None
        
        st.header("Training Configuration")
        
        # Get default hyperparameters for UI
        hp = get_hyperparams()
        
        # File upload section
        st.subheader("Upload Training Data")
        uploaded_files = st.file_uploader(
            "Upload markdown files for training", 
            type=['md', 'txt'], 
            accept_multiple_files=True,
            help="Upload one or more markdown files for training"
        )
        
        # Training parameters
        st.subheader("Training Parameters")
        max_iters = st.number_input("Max Training Iterations", min_value=100, max_value=10000, value=1000, step=100)
        
        # Add hyperparameter tuning in expandable section
        with st.expander("Advanced Settings"):
            custom_n_embd = st.slider("Embedding Dimension", min_value=64, max_value=512, value=hp['n_embd'])
            custom_n_head = st.slider("Number of Attention Heads", min_value=1, max_value=8, value=hp['n_head'])
            custom_n_layer = st.slider("Number of Layers", min_value=1, max_value=8, value=hp['n_layer'])
            learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=hp['learning_rate'], format="%.5f")
            
        train_button = st.button("Start Training")

    if train_button:
        if not uploaded_files:
            st.error("Please upload at least one markdown file for training.")
            return
            
        try:
            with st.spinner('Processing uploaded files...'):
                # Process uploaded files
                text = process_uploaded_files(uploaded_files)
                if not text:
                    st.error("No valid content found in uploaded files.")
                    return
                
                if len(text) < 1000:
                    st.error("Not enough text data. Please provide more text (minimum 1000 characters).")
                    return
                    
                # Fix block size to 100 for training
                suggested_block_size = min(100, len(text) // 2)  # Make sure block size isn't too large
                
                # Prepare training parameters
                train_params = {
                    'n_embd': custom_n_embd,
                    'n_head': custom_n_head,
                    'n_layer': custom_n_layer,
                    'dropout': hp['dropout'],
                    'block_size': suggested_block_size,
                    'batch_size': min(hp['batch_size'], len(text) // (suggested_block_size * 2))
                }
                
                # Update global parameters
                update_global_params(train_params)
                
                # Update session state block size
                st.session_state.block_size = block_size
                
                st.info(f"Using block size of {block_size} tokens for training")
                
                # Create vocabulary
                chars = sorted(list(set(text)))
                vocab_size = len(chars)
                string_to_int = {ch: i for i, ch in enumerate(chars)}
                int_to_string = {i: ch for i, ch in enumerate(chars)}
                
                # Store vocabulary in session state
                st.session_state.string_to_int = string_to_int
                st.session_state.int_to_string = int_to_string
                st.session_state.vocab_size = vocab_size
                
                # Create encode/decode functions
                encode_fn, decode_fn = create_encode_decode_functions(string_to_int, int_to_string)
                st.session_state.encode_fn = encode_fn
                st.session_state.decode_fn = decode_fn
                
                # Prepare vocabulary data for saving
                vocab_data = {
                    'vocab_size': vocab_size,
                    'string_to_int': string_to_int,
                    'int_to_string': int_to_string
                }
                
                # Prepare hyperparameters for saving
                hyperparams = {
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'dropout': dropout,
                    'block_size': block_size,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }

                train_data, val_data = prepare_data(text, encode_fn)

            with st.spinner('Initializing model...'):
                model = create_model(vocab_size, train_params)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

            st.info("Starting training...")

            # Training progress containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_chart = st.empty()
            training_losses = []
            val_losses = []

            # Training loop with improved error handling
            for iter in range(max_iters):
                try:
                    if iter % hp['eval_iters'] == 0:
                        with torch.no_grad():
                            model.eval()
                            losses = []
                            for _ in range(5):
                                xb, yb = get_batch(val_data, block_size, batch_size)
                                _, loss = model(xb, yb)
                                losses.append(loss.item())
                            avg_val_loss = sum(losses) / len(losses)
                            val_losses.append(avg_val_loss)
                            status_text.text(f"Iteration {iter}/{max_iters} - Validation Loss: {avg_val_loss:.4f}")
                            model.train()

                    # Training step
                    xb, yb = get_batch(train_data, block_size, batch_size)
                    _, loss = model(xb, yb)
                    training_losses.append(loss.item())
                    
                    # Update progress
                    progress = (iter + 1) / max_iters
                    progress_bar.progress(progress)

                    # Update loss chart
                    if iter % 10 == 0:
                        loss_data = {"iteration": list(range(len(training_losses))), "training_loss": training_losses}
                        loss_chart.line_chart(loss_data)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                except Exception as batch_error:
                    st.error(f"Error during training iteration {iter}: {str(batch_error)}")
                    break

            # After successful training, save the model
            try:
                timestamp = save_model_state(model, vocab_data, hyperparams)
                st.session_state.current_model_timestamp = timestamp
                st.success(f"Model saved successfully! Timestamp: {timestamp}")
            except Exception as e:
                st.warning(f"Model trained successfully but couldn't be saved: {str(e)}")

            # Store the trained model
            st.session_state.model = model
            st.session_state.is_trained = True
            st.success("Training completed!")

        except Exception as e:
            st.error(f"An error occurred during setup: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return

    # Text generation interface
    if st.session_state.is_trained and st.session_state.model is not None:
        st.header("Text Generation")
        
        # Generation settings
        col1, col2 = st.columns(2)
        with col1:
            prompt = st.text_area("Enter your prompt:", "What is linear regression?", height=100)
        with col2:
            gen_max_tokens = st.number_input(
                "Maximum tokens to generate", 
                min_value=10, 
                max_value=1000, 
                value=100,
                help="Maximum number of tokens to generate in the response"
            )
            temperature = st.slider(
                "Temperature", 
                min_value=0.1, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="Higher values make the output more random, lower values make it more deterministic"
            )
        
        if st.button("Generate"):
            # Verify we have everything needed for generation
            if not st.session_state.model:
                st.error("Model not properly loaded. Please train a new model or load a saved one.")
                return
                
            if not st.session_state.block_size:
                st.error("Block size not properly initialized. Please try reloading the model.")
                return
                
            if not st.session_state.string_to_int or not st.session_state.int_to_string:
                st.error("Vocabulary not properly initialized. Please try reloading the model.")
                return
                
            with st.spinner("Generating response..."):
                try:
                    # Recreate encode/decode functions to ensure latest vocabulary
                    encode_fn, decode_fn = create_encode_decode_functions(
                        st.session_state.string_to_int,
                        st.session_state.int_to_string
                    )
                    
                    # Generate text with improved error handling
                    original_prompt, generated_text = generate_with_model(
                        model=st.session_state.model,
                        prompt=prompt,
                        encode_fn=encode_fn,
                        decode_fn=decode_fn,
                        max_tokens=gen_max_tokens,
                        temperature=temperature,
                        block_size=st.session_state.block_size
                    )
                    
                    # Display response in a more structured format
                    st.markdown("### Generated Response")
                    
                    # Show prompt in one color/box
                    st.markdown("**Original Prompt:**")
                    st.info(original_prompt)
                    
                    # Show generated text in another color/box
                    st.markdown("**Generated Text:**")
                    st.success(generated_text)
                    
                except Exception as e:
                    st.error(f"Error during text generation: {str(e)}")
                    st.info("Try adjusting the temperature, reducing the number of tokens, or using a simpler prompt.")
    else:
        st.info("Please upload markdown files and complete training, or load a saved model to start generating text.")

if __name__ == '__main__':
    main()