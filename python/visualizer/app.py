import streamlit as st
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Model

# 1. Load Model (Cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
    return tokenizer, model

tokenizer, model = load_model()

# 2. Website Interface
st.title("GPT-2 Attention Visualizer")
text = st.text_input("Write a sentence:", "The quick brown fox jumps.")

if text:
    # 3. Process Input
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # 4. Run Model
    outputs = model(**inputs)
    attentions = outputs.attentions  # Tuple of (batch, num_heads, seq_len, seq_len)

    # 5. Select Layer and Head
    layer = st.slider("Select Layer", 0, 11, 11)
    head = st.slider("Select Head", 0, 11, 0)
    
    # Extract specific attention matrix
    # [0] for batch index, [head] for head index
    attn_matrix = attentions[layer][0, head].detach().numpy()

    # 6. Visualize
    st.write(f"Showing attention for Layer {layer}, Head {head}")
    
    # Create a DataFrame for better labeling
    df = pd.DataFrame(attn_matrix, index=tokens, columns=tokens)
    
    # Plot using Seaborn/Matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    st.pyplot(fig)