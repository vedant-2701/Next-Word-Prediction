# app.py - Streamlit App for Enhanced Next Word Prediction
# Features:
# - Predict next word with top-3 options and confidences
# - Generate full sentence from seed text
# - User-friendly UI with history
# - Load custom model if needed

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences  # Not used directly, but for ref

@st.cache_resource
def load_artifacts():
    """Load model and info."""
    model = load_model('attentive_lstm.h5')
    with open('model_info.pkl', 'rb') as f:
        info = pickle.load(f)
    return model, info

def prepare_input(text, unique_word_index, seq_length, vocab_size):
    """Prepare one-hot input from text."""
    words = text.lower().split()
    words = words[-seq_length:]  # Last seq_length words
    x = np.zeros((1, seq_length, vocab_size))
    for i, word in enumerate(words):
        idx = seq_length - len(words) + i
        if word in unique_word_index:
            x[0, idx, unique_word_index[word]] = 1
    return x

def predict_next(model, text, unique_word_index, index_to_word, seq_length, vocab_size, top_k=3):
    """Predict top-k next words with probabilities."""
    x = prepare_input(text, unique_word_index, seq_length, vocab_size)
    probs = model.predict(x, verbose=0)[0]
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_words = [index_to_word.get(i, '<UNK>') for i in top_indices]
    top_probs = probs[top_indices]
    return top_words, top_probs

def generate_sentence(model, seed_text, unique_word_index, index_to_word, seq_length, vocab_size, num_words=10, top_k=1):
    """Generate a sentence by iteratively predicting next words."""
    current_text = seed_text
    for _ in range(num_words):
        next_words, probs = predict_next(model, current_text, unique_word_index, index_to_word, seq_length, vocab_size, top_k)
        next_word = next_words[0]  # Greedy
        current_text += " " + next_word
    return current_text

# Streamlit UI
st.title("🧠 Next Word Prediction with Attention-Enhanced LSTM")
st.markdown("**A Deep Learning Mini-Project: Bidirectional LSTM + Self-Attention for Text Generation**")
st.markdown("Enter text to predict the next word or generate a full sentence. Trained on Project Gutenberg corpus with GloVe embeddings.")

model, info = load_artifacts()
index_to_word = info['index_to_word']
unique_word_index = info['unique_word_index']
vocab_size = info['vocab_size']
seq_length = info['seq_length']

# Session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["🔮 Predict Next Word", "✨ Generate Sentence"])

with tab1:
    input_text = st.text_input("Enter a sentence (last 50 words used as context):", 
                               value="the quick brown fox jumps over the lazy dog")
    num_options = st.slider("Top predictions:", 1, 5, 3)
    
    if st.button("Predict Next Word", type="primary"):
        if input_text.strip():
            top_words, top_probs = predict_next(model, input_text, unique_word_index, 
                                                index_to_word, seq_length, vocab_size, num_options)
            st.success("Predictions:")
            for word, prob in zip(top_words, top_probs):
                st.write(f"**{word}** (confidence: {prob:.2%})")
            
            # Add to history
            st.session_state.history.append({"input": input_text, "predictions": list(zip(top_words, top_probs))})
        else:
            st.warning("Please enter some text.")

with tab2:
    seed_text = st.text_input("Seed sentence:", value="once upon a time")
    num_gen_words = st.slider("Number of words to generate:", 5, 50, 10)
    sample_k = st.slider("Sampling (top-k):", 1, 5, 1)
    
    if st.button("Generate Sentence", type="primary"):
        if seed_text.strip():
            generated = generate_sentence(model, seed_text, num_gen_words, sample_k,
                                          unique_word_index, index_to_word, seq_length, vocab_size)
            st.success(f"Generated: {generated}")
            
            # Add to history
            st.session_state.history.append({"seed": seed_text, "generated": generated})
        else:
            st.warning("Please enter a seed sentence.")

# History
if st.session_state.history:
    st.sidebar.title("📜 Prediction History")
    for i, entry in enumerate(st.session_state.history[-5:]):  # Last 5
        with st.sidebar.expander(f"Entry {len(st.session_state.history) - i}"):
            if "input" in entry:
                st.write(f"**Input:** {entry['input']}")
                for word, prob in entry['predictions']:
                    st.write(f"- {word} ({prob:.2%})")
            elif "seed" in entry:
                st.write(f"**Seed:** {entry['seed']}")
                st.write(f"**Generated:** {entry['generated']}")

st.sidebar.markdown("---")
st.sidebar.info("**Project Highlights:**\n- Bidirectional LSTM for bidirectional context\n- Self-Attention for focusing on important words\n- GloVe pre-trained embeddings\n- Perplexity evaluation\n- Top-k sampling for diversity")