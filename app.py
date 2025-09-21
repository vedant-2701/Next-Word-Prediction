# # app.py - Streamlit App for Enhanced Next Word Prediction
# # Features:
# # - Predict next word with top-3 options and confidences
# # - Generate full sentence from seed text
# # - User-friendly UI with history
# # - Load custom model if needed

# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import pad_sequences  # Not used directly, but for ref

# @st.cache_resource
# def load_artifacts():
#     """Load model and info."""
#     model = load_model('attentive_lstm.h5')
#     with open('model_info.pkl', 'rb') as f:
#         info = pickle.load(f)
#     return model, info

# def prepare_input(text, unique_word_index, seq_length, vocab_size):
#     """Prepare one-hot input from text."""
#     words = text.lower().split()
#     words = words[-seq_length:]  # Last seq_length words
#     x = np.zeros((1, seq_length, vocab_size))
#     for i, word in enumerate(words):
#         idx = seq_length - len(words) + i
#         if word in unique_word_index:
#             x[0, idx, unique_word_index[word]] = 1
#     return x

# def predict_next(model, text, unique_word_index, index_to_word, seq_length, vocab_size, top_k=3):
#     """Predict top-k next words with probabilities."""
#     x = prepare_input(text, unique_word_index, seq_length, vocab_size)
#     probs = model.predict(x, verbose=0)[0]
#     top_indices = np.argsort(probs)[-top_k:][::-1]
#     top_words = [index_to_word.get(i, '<UNK>') for i in top_indices]
#     top_probs = probs[top_indices]
#     return top_words, top_probs

# def generate_sentence(model, seed_text, unique_word_index, index_to_word, seq_length, vocab_size, num_words=10, top_k=1):
#     """Generate a sentence by iteratively predicting next words."""
#     current_text = seed_text
#     for _ in range(num_words):
#         next_words, probs = predict_next(model, current_text, unique_word_index, index_to_word, seq_length, vocab_size, top_k)
#         next_word = next_words[0]  # Greedy
#         current_text += " " + next_word
#     return current_text

# # Streamlit UI
# st.title("🧠 Next Word Prediction with Attention-Enhanced LSTM")
# st.markdown("**A Deep Learning Mini-Project: Bidirectional LSTM + Self-Attention for Text Generation**")
# st.markdown("Enter text to predict the next word or generate a full sentence. Trained on Project Gutenberg corpus with GloVe embeddings.")

# model, info = load_artifacts()
# index_to_word = info['index_to_word']
# unique_word_index = info['unique_word_index']
# vocab_size = info['vocab_size']
# seq_length = info['seq_length']

# # Session state for history
# if 'history' not in st.session_state:
#     st.session_state.history = []

# tab1, tab2 = st.tabs(["🔮 Predict Next Word", "✨ Generate Sentence"])

# with tab1:
#     input_text = st.text_input("Enter a sentence (last 50 words used as context):", 
#                                value="the quick brown fox jumps over the lazy dog")
#     num_options = st.slider("Top predictions:", 1, 5, 3)
    
#     if st.button("Predict Next Word", type="primary"):
#         if input_text.strip():
#             top_words, top_probs = predict_next(model, input_text, unique_word_index, 
#                                                 index_to_word, seq_length, vocab_size, num_options)
#             st.success("Predictions:")
#             for word, prob in zip(top_words, top_probs):
#                 st.write(f"**{word}** (confidence: {prob:.2%})")
            
#             # Add to history
#             st.session_state.history.append({"input": input_text, "predictions": list(zip(top_words, top_probs))})
#         else:
#             st.warning("Please enter some text.")

# with tab2:
#     seed_text = st.text_input("Seed sentence:", value="once upon a time")
#     num_gen_words = st.slider("Number of words to generate:", 5, 50, 10)
#     sample_k = st.slider("Sampling (top-k):", 1, 5, 1)
    
#     if st.button("Generate Sentence", type="primary"):
#         if seed_text.strip():
#             generated = generate_sentence(model, seed_text, num_gen_words, sample_k,
#                                           unique_word_index, index_to_word, seq_length, vocab_size)
#             st.success(f"Generated: {generated}")
            
#             # Add to history
#             st.session_state.history.append({"seed": seed_text, "generated": generated})
#         else:
#             st.warning("Please enter a seed sentence.")

# # History
# if st.session_state.history:
#     st.sidebar.title("📜 Prediction History")
#     for i, entry in enumerate(st.session_state.history[-5:]):  # Last 5
#         with st.sidebar.expander(f"Entry {len(st.session_state.history) - i}"):
#             if "input" in entry:
#                 st.write(f"**Input:** {entry['input']}")
#                 for word, prob in entry['predictions']:
#                     st.write(f"- {word} ({prob:.2%})")
#             elif "seed" in entry:
#                 st.write(f"**Seed:** {entry['seed']}")
#                 st.write(f"**Generated:** {entry['generated']}")

# st.sidebar.markdown("---")
# st.sidebar.info("**Project Highlights:**\n- Bidirectional LSTM for bidirectional context\n- Self-Attention for focusing on important words\n- GloVe pre-trained embeddings\n- Perplexity evaluation\n- Top-k sampling for diversity")
# app.py - Streamlit App for Enhanced Next Word Prediction
# Features:
# - Predict next word with top-3 options and confidences
# - Generate full sentence from seed text
# - User-friendly UI with history
# - Load custom model if needed

# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import pad_sequences

# @st.cache_resource
# def load_artifacts():
#     """Load model and info."""
#     model = load_model('attentive_lstm.h5')
#     with open('model_info.pkl', 'rb') as f:
#         info = pickle.load(f)
#     return model, info

# def prepare_input(text, unique_word_index, seq_length):
#     """Prepare integer sequence input from text."""
#     words = text.lower().split()
#     # Convert words to indices
#     sequence = [unique_word_index.get(word, 0) for word in words]  # 0 for unknown words
    
#     # Keep only the last seq_length words
#     if len(sequence) > seq_length:
#         sequence = sequence[-seq_length:]
    
#     # Pad sequence to seq_length
#     sequence = pad_sequences([sequence], maxlen=seq_length, padding='pre')[0]
    
#     # Reshape to (1, seq_length) for prediction
#     return sequence.reshape(1, seq_length)

# def predict_next(model, text, unique_word_index, index_to_word, seq_length, vocab_size, top_k=3):
#     """Predict top-k next words with probabilities."""
#     x = prepare_input(text, unique_word_index, seq_length)
#     probs = model.predict(x, verbose=0)[0]
#     top_indices = np.argsort(probs)[-top_k:][::-1]
#     top_words = [index_to_word.get(i, '<UNK>') for i in top_indices]
#     top_probs = probs[top_indices]
#     return top_words, top_probs

# def generate_sentence(model, seed_text, unique_word_index, index_to_word, seq_length, vocab_size, num_words=10, top_k=1):
#     """Generate a sentence by iteratively predicting next words."""
#     current_text = seed_text
#     for _ in range(num_words):
#         next_words, probs = predict_next(model, current_text, unique_word_index, index_to_word, seq_length, vocab_size, top_k)
        
#         if top_k == 1:
#             next_word = next_words[0]  # Greedy selection
#         else:
#             # Sample from top-k words based on probabilities
#             indices = np.arange(len(next_words))
#             selected_idx = np.random.choice(indices, p=probs/np.sum(probs))
#             next_word = next_words[selected_idx]
        
#         current_text += " " + next_word
        
#         # Stop if we hit an end token or unknown word
#         if next_word in ['<UNK>', '.', '!', '?']:
#             break
            
#     return current_text

# # Streamlit UI
# st.title("🧠 Next Word Prediction with Attention-Enhanced LSTM")
# st.markdown("**A Deep Learning Mini-Project: Bidirectional LSTM + Self-Attention for Text Generation**")
# st.markdown("Enter text to predict the next word or generate a full sentence. Trained on Project Gutenberg corpus with GloVe embeddings.")

# try:
#     model, info = load_artifacts()
#     index_to_word = info['index_to_word']
#     unique_word_index = info['unique_word_index']
#     vocab_size = info['vocab_size']
#     seq_length = info['seq_length']
    
#     # Display model info
#     st.sidebar.title("📊 Model Info")
#     st.sidebar.write(f"**Vocabulary Size:** {vocab_size:,}")
#     st.sidebar.write(f"**Sequence Length:** {seq_length}")
#     st.sidebar.write(f"**Model Shape Expected:** (batch_size, {seq_length})")
    
# except Exception as e:
#     st.error(f"Error loading model: {e}")
#     st.stop()

# # Session state for history
# if 'history' not in st.session_state:
#     st.session_state.history = []

# tab1, tab2 = st.tabs(["🔮 Predict Next Word", "✨ Generate Sentence"])

# with tab1:
#     input_text = st.text_input("Enter a sentence (last 50 words used as context):", 
#                                value="the quick brown fox jumps over the lazy dog")
#     num_options = st.slider("Top predictions:", 1, 5, 3)
    
#     if st.button("Predict Next Word", type="primary"):
#         if input_text.strip():
#             try:
#                 top_words, top_probs = predict_next(model, input_text, unique_word_index, 
#                                                     index_to_word, seq_length, vocab_size, num_options)
#                 st.success("Predictions:")
#                 for word, prob in zip(top_words, top_probs):
#                     st.write(f"**{word}** (confidence: {prob:.2%})")
                
#                 # Add to history
#                 st.session_state.history.append({
#                     "type": "prediction",
#                     "input": input_text, 
#                     "predictions": list(zip(top_words, top_probs))
#                 })
#             except Exception as e:
#                 st.error(f"Prediction error: {e}")
#         else:
#             st.warning("Please enter some text.")

# with tab2:
#     seed_text = st.text_input("Seed sentence:", value="once upon a time")
#     num_gen_words = st.slider("Number of words to generate:", 5, 50, 10)
#     sample_k = st.slider("Sampling (top-k):", 1, 5, 1)
    
#     if st.button("Generate Sentence", type="primary"):
#         if seed_text.strip():
#             try:
#                 generated = generate_sentence(model, seed_text, unique_word_index, index_to_word, 
#                                             seq_length, vocab_size, num_gen_words, sample_k)
#                 st.success(f"**Generated:** {generated}")
                
#                 # Add to history
#                 st.session_state.history.append({
#                     "type": "generation",
#                     "seed": seed_text, 
#                     "generated": generated,
#                     "params": f"words={num_gen_words}, top_k={sample_k}"
#                 })
#             except Exception as e:
#                 st.error(f"Generation error: {e}")
#         else:
#             st.warning("Please enter a seed sentence.")

# # History
# if st.session_state.history:
#     st.sidebar.title("📜 Prediction History")
#     for i, entry in enumerate(reversed(st.session_state.history[-5:])):  # Last 5, most recent first
#         entry_num = len(st.session_state.history) - i
#         with st.sidebar.expander(f"Entry {entry_num}"):
#             if entry["type"] == "prediction":
#                 st.write(f"**Input:** {entry['input'][:50]}...")
#                 for word, prob in entry['predictions']:
#                     st.write(f"- {word} ({prob:.2%})")
#             elif entry["type"] == "generation":
#                 st.write(f"**Seed:** {entry['seed']}")
#                 st.write(f"**Generated:** {entry['generated'][:100]}...")
#                 st.write(f"*{entry['params']}*")

# # Clear history button
# if st.session_state.history and st.sidebar.button("🗑️ Clear History"):
#     st.session_state.history = []
#     st.rerun()

# st.sidebar.markdown("---")
# st.sidebar.info("**Project Highlights:**\n- Bidirectional LSTM for bidirectional context\n- Self-Attention for focusing on important words\n- GloVe pre-trained embeddings\n- Perplexity evaluation\n- Top-k sampling for diversity")

# # Debug info (optional)
# with st.expander("🔧 Debug Info"):
#     if st.button("Test Input Shape"):
#         test_text = "the quick brown fox"
#         test_input = prepare_input(test_text, unique_word_index, seq_length)
#         st.write(f"Input text: '{test_text}'")
#         st.write(f"Processed shape: {test_input.shape}")
#         st.write(f"Sample values: {test_input[0][:10]}")  # First 10 values


import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

@st.cache_resource
def load_artifacts():
    """Load model and info."""
    model = load_model('enhanced_next_word_model.h5')
    with open('model_info.pkl', 'rb') as f:
        info = pickle.load(f)
    return model, info

def prepare_input(text, unique_word_index, seq_length):
    """Prepare integer sequence input from text."""
    words = text.lower().split()
    # Convert words to indices
    sequence = [unique_word_index.get(word, 0) for word in words]  # 0 for unknown words
    
    # Keep only the last seq_length words
    if len(sequence) > seq_length:
        sequence = sequence[-seq_length:]
    
    # Pad sequence to seq_length
    sequence = pad_sequences([sequence], maxlen=seq_length, padding='pre')[0]
    
    # Reshape to (1, seq_length) for prediction
    return sequence.reshape(1, seq_length)

@st.cache_data
def predict_next_cached(_model, text, unique_word_index, index_to_word, seq_length, vocab_size, top_k=3):
    """Cached version of predict_next."""
    x = prepare_input(text, unique_word_index, seq_length)
    probs = _model.predict(x, verbose=0)[0]
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_words = [index_to_word.get(i, '<UNK>') for i in top_indices]
    top_probs = probs[top_indices]
    return top_words, top_probs

def generate_sentence(model, seed_text, unique_word_index, index_to_word, seq_length, vocab_size, num_words=10, top_k=1):
    """Generate a sentence by iteratively predicting next words."""
    current_text = seed_text
    for _ in range(num_words):
        next_words, probs = predict_next_cached(model, current_text, unique_word_index, index_to_word, seq_length, vocab_size, top_k)
        
        if top_k == 1:
            next_word = next_words[0]  # Greedy selection
        else:
            # Sample from top-k words based on probabilities
            indices = np.arange(len(next_words))
            selected_idx = np.random.choice(indices, p=probs/np.sum(probs))
            next_word = next_words[selected_idx]
        
        current_text += " " + next_word
        
        # Stop if we hit an end token or unknown word
        if next_word in ['<UNK>', '<END>', '.', '!', '?']:
            break
            
    return current_text

# Streamlit UI
st.title("🧠 Next Word Prediction with Attention-Enhanced LSTM")
st.markdown("**A Deep Learning Mini-Project: Bidirectional LSTM + Self-Attention for Text Generation**")
st.markdown("Enter text to predict the next word or generate a full sentence. Trained on Project Gutenberg corpus with GloVe embeddings.")

try:
    model, info = load_artifacts()
    index_to_word = info['index_to_word']
    unique_word_index = info['unique_word_index']
    vocab_size = info['vocab_size']
    seq_length = info['seq_length']
    
    # Display model info
    st.sidebar.title("📊 Model Info")
    st.sidebar.write(f"**Vocabulary Size:** {vocab_size:,}")
    st.sidebar.write(f"**Sequence Length:** {seq_length}")
    st.sidebar.write(f"**Model Shape Expected:** (batch_size, {seq_length})")
    
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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
            words = input_text.strip().split()
            if len(words) < 5:
                st.warning("Please enter a sentence with at least 5 words for better context.")
            try:
                top_words, top_probs = predict_next_cached(model, input_text, unique_word_index, 
                                                    index_to_word, seq_length, vocab_size, num_options)
                st.success("Predictions:")
                for word, prob in zip(top_words, top_probs):
                    st.write(f"**{word}** (confidence: {prob:.2%})")
                
                # Add to history
                st.session_state.history.append({
                    "type": "prediction",
                    "input": input_text, 
                    "predictions": list(zip(top_words, top_probs))
                })
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("Please enter some text.")

with tab2:
    seed_text = st.text_input("Seed sentence:", value="once upon a time")
    num_gen_words = st.slider("Number of words to generate:", 5, 50, 10)
    sample_k = st.slider("Sampling (top-k):", 1, 5, 1)
    
    if st.button("Generate Sentence", type="primary"):
        if seed_text.strip():
            words = seed_text.strip().split()
            if len(words) < 5:
                st.warning("Please enter a seed with at least 5 words for better context.")
            try:
                generated = generate_sentence(model, seed_text, unique_word_index, index_to_word, 
                                            seq_length, vocab_size, num_gen_words, sample_k)
                st.success(f"**Generated:** {generated}")
                
                # Add to history
                st.session_state.history.append({
                    "type": "generation",
                    "seed": seed_text, 
                    "generated": generated,
                    "params": f"words={num_gen_words}, top_k={sample_k}"
                })
            except Exception as e:
                st.error(f"Generation error: {e}")
        else:
            st.warning("Please enter a seed sentence.")

# History
if st.session_state.history:
    st.sidebar.title("📜 Prediction History")
    for i, entry in enumerate(reversed(st.session_state.history[-5:])):  # Last 5, most recent first
        entry_num = len(st.session_state.history) - i
        with st.sidebar.expander(f"Entry {entry_num}"):
            if entry["type"] == "prediction":
                st.write(f"**Input:** {entry['input']}")
                for word, prob in entry['predictions']:
                    st.write(f"- {word} ({prob:.2%})")
            elif entry["type"] == "generation":
                st.write(f"**Seed:** {entry['seed']}")
                st.write(f"**Generated:** {entry['generated']}")
                st.write(f"*{entry['params']}*")

# Clear history button
if st.session_state.history and st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("**Project Highlights:**\n- Bidirectional LSTM for bidirectional context\n- Self-Attention for focusing on important words\n- GloVe pre-trained embeddings\n- Perplexity evaluation\n- Top-k sampling for diversity")

# Debug info (optional)
with st.expander("🔧 Debug Info"):
    if st.button("Test Input Shape"):
        test_text = "the quick brown fox"
        test_input = prepare_input(test_text, unique_word_index, seq_length)
        st.write(f"Input text: '{test_text}'")
        st.write(f"Processed shape: {test_input.shape}")
        st.write(f"Sample values: {test_input[0][:10]}")  # First 10 values