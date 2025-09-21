# train.py - Enhanced Next Word Prediction Model with Bidirectional LSTM and Self-Attention
# This version adds:
# - Bidirectional LSTM for better context capture
# - Self-Attention mechanism using Functional API
# - Dropout for regularization
# - Adam optimizer with learning rate
# - Callbacks: EarlyStopping and ModelCheckpoint
# - Pre-trained GloVe embeddings
# - Evaluation with perplexity
# - Pickle dump for model info
# Suitable for B.Tech 4th year DL mini-project: Demonstrates advanced RNN architectures with attention.

import numpy as np
import pickle
import os
from urllib.request import urlretrieve
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt', quiet=True)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report  # For potential multi-class eval, but simplified here

# Configuration
DATA_FILE = '1661-0.txt'  # Download from https://www.gutenberg.org/files/1661/1661-0.txt
GLOVE_FILE = 'glove.6B.100d.txt'  # Download from https://nlp.stanford.edu/data/glove.6B.zip (extract)
EMBEDDING_DIM = 100
VOCAB_SIZE = 10000  # Max unique words
SEQ_LENGTH = 50  # Sequence length for context

# Download data if not present
if not os.path.exists(DATA_FILE):
    urlretrieve('https://www.gutenberg.org/files/1661/1661-0.txt', DATA_FILE)
    print(f"Downloaded {DATA_FILE}")

if not os.path.exists(GLOVE_FILE):
    # Note: User needs to download manually or automate unzip
    print(f"Please download {GLOVE_FILE} from https://nlp.stanford.edu/projects/glove/ and place in current directory.")

def load_and_process_data(file_path, max_words):
    """Load text, tokenize, and create unique word index limited to max_words."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]  # Filter non-alpha
    
    # Limit vocab
    unique_words = list(set(tokens))[:max_words]
    unique_word_index = {word: i for i, word in enumerate(unique_words)}
    
    return unique_word_index, tokens

def create_sequences(tokens, unique_word_index, seq_length):
    """Create input sequences and targets."""
    sequences = []
    next_words = []
    for i in range(len(tokens) - seq_length):
        seq = tokens[i:i + seq_length]
        if all(word in unique_word_index for word in seq):  # Skip if OOV
            sequences.append([unique_word_index[word] for word in seq])
            next_words.append(unique_word_index[tokens[i + seq_length]])
    
    return np.array(sequences), np.array(next_words)

def load_glove_embeddings(glove_path, unique_word_index, embedding_dim):
    """Load GloVe embeddings into matrix."""
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == embedding_dim:
                embeddings_index[word] = coefs
    
    vocab_size = len(unique_word_index)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in unique_word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        # Random init for OOV
        else:
            embedding_matrix[i] = np.random.normal(0, 0.1, (embedding_dim,))
    
    return embedding_matrix

def build_model(vocab_size, embedding_dim, seq_length, embedding_matrix):
    """Build model with Bidirectional LSTM + Self-Attention."""
    inputs = Input(shape=(seq_length,))
    embedding = Embedding(vocab_size, embedding_dim,
                          weights=[embedding_matrix], trainable=False)(inputs)
    
    # Bidirectional LSTM with return_sequences for attention
    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(embedding)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Self-Attention
    attention_out = Attention()([lstm_out, lstm_out])
    
    # Global pooling to get fixed-size output
    pooled = GlobalAveragePooling1D()(attention_out)
    
    # Dense layers
    dense = Dense(128, activation='relu')(pooled)
    dense = Dropout(0.2)(dense)
    outputs = Dense(vocab_size, activation='softmax')(dense)
    
    model = Model(inputs, outputs)
    return model

def calculate_perplexity(model, X, y):
    """Calculate perplexity on validation set."""
    preds = model.predict(X, verbose=0)
    cross_entropy = -np.sum(y * np.log(preds + 1e-8)) / len(y)
    perplexity = np.exp(cross_entropy)
    return perplexity

# Main training
print("Loading and processing data...")
unique_word_index, tokens = load_and_process_data(DATA_FILE, VOCAB_SIZE)
index_to_word = {i: word for word, i in unique_word_index.items()}
vocab_size = len(unique_word_index)

print("Creating sequences...")
X, y = create_sequences(tokens, unique_word_index, SEQ_LENGTH)
y = to_categorical(y, vocab_size)

# Split for validation
split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print("Loading GloVe embeddings...")
embedding_matrix = load_glove_embeddings(GLOVE_FILE, unique_word_index, EMBEDDING_DIM)

print("Building model...")
model = build_model(vocab_size, EMBEDDING_DIM, SEQ_LENGTH, embedding_matrix)

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('attentive_lstm.h5', monitor='val_loss', save_best_only=True)
]

print("Training model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=128,
                    validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

# Evaluate
train_perplexity = calculate_perplexity(model, X_train, y_train)
val_perplexity = calculate_perplexity(model, X_val, y_val)
print(f"Training Perplexity: {train_perplexity:.2f}")
print(f"Validation Perplexity: {val_perplexity:.2f}")

# Save model and info
model.save('attentive_lstm.h5')
model_info = {
    'index_to_word': index_to_word,
    'unique_word_index': unique_word_index,
    'vocab_size': vocab_size,
    'seq_length': SEQ_LENGTH
}
with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("Training complete! Model saved as 'attentive_lstm.h5'")
print("Run: streamlit run app.py")