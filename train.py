# # train.py - Enhanced Next Word Prediction Model with Bidirectional LSTM and Self-Attention
# # This version adds:
# # - Bidirectional LSTM for better context capture
# # - Self-Attention mechanism using Functional API
# # - Dropout for regularization
# # - Adam optimizer with learning rate
# # - Callbacks: EarlyStopping and ModelCheckpoint
# # - Pre-trained GloVe embeddings
# # - Evaluation with perplexity
# # - Pickle dump for model info
# # Suitable for B.Tech 4th year DL mini-project: Demonstrates advanced RNN architectures with attention.

# import numpy as np
# import pickle
# import os
# from urllib.request import urlretrieve
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download('punkt', quiet=True)

# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, Attention, GlobalAveragePooling1D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical
# from sklearn.metrics import classification_report  # For potential multi-class eval, but simplified here

# # Configuration
# DATA_FILE = 'training_data/1661-0.txt'# Download from https://www.gutenberg.org/files/1661/1661-0.txt
# GLOVE_FILE = 'training_data/glove.6B.100d.txt' # Download from https://nlp.stanford.edu/data/glove.6B.zip (extract)
# EMBEDDING_DIM = 100
# VOCAB_SIZE = 10000  # Max unique words
# SEQ_LENGTH = 50  # Sequence length for context

# # Download data if not present
# if not os.path.exists(DATA_FILE):
#     urlretrieve('https://www.gutenberg.org/files/1661/1661-0.txt', DATA_FILE)
#     print(f"Downloaded {DATA_FILE}")

# if not os.path.exists(GLOVE_FILE):
#     # Note: User needs to download manually or automate unzip
#     print(f"Please download {GLOVE_FILE} from https://nlp.stanford.edu/projects/glove/ and place in current directory.")


# def load_and_process_data(file_path, max_words):
#     """Load text, tokenize, and create unique word index limited to max_words."""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         text = f.read().lower()
    
#     # Tokenize
#     tokens = word_tokenize(text)
#     tokens = [word for word in tokens if word.isalpha()]  # Filter non-alpha
    
#     # Limit vocab
#     unique_words = list(set(tokens))[:max_words]
#     unique_word_index = {word: i for i, word in enumerate(unique_words)}
    
#     return unique_word_index, tokens

# def create_sequences(tokens, unique_word_index, seq_length):
#     """Create input sequences and targets."""
#     sequences = []
#     next_words = []
#     for i in range(len(tokens) - seq_length):
#         seq = tokens[i:i + seq_length]
#         if all(word in unique_word_index for word in seq):  # Skip if OOV
#             sequences.append([unique_word_index[word] for word in seq])
#             next_words.append(unique_word_index[tokens[i + seq_length]])
    
#     return np.array(sequences), np.array(next_words)

# def load_glove_embeddings(glove_path, unique_word_index, embedding_dim):
#     """Load GloVe embeddings into matrix."""
#     embeddings_index = {}
#     with open(glove_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             coefs = np.asarray(values[1:], dtype='float32')
#             if len(coefs) == embedding_dim:
#                 embeddings_index[word] = coefs
    
#     vocab_size = len(unique_word_index)
#     embedding_matrix = np.zeros((vocab_size, embedding_dim))
#     for word, i in unique_word_index.items():
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#         # Random init for OOV
#         else:
#             embedding_matrix[i] = np.random.normal(0, 0.1, (embedding_dim,))
    
#     return embedding_matrix

# def build_model(vocab_size, embedding_dim, seq_length, embedding_matrix):
#     """Build model with Bidirectional LSTM + Self-Attention."""
#     inputs = Input(shape=(seq_length,))
#     embedding = Embedding(vocab_size, embedding_dim,
#                           weights=[embedding_matrix], trainable=False)(inputs)
    
#     # Bidirectional LSTM with return_sequences for attention
#     lstm_out = Bidirectional(LSTM(128, return_sequences=True))(embedding)
#     lstm_out = Dropout(0.2)(lstm_out)
    
#     # Self-Attention
#     attention_out = Attention()([lstm_out, lstm_out])
    
#     # Global pooling to get fixed-size output
#     pooled = GlobalAveragePooling1D()(attention_out)
    
#     # Dense layers
#     dense = Dense(128, activation='relu')(pooled)
#     dense = Dropout(0.2)(dense)
#     outputs = Dense(vocab_size, activation='softmax')(dense)
    
#     model = Model(inputs, outputs)
#     return model

# def calculate_perplexity(model, X, y):
#     """Calculate perplexity on validation set."""
#     preds = model.predict(X, verbose=0)
#     cross_entropy = -np.sum(y * np.log(preds + 1e-8)) / len(y)
#     perplexity = np.exp(cross_entropy)
#     return perplexity

# # Main training
# print("Loading and processing data...")
# unique_word_index, tokens = load_and_process_data(DATA_FILE, VOCAB_SIZE)
# index_to_word = {i: word for word, i in unique_word_index.items()}
# vocab_size = len(unique_word_index)

# print("Creating sequences...")
# X, y = create_sequences(tokens, unique_word_index, SEQ_LENGTH)
# y = to_categorical(y, vocab_size)

# # Split for validation
# split = int(0.9 * len(X))
# X_train, X_val = X[:split], X[split:]
# y_train, y_val = y[:split], y[split:]

# print("Loading GloVe embeddings...")
# embedding_matrix = load_glove_embeddings(GLOVE_FILE, unique_word_index, EMBEDDING_DIM)

# print("Building model...")
# model = build_model(vocab_size, EMBEDDING_DIM, SEQ_LENGTH, embedding_matrix)

# optimizer = Adam(learning_rate=0.001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
#     ModelCheckpoint('attentive_lstm.h5', monitor='val_loss', save_best_only=True)
# ]

# print("Training model...")
# history = model.fit(X_train, y_train, epochs=20, batch_size=128,
#                     validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

# # Evaluate
# train_perplexity = calculate_perplexity(model, X_train, y_train)
# val_perplexity = calculate_perplexity(model, X_val, y_val)
# print(f"Training Perplexity: {train_perplexity:.2f}")
# print(f"Validation Perplexity: {val_perplexity:.2f}")

# # Save model and info
# model.save('attentive_lstm.h5')
# model_info = {
#     'index_to_word': index_to_word,
#     'unique_word_index': unique_word_index,
#     'vocab_size': vocab_size,
#     'seq_length': SEQ_LENGTH
# }
# with open('model_info.pkl', 'wb') as f:
#     pickle.dump(model_info, f)
# print("Saved model_info.pkl successfully")

# print("Training complete! Model saved as 'attentive_lstm.h5'")
# print("Run: streamlit run app.py")

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Enhanced Configuration for Multi-Domain Training
DATA_SOURCES = [
    'training_data/news_articles.txt',
    'training_data/wikipedia_dump.txt', 
    'training_data/search_queries.txt',
    'training_data/social_media.txt',
    'training_data/technical_docs.txt',
    'training_data/expanded_queries.txt',
    'training_data/alice_in_wonderland.txt',
    'training_data/pride_and_prejudice.txt',
    'training_data/sherlock_holmes.txt'
]

GLOVE_FILE = 'training_data/glove.6B.100d.txt'
EMBEDDING_DIM = 100
VOCAB_SIZE = 15000  # Increased for multi-domain coverage
SEQ_LENGTH = 50  # Context window
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001

def ensure_data_directory():
    """Ensure training_data directory exists."""
    os.makedirs('training_data', exist_ok=True)

def download_fallback_data():
    """Download fallback data if other sources are not available."""
    fallback_file = 'training_data/sherlock_holmes.txt'
    if not os.path.exists(fallback_file):
        print("Downloading fallback training data...")
        urlretrieve('https://www.gutenberg.org/files/1661/1661-0.txt', fallback_file)
        print(f"Downloaded {fallback_file}")

def load_multiple_sources(file_paths, max_words):
    """Load and combine multiple data sources for diverse training."""
    all_tokens = []
    sources_loaded = []
    
    print("Loading data from multiple sources...")
    for path in file_paths:
        if os.path.exists(path):
            try:
                print(f"  Loading: {path}")
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().lower()
                
                # Clean Gutenberg boilerplate if present
                if '*** start' in text and '*** end' in text:
                    text = text.split('*** start', 1)[1].split('*** end')[0]
                
                # Tokenize into sentences and add <START>/<END>
                sentences = sent_tokenize(text)
                for sent in sentences:
                    words = [word for word in word_tokenize(sent) if word.isalpha()]
                    if words:
                        all_tokens.extend(['<START>'] + words + ['<END>'])
                
                sources_loaded.append(path)
                print(f"    Added {len(all_tokens):,} tokens (cumulative)")
            except Exception as e:
                print(f"    Error loading {path}: {e}")
        else:
            print(f"  File not found: {path}")
    
    if not sources_loaded:
        print("No data sources found! Please add training data files.")
        return None, None, None
    
    print(f"Total tokens loaded: {len(all_tokens):,}")
    
    # Add special tokens for better handling
    unique_words = ['<PAD>', '<UNK>', '<START>', '<END>'] + list(set(all_tokens))[:max_words-4]
    unique_word_index = {word: i for i, word in enumerate(unique_words)}
    
    print(f"Vocabulary size: {len(unique_words)}")
    return unique_word_index, all_tokens, sources_loaded

def create_sequences_with_unk(tokens, unique_word_index, seq_length):
    """Create sequences with unknown word handling."""
    sequences = []
    next_words = []
    unk_id = unique_word_index['<UNK>']
    
    print("Creating sequences with UNK token handling...")
    
    for i in range(len(tokens) - seq_length):
        seq = tokens[i:i + seq_length]
        # Map unknown words to <UNK> instead of skipping
        seq_indices = [unique_word_index.get(word, unk_id) for word in seq]
        next_word_idx = unique_word_index.get(tokens[i + seq_length], unk_id)
        
        sequences.append(seq_indices)
        next_words.append(next_word_idx)
    
    print(f"Created {len(sequences):,} training sequences")
    return np.array(sequences), np.array(next_words)

def augment_training_data(sequences, next_words, augment_ratio=0.1):
    """Augment training data with noise for better generalization."""
    print(f"Augmenting training data with {augment_ratio:.1%} noise...")
    
    augmented_seq = list(sequences)
    augmented_next = list(next_words)
    
    num_augment = int(len(sequences) * augment_ratio)
    indices = np.random.choice(len(sequences), size=num_augment, replace=False)
    
    for idx in indices:
        seq = sequences[idx].copy()
        next_word = next_words[idx]
        
        # Add noise (randomly replace some words with <UNK>)
        mask_positions = np.random.choice(len(seq), size=max(1, len(seq)//10), replace=False)
        seq[mask_positions] = 1  # <UNK> token index
        
        augmented_seq.append(seq)
        augmented_next.append(next_word)
    
    print(f"Augmented dataset size: {len(augmented_seq):,} sequences")
    return np.array(augmented_seq), np.array(augmented_next)

def load_glove_embeddings(glove_path, unique_word_index, embedding_dim):
    """Load GloVe embeddings with improved coverage and fallback."""
    print("Loading GloVe embeddings...")
    
    if not os.path.exists(glove_path):
        print(f"Warning: {glove_path} not found. Using random embeddings.")
        vocab_size = len(unique_word_index)
        return np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == embedding_dim:
                embeddings_index[word] = coefs
    
    print(f"Found {len(embeddings_index):,} word vectors")
    
    vocab_size = len(unique_word_index)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    hits = 0
    for word, i in unique_word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            # Better initialization for OOV words
            embedding_matrix[i] = np.random.normal(0, 0.1, (embedding_dim,))
    
    print(f"GloVe coverage: {hits}/{vocab_size} ({hits/vocab_size:.1%})")
    return embedding_matrix

def build_improved_model(vocab_size, embedding_dim, seq_length, embedding_matrix):
    """Build improved model with reduced size to prevent overfitting."""
    print("Building enhanced model architecture...")
    
    inputs = Input(shape=(seq_length,))
    
    # Make embedding trainable for better adaptation
    if embedding_matrix is not None:
        embedding = Embedding(vocab_size, embedding_dim, 
                             weights=[embedding_matrix], trainable=True)(inputs)
    else:
        embedding = Embedding(vocab_size, embedding_dim, trainable=True)(inputs)
    
    # Reduced capacity with two LSTM layers
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(lstm_out)
    
    # Self-attention mechanism
    attention_out = Attention()([lstm_out, lstm_out])
    pooled = GlobalAveragePooling1D()(attention_out)
    
    # Reduced dense layers with regularization
    dense = Dense(256, activation='relu')(pooled)
    dense = Dropout(0.4)(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    
    outputs = Dense(vocab_size, activation='softmax')(dense)
    
    model = Model(inputs, outputs)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    return model

def calculate_perplexity(model, X, y):
    """Calculate perplexity with better numerical stability."""
    preds = model.predict(X, verbose=0, batch_size=256)
    # Convert one-hot back to indices for perplexity calculation
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_indices = np.argmax(y, axis=1)
        relevant_probs = preds[np.arange(len(preds)), y_indices]
    else:
        relevant_probs = preds[np.arange(len(preds)), y]
    
    # Clip probabilities to avoid log(0)
    relevant_probs = np.clip(relevant_probs, 1e-8, 1.0)
    cross_entropy = -np.mean(np.log(relevant_probs))
    perplexity = np.exp(cross_entropy)
    
    return perplexity

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training history saved to {save_path}")

class CustomCallback(ModelCheckpoint):
    """Custom callback to save model info and track metrics."""
    def __init__(self, filepath, model_info, **kwargs):
        super().__init__(filepath, **kwargs)
        self.model_info = model_info
        self.best_perplexity = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        
        # Save model info
        with open('model_info.pkl', 'wb') as f:
            pickle.dump(self.model_info, f)
        
        # Track perplexity if available
        if logs and 'val_loss' in logs:
            val_perplexity = np.exp(logs['val_loss'])
            if val_perplexity < self.best_perplexity:
                self.best_perplexity = val_perplexity
                self.model_info['best_val_perplexity'] = val_perplexity
        
        print(f"\nEpoch {epoch + 1} completed. Model info saved.")

def main():
    """Main training function."""
    print("=" * 60)
    print("Enhanced Multi-Domain Next Word Prediction Training")
    print("=" * 60)
    
    # Setup
    ensure_data_directory()
    download_fallback_data()
    
    # Load multi-domain data
    unique_word_index, tokens, sources_loaded = load_multiple_sources(DATA_SOURCES, VOCAB_SIZE)
    
    if unique_word_index is None:
        print("Failed to load training data. Exiting.")
        return
    
    # Create reverse mapping
    index_to_word = {i: word for word, i in unique_word_index.items()}
    vocab_size = len(unique_word_index)
    
    print(f"\nDataset Statistics:")
    print(f"  Sources loaded: {len(sources_loaded)}")
    print(f"  Total tokens: {len(tokens):,}")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Sequence length: {SEQ_LENGTH}")
    
    # Create sequences with UNK handling
    X, y = create_sequences_with_unk(tokens, unique_word_index, SEQ_LENGTH)
    
    # Data augmentation
    X, y = augment_training_data(X, y, augment_ratio=0.1)
    
    # Convert to categorical
    y = to_categorical(y, vocab_size)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )
    
    print(f"\nTraining set: {len(X_train):,} sequences")
    print(f"Validation set: {len(X_val):,} sequences")
    
    # Load embeddings
    embedding_matrix = load_glove_embeddings(GLOVE_FILE, unique_word_index, EMBEDDING_DIM)
    
    # Build model
    model = build_improved_model(vocab_size, EMBEDDING_DIM, SEQ_LENGTH, embedding_matrix)
    
    # Compile with improved optimizer
    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=optimizer, 
        metrics=['accuracy']
    )
    
    # Prepare model info for saving
    model_info = {
        'index_to_word': index_to_word,
        'unique_word_index': unique_word_index,
        'vocab_size': vocab_size,
        'seq_length': SEQ_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'sources_loaded': sources_loaded,
        'total_tokens': len(tokens),
        'training_sequences': len(X_train),
        'validation_sequences': len(X_val)
    }
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=8, 
            restore_best_weights=True,
            verbose=1
        ),
        CustomCallback(
            'enhanced_next_word_model.h5', 
            model_info,
            monitor='val_loss', 
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training Complete - Final Evaluation")
    print("="*60)
    
    try:
        # Calculate final perplexities
        train_perplexity = calculate_perplexity(model, X_train, y_train)
        val_perplexity = calculate_perplexity(model, X_val, y_val)
        
        print(f"Final Training Perplexity: {train_perplexity:.2f}")
        print(f"Final Validation Perplexity: {val_perplexity:.2f}")
        
        # Add to model info
        model_info['train_perplexity'] = train_perplexity
        model_info['val_perplexity'] = val_perplexity
        
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
    
    # Save final model and info
    model.save('enhanced_next_word_model.h5')
    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    # Plot training history
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"Error plotting history: {e}")
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Model saved as: enhanced_next_word_model.h5")
    print(f"Model info saved as: model_info.pkl")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Training sequences: {len(X_train):,}")
    print(f"Data sources: {len(sources_loaded)}")
    print("\nTo run the Streamlit app:")
    print("streamlit run app.py")
    print("="*60)

if __name__ == "__main__":
    main()