import os
import requests
import zipfile
from urllib.request import urlretrieve
import nltk
from nltk.tokenize import word_tokenize
import json

# Download punkt tokenizer
nltk.download('punkt', quiet=True)

def ensure_directory(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

def download_with_progress(url, filename):
    """Download file with progress indication."""
    print(f"Downloading {filename}...")
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\rProgress: {percent}%", end='', flush=True)
        
        urlretrieve(url, filename, reporthook=progress_hook)
        print(f"\n✓ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {filename}: {e}")
        return False

def create_sample_data_files():
    """Create sample data files for different domains."""
    ensure_directory('training_data')
    
    print("Creating sample training data files...")
    
    # Sample news articles
    news_content = """
    Breaking news: Scientists discover new method for renewable energy production.
    The technology could revolutionize how we generate clean electricity.
    Researchers at leading universities collaborate on climate change solutions.
    Economic markets respond positively to green technology investments.
    Government announces new policies for environmental protection.
    Tech companies invest billions in sustainable development projects.
    Innovation drives progress in artificial intelligence and machine learning.
    Healthcare advances improve quality of life for millions worldwide.
    Education systems adapt to digital transformation trends.
    Global cooperation essential for addressing climate challenges.
    """
    
    # Sample Wikipedia-style content
    wikipedia_content = """
    Artificial intelligence is intelligence demonstrated by machines.
    Machine learning is a subset of artificial intelligence.
    Deep learning uses artificial neural networks with multiple layers.
    Natural language processing enables computers to understand human language.
    Computer vision allows machines to interpret and understand visual information.
    Robotics combines mechanical engineering with artificial intelligence.
    Data science involves extracting insights from large datasets.
    Cloud computing provides on-demand access to computing resources.
    Cybersecurity protects digital systems from malicious attacks.
    Internet of things connects everyday objects to the internet.
    """
    
    # Sample search queries and responses
    search_queries_content = """
    How to learn machine learning
    What is artificial intelligence
    Best programming languages for beginners
    How does natural language processing work
    What are the applications of deep learning
    How to build a neural network
    What is computer vision used for
    How to get started with data science
    What are the benefits of cloud computing
    How does blockchain technology work
    Best practices for cybersecurity
    What is the future of artificial intelligence
    How to choose the right programming language
    What skills are needed for data science
    How machine learning algorithms work
    """
    
    # Sample social media content
    social_media_content = """
    Just learned about amazing new AI developments today!
    Working on exciting machine learning project this weekend.
    Anyone know good resources for learning data science?
    Fascinating article about the future of technology.
    Great discussion about programming best practices.
    Attending virtual conference on artificial intelligence.
    Sharing insights from latest research in computer science.
    Collaborating with team on innovative software solution.
    Exploring new frameworks for web development.
    Excited about opportunities in tech industry.
    """
    
    # Sample technical documentation
    technical_docs_content = """
    Installation guide for machine learning libraries.
    Configuration settings for neural network training.
    API documentation for natural language processing tools.
    Tutorial on implementing computer vision algorithms.
    Best practices for data preprocessing and cleaning.
    Optimization techniques for improving model performance.
    Debugging common issues in machine learning pipelines.
    Version control workflows for collaborative development.
    Testing strategies for artificial intelligence applications.
    Deployment procedures for production machine learning models.
    Security considerations for AI-powered systems.
    Monitoring and maintenance of deployed models.
    Performance metrics for evaluating algorithm effectiveness.
    Documentation standards for technical projects.
    Code review guidelines for development teams.
    """
    
    # Write sample files
    sample_files = {
        'training_data/news_articles.txt': news_content,
        'training_data/wikipedia_dump.txt': wikipedia_content,
        'training_data/search_queries.txt': search_queries_content,
        'training_data/social_media.txt': social_media_content,
        'training_data/technical_docs.txt': technical_docs_content
    }
    
    for filename, content in sample_files.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"✓ Created: {filename}")

def download_gutenberg_books():
    """Download additional books from Project Gutenberg."""
    ensure_directory('training_data')
    
    # List of free books from Project Gutenberg
    books = {
        'training_data/alice_in_wonderland.txt': 'https://www.gutenberg.org/files/11/11-0.txt',
        'training_data/pride_and_prejudice.txt': 'https://www.gutenberg.org/files/1342/1342-0.txt',
        'training_data/sherlock_holmes.txt': 'https://www.gutenberg.org/files/1661/1661-0.txt',
    }
    
    print("\nDownloading additional books from Project Gutenberg...")
    for filename, url in books.items():
        if not os.path.exists(filename):
            download_with_progress(url, filename)

def download_glove_embeddings():
    """Download GloVe embeddings if not present."""
    glove_file = 'training_data/glove.6B.100d.txt'
    
    if os.path.exists(glove_file):
        print(f"✓ GloVe embeddings already exist: {glove_file}")
        return
    
    print("\nGloVe embeddings not found.")
    print("Please download manually from: https://nlp.stanford.edu/projects/glove/")
    print("1. Download glove.6B.zip")
    print("2. Extract glove.6B.100d.txt to training_data/")
    print("3. Rename it if necessary")

def expand_training_data():
    """Expand existing training data by creating variations."""
    print("\nExpanding training data with variations...")
    
    base_files = [
        'training_data/news_articles.txt',
        'training_data/wikipedia_dump.txt',
        'training_data/search_queries.txt'
    ]
    
    # Common tech/AI terms to inject into training
    tech_terms = [
        'artificial intelligence', 'machine learning', 'deep learning',
        'neural networks', 'computer vision', 'natural language processing',
        'data science', 'big data', 'cloud computing', 'cybersecurity',
        'blockchain', 'internet of things', 'augmented reality',
        'virtual reality', 'robotics', 'automation'
    ]
    
    # Common question starters (search-engine like)
    question_starters = [
        'how to', 'what is', 'why does', 'when should', 'where can',
        'which are', 'who invented', 'best way to', 'tutorial for',
        'guide to', 'introduction to', 'basics of', 'advanced',
        'beginner', 'expert', 'professional'
    ]
    
    expanded_content = []
    
    # Generate tech-focused content
    for starter in question_starters:
        for term in tech_terms[:10]:  # Use first 10 terms
            expanded_content.append(f"{starter} {term}")
            expanded_content.append(f"{starter} learn {term}")
            expanded_content.append(f"{starter} understand {term}")
    
    # Write expanded content
    with open('training_data/expanded_queries.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(expanded_content))
    
    print(f"✓ Created expanded training data: {len(expanded_content)} entries")

def validate_training_data():
    """Validate that training data files exist and have content."""
    required_files = [
        'training_data/news_articles.txt',
        'training_data/wikipedia_dump.txt',
        'training_data/search_queries.txt',
        'training_data/social_media.txt',
        'training_data/technical_docs.txt'
    ]
    
    print("\nValidating training data files...")
    total_words = 0
    valid_files = 0
    
    for filename in required_files:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                words = len(word_tokenize(content))
                total_words += words
                valid_files += 1
                print(f"✓ {filename}: {words:,} words")
        else:
            print(f"✗ Missing: {filename}")
    
    print(f"\nValidation Summary:")
    print(f"Valid files: {valid_files}/{len(required_files)}")
    print(f"Total words: {total_words:,}")
    
    if valid_files < len(required_files):
        print("\n⚠️  Some training files are missing. The model will still work but with limited diversity.")
    else:
        print("\n✅ All training data files are ready!")
    
    return valid_files > 0

def create_dataset_info():
    """Create information file about the dataset."""
    info = {
        "dataset_info": {
            "name": "Multi-Domain Next Word Prediction Dataset",
            "description": "Diverse training data for search engine-like behavior",
            "domains": [
                "News articles",
                "Wikipedia-style encyclopedia content", 
                "Search queries and responses",
                "Social media posts",
                "Technical documentation",
                "Literature (Project Gutenberg)"
            ],
            "purpose": "Train next word prediction model with broad domain knowledge",
            "preprocessing": "Tokenized, lowercased, filtered for alphabetic words"
        }
    }
    
    with open('training_data/dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    print("✓ Created dataset information file")

def main():
    """Main data preparation function."""
    print("=" * 60)
    print("Multi-Domain Training Data Preparation")
    print("=" * 60)
    
    # Create directory structure
    ensure_directory('training_data')
    
    # Create sample data files
    create_sample_data_files()
    
    # Download additional resources
    download_gutenberg_books()
    download_glove_embeddings()
    
    # Expand training data
    expand_training_data()
    
    # Create dataset info
    create_dataset_info()
    
    # Validate everything
    if validate_training_data():
        print("\n" + "=" * 60)
        print("Data Preparation Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review the generated training data files")
        print("2. Add your own domain-specific data if needed")
        print("3. Download GloVe embeddings (if not already done)")
        print("4. Run the enhanced training script:")
        print("   python enhanced_train.py")
        print("=" * 60)
    else:
        print("\n❌ Data preparation failed. Please check the errors above.")

if __name__ == "__main__":
    main()