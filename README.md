# Neural Network Sentiment Analysis Tutorial

## Introduction
This repository provides a beginner-friendly tutorial for building neural network-based sentiment analysis models. You'll learn how to process raw text data and create a model that can predict whether movie reviews are positive or negative, using the IMDB dataset.

## What You'll Learn
- How to preprocess text data for machine learning
- Creating word embeddings to represent text numerically
- Building and training a neural network for sentiment classification
- Evaluating model performance and understanding results

## Core Concepts Explained

### What is Sentiment Analysis?
Sentiment analysis is the process of determining whether a piece of text expresses positive, negative, or neutral sentiment. In this tutorial, we focus on binary classification (positive/negative) using movie reviews.

### What is Word Embedding?
Word embeddings convert words into numerical vectors that capture meaning. Similar words will have similar vectors, allowing our neural network to understand relationships between words. For example, "good", "great", and "excellent" will have similar representations because they express similar sentiments.

Our model uses 100-dimensional embeddings, which means each word is represented by 100 numbers. This gives our model enough information to understand sentiment while keeping the model size manageable.

### Understanding Sequence Length
Text documents vary in length, but neural networks need fixed-size inputs. We'll use a maximum sequence length of 1500 words per review. This captures the content of most reviews while making training efficient.

## Implementation Guide

### 1. Data Preparation and Exploration
The IMDB dataset contains 50,000 movie reviews split evenly between positive and negative sentiment labels.

```python
# Load the dataset
data = pd.read_csv("IMDB_Dataset.csv")

# Check class distribution
data['sentiment'].value_counts()
# Output: 
# positive    25000
# negative    25000
```

### 2. Text Preprocessing Pipeline
Text preprocessing cleans and standardizes our input data:

```python
def preprocess_text(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    for punct in string.punctuation:
        text = text.replace(punct, '')
    
    # Expand contractions (e.g., "don't" → "do not")
    text = contractions.fix(text)
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    
    return ' '.join(words)
```

Why each step helps:
- **HTML removal**: Removes formatting tags that aren't relevant to sentiment
- **Lowercasing**: Makes "Good" and "good" the same word to the model
- **Contraction expansion**: Converts shortened forms like "don't" to "do not"
- **Stopword removal**: Removes common words like "the" and "and" that don't carry sentiment

### 3. Text Vectorization and Tokenization
Converting preprocessed text to numbers:

```python
# Define parameters
vocab_size = 10000  # Use the 10,000 most common words
max_length = 1500   # Maximum review length
oov_token = '<OOV>'  # Represents words not in our vocabulary

# Create and fit tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to uniform length
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
```

This process:
- Converts each word to a unique number (token)
- Limits vocabulary to the 10,000 most common words
- Makes all reviews the same length (1500 words) by either:
  - Cutting longer reviews (truncation)
  - Adding zeros to shorter reviews (padding)

### 4. Neural Network Architecture
Our sentiment analysis model has a simple but effective structure:

```python
def build_sentiment_model():
    model = Sequential()
    
    # Input embedding layer
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    
    # Global pooling layer
    model.add(GlobalAveragePooling1D())
    
    # Fully connected layers with dropout to prevent overfitting
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile with binary cross-entropy loss
    model.compile(loss='binary_crossentropy', 
                 optimizer=Adam(learning_rate=0.0001),
                 metrics=['accuracy'])
    
    return model
```

Key components:
- **Embedding layer**: Learns 100-dimensional word representations
- **Global Average Pooling**: Creates a single vector representing the entire review
- **Dense layers**: Learn patterns that determine sentiment
- **Dropout layers**: Prevent overfitting by randomly deactivating neurons during training
- **Sigmoid output**: Produces a probability between 0 (negative) and 1 (positive)

### 5. Model Training
We train the model with early stopping to prevent overfitting:

```python
# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(
    X_train_padded, y_train_encoded,
    epochs=200,
    batch_size=250,
    validation_split=0.2,
    callbacks=[early_stopping]
)
```

This approach:
- Uses 80% of training data for actual training
- Uses 20% for validation during training
- Stops training when validation performance stops improving
- Uses batches of 250 examples for efficient processing

### 6. Model Evaluation
We evaluate our model on the test data:

```python
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test_encoded)

# Generate predictions
predictions = model.predict(X_test_padded)
predicted_labels = (predictions > 0.5).astype(int)

# Calculate performance metrics
conf_matrix = confusion_matrix(y_test_encoded, predicted_labels)
f1 = f1_score(y_test_encoded, predicted_labels, average='weighted')
report = classification_report(y_test_encoded, predicted_labels, target_names=['negative', 'positive'])
```

Our model achieves around 89% accuracy in predicting sentiment.

## Common Challenges in Sentiment Analysis

Some types of text are particularly difficult for sentiment models:

1. **Mixed sentiment**: "Great acting but terrible plot"
2. **Sarcasm**: "Just what I needed, another bad movie"
3. **Subtle meanings**: "Almost a good thriller" 
4. **Negations**: "Not bad" (which is actually positive)

## Next Steps for Improvement

After completing this tutorial, here are ways you could enhance the model:

### 1. Try Different Model Architectures
- **LSTM/GRU networks**: Better at understanding word sequence and context
- **Transformer models**: State-of-the-art for many NLP tasks
- **Compare performance** with simpler models like Logistic Regression or SVM

### 2. Architecture Optimization
- Experiment with different embedding dimensions (50, 200, 300)
- Try different sequence lengths to balance coverage and efficiency
- Optimize batch size and learning rate for your hardware
- Test different layer sizes and dropout rates

### 3. Deployment Considerations
- Model compression techniques like quantization for mobile devices
- Cloud deployment options and cost considerations 
- Building active learning pipelines for continuous improvement

## Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-network-sentiment-analysis.git

# Install dependencies
pip install -r requirements.txt

# Run the tutorial notebook
jupyter notebook sentiment_analysis_tutorial.ipynb
```

## Requirements
```
numpy>=1.19.0
pandas>=1.0.0
nltk>=3.5
contractions==0.1.73
beautifulsoup4>=4.9.0
scikit-learn>=0.24.0
keras>=2.4.0
tensorflow>=2.4.0
```
