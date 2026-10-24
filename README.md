---
title: High-Accuracy Email Classifier
emoji: 📧
colorFrom: blue
colorTo: green
sdk: tensorflow
sdk_version: 2.19.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
- email-classification
- text-classification
- cnn-gru
- edge-deployment
- tflite
language:
- en
metrics:
- accuracy
- precision
- recall
- f1
pipeline_tag: text-classification
widget:
- text: "Congratulations! You've won $1000! Click here to claim your prize!"
  example_title: "Spam Email"
- text: "Your verification code is 123456. Please enter this code to complete your login."
  example_title: "Verification Code"
- text: "New reply posted in the Python Programming forum."
  example_title: "Forum Notification"
- text: "Flash Sale! 50% off all items. Limited time offer!"
  example_title: "Promotional Email"
- text: "You have 5 new notifications on Facebook."
  example_title: "Social Media"
- text: "Security update available for your system."
  example_title: "System Update"
---

# High-Accuracy Email Classifier

## Model Description

This is a high-accuracy email classification model trained to categorize emails into 6 distinct categories with 98%+ accuracy. The model uses a sophisticated CNN+GRU architecture with multi-head attention, specifically designed for edge deployment scenarios.

## Categories

The model classifies emails into the following categories:

1. **📱 Social Media** - Notifications from social platforms (Facebook, Instagram, Twitter, etc.)
2. **🛒 Promotions** - Marketing emails, sales, offers, and advertisements  
3. **🗣️ Forum** - Forum posts, discussions, and community notifications
4. **⚠️ Spam** - Unwanted emails, scams, and phishing attempts
5. **🔐 Verify Code** - Authentication codes and verification emails
6. **🔄 Updates** - System updates, security patches, and maintenance notices

## Model Architecture

- **Base Architecture**: CNN + Bidirectional GRU with Multi-Head Attention
- **Vocabulary Size**: 25,000 words
- **Sequence Length**: 250 tokens
- **Embedding Dimension**: 300
- **Model Size**: 94MB (H5), 7.9MB (TFLite)

### Architecture Details

```
Input Layer (250,) 
    ↓
Embedding Layer (25000 → 300)
    ↓
Multi-scale CNN (kernels: 3, 4, 5)
    ↓
Bidirectional GRU (256 units)
    ↓
Multi-Head Attention (8 heads)
    ↓
Dense Layers + Dropout
    ↓
Output Layer (6 classes)
```

## Performance

- **Training Accuracy**: 98.13%
- **Validation Accuracy**: 98%+
- **Model Size**: 94MB (H5 format), 7.9MB (TFLite)
- **Inference Speed**: Optimized for mobile/edge deployment

## Quick Start

### Loading the Model

```python
import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = tf.keras.models.load_model('best_high_accuracy_model.h5')

# Load tokenizer configuration
with open('high_accuracy_tokenizer_config.json', 'r') as f:
    config = json.load(f)

categories = config['categories']
word_index = config['word_index']
max_len = config['max_len']
```

### Preprocessing Function

```python
import re

def preprocess_text(text):
    """Preprocess text exactly as done during training"""
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs
    text = re.sub(r'http[s]?://\S+', 'URL', text)
    text = re.sub(r'www\.\S+', 'URL', text)
    
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL', text)
    
    # Replace numbers
    text = re.sub(r'\b\d+\b', 'NUMBER', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

def text_to_sequence(text, word_index, max_len):
    """Convert text to padded sequence"""
    words = text.split()
    sequence = [word_index.get(word, 1) for word in words]  # 1 is OOV token
    return pad_sequences([sequence], maxlen=max_len, padding='post', truncating='post')
```

### Making Predictions

```python
def predict_email_category(text, model, word_index, categories, max_len):
    """Predict email category with confidence scores"""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Convert to sequence
    sequence = text_to_sequence(processed_text, word_index, max_len)
    
    # Get prediction
    prediction = model.predict(sequence, verbose=0)
    probabilities = prediction[0]
    
    # Get predicted class
    predicted_idx = np.argmax(probabilities)
    predicted_category = categories[predicted_idx]
    confidence = probabilities[predicted_idx]
    
    # Return all probabilities
    results = {
        'predicted_category': predicted_category,
        'confidence': float(confidence),
        'all_probabilities': {
            category: float(prob) 
            for category, prob in zip(categories, probabilities)
        }
    }
    
    return results

# Example usage
email_text = "Your verification code is 123456. Please enter this code."
result = predict_email_category(email_text, model, word_index, categories, max_len)
print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## TFLite Mobile Deployment

For mobile/edge deployment, use the optimized TFLite version:

```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='high_accuracy_email_classifier.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(text, interpreter, word_index, categories, max_len):
    """Predict using TFLite model"""
    # Preprocess and convert to sequence
    processed_text = preprocess_text(text)
    sequence = text_to_sequence(processed_text, word_index, max_len)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], sequence.astype(np.float32))
    interpreter.invoke()
    
    # Get output (already softmax probabilities)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = output_data[0]
    
    predicted_idx = np.argmax(probabilities)
    return categories[predicted_idx], probabilities
```

## Training Details

### Data Augmentation
- Synonym replacement
- Random word deletion  
- Word position swapping
- Contextual word insertion

### Advanced Techniques
- Multi-scale CNN filters (3, 4, 5)
- Bidirectional GRU with attention
- Class weight balancing
- Cosine annealing learning rate
- Early stopping with patience

### Preprocessing
- URL/Email/Number standardization
- Punctuation removal
- Case normalization
- OOV token handling

## Files Included

- `best_high_accuracy_model.h5` - Main Keras model (94MB)
- `high_accuracy_email_classifier.tflite` - Mobile-optimized TFLite model (7.9MB)
- `high_accuracy_tokenizer_config.json` - Tokenizer configuration and word mappings
- `android_config.json` - Mobile deployment configuration
- `confusion_matrix.png` - Model performance visualization

## Requirements

```
tensorflow>=2.19.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## License

This model is released under the Apache 2.0 License.

## Citation

```bibtex
@misc{high_accuracy_email_classifier,
  title={High-Accuracy Email Classifier with CNN-GRU Architecture},
  author={Email Classification Team},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/your-username/high-accuracy-email-classifier}
}
```

## Model Card Contact

For questions and issues, please open an issue in the repository or contact the model authors. 