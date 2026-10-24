
import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# 下载并加载模型
model = tf.keras.models.load_model('best_high_accuracy_model.h5')

# 加载配置
with open('high_accuracy_tokenizer_config.json', 'r') as f:
    config = json.load(f)

categories = config['categories']
word_index = config['word_index']
max_len = config['max_len']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', 'URL', text)
    text = re.sub(r'www\.\S+', 'URL', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL', text)
    text = re.sub(r'\b\d+\b', 'NUMBER', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def predict_email(text):
    processed_text = preprocess_text(text)
    words = processed_text.split()
    sequence = [word_index.get(word, 1) for word in words]
    sequence = pad_sequences([sequence], maxlen=max_len, padding='post', truncating='post')
    
    prediction = model.predict(sequence, verbose=0)
    probabilities = prediction[0]
    
    predicted_idx = np.argmax(probabilities)
    predicted_category = categories[predicted_idx]
    confidence = probabilities[predicted_idx]
    
    return predicted_category, confidence, dict(zip(categories, probabilities))

# 示例使用
email_text = "Your verification code is 123456"
category, confidence, all_probs = predict_email(email_text)
print(f"预测类别: {category}")
print(f"置信度: {confidence:.4f}")
