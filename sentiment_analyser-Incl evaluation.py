from transformers import pipeline 
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading a pre-trained sentiment analysis model
print('Loading...')
classifier=pipeline('sentiment-analysis', model='distilbert-base-' \
'uncased-finetuned-sst-2-english')

#Loading IMDB dataset
print('Loading Dataset...')
dataset=load_dataset('imdb', split='test')
dataset=dataset.shuffle(seed=42).selectrange(1000)


# Function to predict sentiment for evaluation
def predict_sentiment(texts,batch_size=32):
    predictions=[]
    for i in range(0, len(texts), batch_size):
        batch=texts[i:i+batch_size]
        results=classifier(batch, truncation=True, max_length=512)
        predictions.extend(results)

    return predictions

# Get predictions 
print('Make predictions on 1000 reviews')
texts=dataset['text']
predictions= predict_sentiment(texts)

# Convert predictions to binary (POSITIVE=1, NEGATIVE=0)
pred_labels= [1 if p['label'] == 'POSITIVE' else 0 for p in 
                     predictions]
true_labels= dataset['label']

# Evaluate 
accuracy= accuracy_score (true_labels, pred_labels)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(true_labels, pred_labels, target_names=['NEGATIVE', 'POSITIVE']))

# Show some examples
print ('\n---Example predictions---')
for i in range(5):
    print(f'\nReview: {texts[i][:200]}...')
    print(f'True:{"Positive" if true_labels[i]==1 else "Negative"}')
    print(f"Predicted: {predictions[i]['label']} (confidence: {predictions[i]['score']:.3f})")
                                                            