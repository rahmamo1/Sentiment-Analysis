# Sentiment Analysis

This project performs **sentiment analysis** on text data using deep learning models. It includes experiments with **LSTM** and **BERT**, while the deployed application uses the **LSTM model** for real-time predictions.

## Features
- Text preprocessing and cleaning
- Sentiment classification into **Positive, Negative, Neutral, and Irrelevant**
- Model training using **LSTM** and **BERT**
- Evaluation with accuracy, precision, recall, and F1-score
- Streamlit-based web app for interactive sentiment prediction

---

## Example Predictions
| Text | Sentiment |
|------|-----------|
| "I love this product!" | Positive |
| "This is terrible." | Negative |
| "Not sure how I feel about this." | Neutral |
| "I don't care about this." | Irrelevant |

## Model Details
**LSTM Model:**
- Input: TF-IDF features
- Layers: LSTM -> Dropout -> Dense
- Output: Softmax classification for 4 sentiment classes
- Saved as `best_lstm_model.h5`

**BERT Model:**
- Fine-tuned for sentiment classification using Hugging Face Transformers
- Used for experimentation and evaluation

## Dataset & Implementation
- **Dataset**: [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data)
- **Kaggle Notebook**: [Sentiment Analysis 97%](https://www.kaggle.com/code/rahmamabdelfattah/sentiment-analysis-97)
