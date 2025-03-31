# Fake News Detection with NLP and ML

This project demonstrates how to preprocess text data and build machine learning models to classify news articles as real or fake. It includes text cleaning, TF-IDF vectorization, and model comparison using Logistic Regression and Support Vector Machine (SVM).

## 📁 Project Structure

```
.
├── data/
│   └── train.csv           # Dataset (not included for licensing reasons)
├── notebooks/
│   └── fake_news_detection.ipynb  # Main notebook
├── README.md
└── requirements.txt
```

## 📌 Features

- Text preprocessing with stopwords removal and lemmatization (NLTK)
- TF-IDF vectorization
- Training and evaluation of Logistic Regression and SVM models
- Accuracy, confusion matrix, and classification report
- Calibration curves for model probability analysis

## 🚀 Getting Started

1. Clone the repository
2. Add your dataset (`train.csv`) to the `data` folder
3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/fake_news_detection.ipynb
   ```

## 📦 Requirements

See `requirements.txt`

## 📜 License

This project is for educational purposes. Dataset usage should follow its original license.