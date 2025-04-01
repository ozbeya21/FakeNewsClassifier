# Fake News Detection using Machine Learning

## 📌 Project Overview
This project is a machine learning-based fake news detection system. The model utilizes **TF-IDF vectorization** and **SGDClassifier (Support Vector Machine-like model)** to classify news articles as **fake** or **real**.

### 🚀 Key Features
- **High Accuracy (96%)** → The model achieves a low **error rate of 4%**.
- **Optimized Preprocessing** → Fast text processing using `swifter`.
- **Machine Learning Pipeline** → A full **TF-IDF + SGDClassifier** pipeline for classification.
- **Advanced Visualizations** → Includes **Confusion Matrix, ROC Curve, and WordClouds**.
- **Model Persistence** → The trained model is saved using `joblib` for future use.

## 📂 Dataset
The dataset used is `train.csv` which taken from [Kaggle](https://www.kaggle.com/competitions/fake-news/data), containing labeled news articles:
- **`text`**: The news article text.
- **`label`**: `0` for **real** news, `1` for **fake** news.

## 🔧 Installation & Requirements
To run this project, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 🎯 How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the script**
   ```bash
   python fake_news_detection.py
   ```
4. **Test with sample texts**
   ```python
   from joblib import load
   model = load("fakenews_model.pkl")
   sample_text = "Breaking news: Stock market crashes due to economic instability!"
   print(model.predict([sample_text]))
   ```

## 📊 Model Performance & Results
### 🔹 Confusion Matrix
A confusion matrix showing model performance on test data:
![Confusion Matrix](https://github.com/user-attachments/assets/5112e195-6bc1-4712-ae2a-c4ee04387e0f)

### 🔹 ROC Curve
AUC Score for model performance evaluation:
![Ekran görüntüsü 2025-04-01 121730](https://github.com/user-attachments/assets/43585e11-c042-4e54-b7c2-45fe7e580f4f)


### 🔹 WordCloud of Fake vs Real News
![Ekran görüntüsü 2025-04-01 121745](https://github.com/user-attachments/assets/70adcd73-b802-4d95-b900-00c024c2ac5e)

## 💾 Model Saving & Loading
The trained model is saved as `fakenews_model.pkl` and can be reloaded for future use:
```python
from joblib import load
model = load("fakenews_model.pkl")
```

## 🛠 Technologies Used
- **Python**
- **Scikit-learn** (`SGDClassifier`, `TF-IDF`)
- **Matplotlib & Seaborn** (for visualizations)
- **Swifter** (for faster text processing)
- **WordCloud** (for data insights)
- **Joblib** (for model persistence)

## 🏆 Contributors
- **Ahmet Özbey** – [LinkedIn](https://www.linkedin.com/in/ahmet-%C3%B6zbey-b43368261/) | [GitHub](https://github.com/ozbeya21)

## 📜 License
This project is licensed under the **MIT License**.

---

# requirements.txt
```
pandas
numpy
swifter
joblib
matplotlib
seaborn
wordcloud
scikit-learn
```

