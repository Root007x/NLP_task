# IMDB Movie Review Sentiment Analysis

## Project Overview

This project was developed as part of the AI/ML Engineer selection process for Backbencher Studio. The goal is to build a sentiment analysis classifier that predicts whether a movie review is positive or negative.

## Live Demo

🚀 Try out the application: [IMDB Review Sentiment Analyzer](https://nlptask-ijdy3xubhgrjtw5t3tuuqq.streamlit.app/)

## Dataset

- **Source**: [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/mantri7/imdb-movie-reviews-dataset)
- **Contents**: Movie reviews labeled with sentiment (positive/negative)
- **Files**:
  - `data/train_data.csv`
  - `data/test_data.csv`

## Implementation Details

### Traditional Machine Learning Solution

1. **Data Preprocessing**

   - Text cleaning (HTML tags, punctuation, numbers removal)
   - Lowercase conversion
   - Stopwords removal using NLTK
   - Text vectorization using TF-IDF

2. **Model Development**

   - Implemented multiple classifiers:
     - Logistic Regression
     - Naive Bayes
     - Random Forest
   - Performed hyperparameter tuning using GridSearchCV
   - Selected best performing model based on accuracy

3. **Evaluation Metrics**
   - Accuracy: 88%
   - Precision: 87%
   - Recall: 88%
   - F1-Score: 88%
   - Confusion Matrix visualization included

### Deep Learning Solution (Bonus)

1. **BERT Implementation**

   - While traditional deep learning models (LSTM, GRU, RNN) were considered, chose transfer learning approach
   - Used BERT base model for its pre-trained knowledge and better language understanding
   - Fine-tuned for sentiment classification
   - Implemented custom prediction functionality

2. **Performance**
   - Achieved 94% accuracy on the test set
   - Superior performance compared to traditional ML approach (6% improvement)
   - Precision, Recall, and F1-Score all approximately 94%

Note: Since the deep learning approach was optional for this task, the deployed application uses the traditional ML model for inference, which provides faster predictions and lighter deployment.

## Project Structure

```
NLP/
├── data/
│   ├── train_data.csv
│   └── test_data.csv
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
├── app.py
├── Machine_learning_approach_for_NLP.ipynb
├── Deep_learning_approach_for_NLP_(Optional).ipynb
└── README.md
```

## Tools & Technologies

- **Python Libraries**:
  - Pandas: Data manipulation
  - NLTK: Text preprocessing
  - Scikit-learn: ML models & evaluation
  - HuggingFace Transformers: BERT implementation
  - Matplotlib: Visualization
- **Development**: Jupyter Notebook
- **Model Serialization**: Pickle

## How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Note: The `requirements.txt` only includes dependencies for the traditional ML approach and the Streamlit app. Deep learning dependencies are not included as the BERT model was trained using Google Colab's GPU environment.

3. Run the demo script:
   ```bash
   streamlit run app.py
   ```
4. For detailed analysis, check the Jupyter notebooks:
   - Traditional ML approach: `Machine_learning_approach_for_NLP.ipynb`
   - Deep Learning approach: `Deep_learning_approach_for_NLP_(Optional).ipynb` (Recommended to run in Google Colab with GPU runtime)

## Demo

The `app.py` script provides an interactive interface where users can:

- Input any movie review text
- Get instant sentiment prediction (positive/negative)
