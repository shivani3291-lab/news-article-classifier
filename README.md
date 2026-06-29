# News Article Classification System

A machine learning project to automatically categorize news articles into one of four categories: **World**, **Sports**, **Business**, or **Sci/Tech** using semantic embeddings and advanced NLP models.

---

## Project Overview

This project addresses the challenge faced by **E-news Express**, a news aggregation startup, in efficiently categorizing large volumes of incoming news articles. Manual categorization is time-consuming and error-prone. By leveraging machine learning and NLP techniques, we've built an automated system that achieves **88%+ accuracy** in article classification.

---

## Problem Statement

### Business Challenge
- **Information Overload**: Processing thousands of news articles daily manually is impractical
- **Time Sensitivity**: Delays in categorization result in outdated or misplaced content
- **Quality Assurance**: Human errors in categorization damage reputation and user experience
- **Scalability**: Need for an automated, scalable solution

### Solution
Build a machine learning model that automatically classifies news articles by analyzing their text content and predicting the most appropriate category.

---

## Dataset

- **Size**: 4,000 news articles
- **Classes**: 4 balanced categories (1,000 articles each)
  - **Class 0**: World News
  - **Class 1**: Sports News
  - **Class 2**: Business News
  - **Class 3**: Science & Technology News
- **Features**: Single text column (article content)
- **Class Distribution**: Perfectly balanced (25% each category)

### Key Statistics
- Average article length: 30-50 words
- Total unique words: ~5,000+
- Data split:
  - Training: 80% (3,200 articles)
  - Validation: 10% (400 articles)
  - Testing: 10% (400 articles)

---

## Methodology

### 1. **Feature Engineering: Sentence Embeddings**
Used **SentenceTransformer (all-MiniLM-L6-v2)** to convert raw text into semantic vectors:
- Output dimension: 384-dimensional embeddings
- Captures contextual meaning and semantic relationships
- Pre-trained on large corpus, ready to use without fine-tuning

### 2. **Classical Machine Learning Models**

#### a) **Random Forest (Base)**
- Default hyperparameters
- Performance: 90% accuracy on validation
- Issue: Slight overfitting on training data

#### b) **Random Forest with Class Weights**
- Added `class_weight="balanced"` to handle imbalance
- Performance: 87% accuracy
- Better generalization than base model

#### c) **Random Forest (Tuned)**
- **GridSearchCV** optimization with 3-fold cross-validation
- **Hyperparameters tuned**:
  - `max_depth`: [4, 7]
  - `max_features`: ['sqrt', 0.5, 0.7]
  - `min_samples_split`: [5, 6]
  - `n_estimators`: [30, 45, 60, 75, 90, 105]
- **Scoring metric**: Weighted recall
- Performance: 88% accuracy on validation
- Best balance of precision and recall

### 3. **Transformer-Based Models**

#### a) **FLAN-T5 Large (Base Prompt)**
- Google's instruction-tuned T5 model
- Prompt: Generic instruction for category classification
- Performance: 66% accuracy (underfitting)
- Issue: Poor prompt guidance

#### b) **FLAN-T5 Large (Improved Prompt)**
- Enhanced prompt with clear category definitions
- Added examples and explicit instructions
- **Improvements**:
  - Better contextual understanding
  - Clear category boundaries
  - Explicit guidance on main theme
- Performance: **92% accuracy on validation**
- **Best overall performer**

---

## Results

### Model Comparison

| Model | Train Accuracy | Validation Accuracy | F1-Score | Best For |
|-------|---|---|---|---|
| Random Forest (Base) | 100% | 90% | 0.90 | Baseline |
| RF with Class Weights | 100% | 87% | 0.87 | Regularization |
| **RF Tuned (GridSearchCV)** | 96% | 88% | 0.88 | Speed & Reliability |
| FLAN-T5 (Base Prompt) | 63% | 66% | 0.61 | Reference |
| **FLAN-T5 (Improved Prompt)** | **92%** | **92%** | **0.92** | **Production** |

### Performance Metrics (Test Set - Best Model)

**FLAN-T5 with Improved Prompt**
- **Overall Accuracy**: 92%
- **Precision**: 0.91 (weighted average)
- **Recall**: 0.92 (weighted average)
- **F1-Score**: 0.92

### Class-wise Performance
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| World | 0.89 | 0.91 | 0.90 | 100 |
| Sports | 0.95 | 0.96 | 0.95 | 100 |
| Business | 0.92 | 0.90 | 0.91 | 100 |
| Sci/Tech | 0.88 | 0.89 | 0.89 | 100 |

### Key Insights
- **Sports** articles are easiest to classify (96% recall) - distinct terminology
- **Business & Sci/Tech** show some confusion (likely overlapping vocabulary)
- **Prompt engineering impact**: Improved FLAN-T5 accuracy by 26 percentage points
- **Trade-off**: FLAN-T5 better generalization, RF faster inference

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/shivani3291-lab/news-article-classifier.git
   cd news-article-classifier
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Run Full Analysis in Jupyter Notebook
```bash
jupyter notebook notebooks/fully_cleaned_notebook.ipynb
```

### Using Google Colab (Recommended)
1. Upload the notebook to Google Colab
2. Set runtime to **T4 GPU** (Menu → Runtime → Change Runtime Type)
3. Run all cells sequentially

### Key Sections in Notebook
- **Cells 1-10**: Data loading and exploration
- **Cells 11-20**: EDA (visualizations, statistics)
- **Cells 21-40**: Random Forest model training and evaluation
- **Cells 41-55**: FLAN-T5 model with prompts
- **Cells 56-65**: Model comparison and final results

---

## Project Structure

```
news-article-classifier/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
│
├── notebooks/
│   └── fully_cleaned_notebook.ipynb   # Main analysis & modeling
│
├── data/
│   ├── article_data.csv               # Training dataset (4,000 articles)
│   └── data_info.txt                  # Data description
│
├── results/
│   ├── confusion_matrices/            # Confusion matrix plots
│   ├── performance_metrics.csv        # Metrics summary
│   ├── model_comparison.csv           # Cross-model comparison
│   └── roc_curves.png                 # ROC-AUC curves
│
└── src/
    ├── preprocessing.py               # Data cleaning utilities
    ├── model_training.py              # Model training functions
    ├── evaluation.py                  # Evaluation metrics
    └── utils.py                       # Helper functions
```

---

## Technologies Used

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Classical ML models and metrics

### NLP & Deep Learning
- **sentence-transformers** - Semantic embeddings (all-MiniLM-L6-v2)
- **transformers** - Pre-trained models (FLAN-T5)
- **torch** - Deep learning backend
- **bitsandbytes** - Memory-efficient inference (8-bit quantization)

### Visualization
- **matplotlib** - Static plots
- **seaborn** - Statistical visualizations
- **wordcloud** - Text frequency visualization

---

## Key Findings & Recommendations

### What Worked Well
1. **Sentence Embeddings** provide excellent semantic representation
2. **Hyperparameter tuning** (GridSearchCV) significantly improves RF performance
3. **Prompt engineering** is powerful - improved FLAN-T5 from 66% → 92% accuracy
4. **Class balance** in dataset ensures reliable metric interpretation

### Why FLAN-T5 Performs Best
- Better generalization on unseen data
- Leverages pre-trained knowledge on massive corpus
- Flexible architecture handles linguistic nuances
- Prompt-based approach scales to new categories without retraining

### Production Deployment Recommendations

#### Option 1: FLAN-T5 (Recommended for Accuracy)
- Pros: 92% accuracy, flexible, handles edge cases
- Cons: Slower inference (~1-2 sec per article), GPU needed
- Best for: Quality-critical applications

#### Option 2: Tuned Random Forest (Alternative for Speed)
- Pros: 88% accuracy, fast inference (~10ms), lightweight
- Cons: Less flexible, requires exact embedding format
- Best for: Real-time streaming, cost-sensitive deployments

### Next Steps
1. **A/B Testing**: Deploy both models and compare in production
2. **Monitoring**: Track performance drift over time
3. **Fine-tuning**: Collect mislabeled examples and retrain
4. **Feature Expansion**: Add metadata (publication date, source)
5. **Multi-language Support**: Extend to other languages

---

## Exploratory Data Analysis (EDA) Highlights

- **Category Distribution**: Perfectly balanced (1,000 per category)
- **Article Length**: 30-50 words average, right-skewed distribution
- **Vocabulary**: ~5,000+ unique terms
- **Word Frequency**: Top words vary by category (e.g., "player", "match" for Sports)

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add improvement"`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Author

**Shivani Chaudhari**  
NLP Enthusiast  
[LinkedIn](https://www.linkedin.com/in/shivani-chaudhari-335200192/) | [GitHub](https://github.com/shivani3291-lab)

---

## Contact

For questions, suggestions, or collaboration:
- Email: shivanichaudhari2001@gmail.com
- Open an Issue on GitHub
- LinkedIn

---

## Acknowledgments

- Dataset provided by E-news Express (fictional case study)
- Sentence-Transformers community
- Hugging Face for FLAN-T5 model
- Scikit-learn for ML tools

---

## References

1. Sentence-Transformers: [https://www.sbert.net/](https://www.sbert.net/)
2. FLAN-T5: [https://huggingface.co/google/flan-t5-large](https://huggingface.co/google/flan-t5-large)
3. Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
4. Random Forest: [https://en.wikipedia.org/wiki/Random_forest](https://en.wikipedia.org/wiki/Random_forest)

---

**Last Updated**: June 29, 2026  
**Status**: Production-Ready
