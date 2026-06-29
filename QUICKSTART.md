# Quick Start Guide

## Get Started in 5 Minutes

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run on Google Colab (Recommended)
```
1. Go to https://colab.research.google.com
2. Upload: notebooks/fully_cleaned_notebook.ipynb
3. Menu → Runtime → Change runtime type → T4 GPU
4. Run all cells (Ctrl+F9)
```

### 3. Run Locally with Jupyter
```bash
jupyter notebook notebooks/fully_cleaned_notebook.ipynb
```

### 4. View Results
- Accuracy: 92% (FLAN-T5 with improved prompt)
- Confusion matrices and plots in the notebook
- Detailed metrics in `results/` folder

---

## Project Structure

```
news-article-classifier/
├── README.md                          ← Full documentation
├── requirements.txt                   ← Dependencies
├── QUICKSTART.md                      ← This file
├── .gitignore                         ← Git ignore rules
│
├── notebooks/
│   └── fully_cleaned_notebook.ipynb   ← Main analysis
│
├── data/
│   ├── article_data.csv               ← Dataset (4,000 articles)
│   └── data_info.txt                  ← Dataset description
│
├── results/                           ← Output files
│   ├── confusion_matrices/
│   ├── performance_metrics.csv
│   └── roc_curves.png
│
└── src/                               ← Optional utilities
    ├── preprocessing.py
    ├── model_training.py
    ├── evaluation.py
    └── utils.py
```

---

## Model Performance at a Glance

| Model | Accuracy | Best For |
|-------|----------|----------|
| Random Forest | 88% | Speed |
| **FLAN-T5** | **92%** | **Production** |

---

## Key Files Explained

### `fully_cleaned_notebook.ipynb`
Complete analysis pipeline:
- Data loading & exploration
- EDA with visualizations
- 5 different models tested
- Comprehensive comparisons
- Final recommendations

### `article_data.csv`
- 4,000 news articles
- 4 balanced categories (World, Sports, Business, Sci/Tech)
- Ready to use, no preprocessing needed

### `requirements.txt`
All Python dependencies with specific versions for reproducibility

---

## What Each Model Does

### Random Forest (Tuned) - 88% Accuracy
```
Input: 384-dim embeddings → RF classifier → Category prediction
Pros: Fast, lightweight
Cons: Less flexible
```

### FLAN-T5 (Improved Prompt) - 92% Accuracy
```
Input: Article text → FLAN-T5 with guided prompt → Category
Pros: Best accuracy, flexible, handles edge cases
Cons: Slower, needs GPU for optimal speed
```

---

## Key Results

### Test Set Performance
- **Overall Accuracy**: 92%
- **Precision**: 0.91
- **Recall**: 0.92
- **F1-Score**: 0.92

### Category Performance
- Sports: 96% recall (best)
- World: 91% recall
- Business: 90% recall
- Sci/Tech: 89% recall

---

## Troubleshooting

### Memory Issues on Google Colab?
→ Restart kernel, reduce batch size, or use CPU

### Slow Inference?
→ Use Random Forest instead of FLAN-T5 (88% vs 92%)

### Missing Dependencies?
→ Run: `pip install -r requirements.txt --upgrade`

### GPU Not Available?
→ Works on CPU, but slower. Colab T4 GPU recommended.

---

## Understanding the Notebook

### Cells 1-10: Data Loading
- Load CSV
- Check shape and info
- Display sample articles

### Cells 11-20: EDA
- Category distribution
- Article length analysis
- Word cloud visualization

### Cells 21-40: Classical ML
- Sentence embeddings
- 3 Random Forest variations
- Performance comparisons

### Cells 41-55: Transformer Models
- FLAN-T5 setup
- Base prompt vs improved prompt
- Prompt engineering impact

### Cells 56-65: Final Evaluation
- Model comparison table
- Test set results
- Business recommendations

---

## Next Steps

1. **Customize for Your Data**
   - Replace `article_data.csv` with your own dataset
   - Adjust category labels in notebooks
   - Retrain models

2. **Deploy to Production**
   - Option 1: Streamlit web app
   - Option 2: FastAPI REST endpoint
   - Option 3: Batch processing script

3. **Improve Further**
   - Fine-tune FLAN-T5 on your data
   - Collect more articles
   - Add additional features (date, source)

4. **Share on GitHub**
   - Push code to GitHub
   - Add this README
   - Share link in portfolio

---

## Need Help?

- Check `README.md` for detailed documentation
- Review notebook comments for step-by-step explanation
- See `data/data_info.txt` for dataset details

---

## Quick Facts

- **Project Type**: Multi-class Text Classification (NLP)
- **Dataset**: 4,000 balanced articles
- **Best Accuracy**: 92% (FLAN-T5)
- **Technologies**: Python, Transformers, Scikit-learn, PyTorch
- **Training Time**: ~15 minutes on Colab T4 GPU
- **Inference Speed**: 1-2 sec per article (FLAN-T5) or 10ms (RF)

---

**Happy learning!**

For more info, see `README.md`
