<div align="center">

<img src="https://img.icons8.com/fluency/120/000000/home.png" alt="House Price Prediction" width="120"/>

# 🏡 Advanced House Price Prediction

> **Predicting Real Estate Value with Machine Learning — Feature Engineering, Regression & Ensemble Modeling**

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data--Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Aditya152602/Advanced-House-Price-Prediction?style=for-the-badge&logo=github)](https://github.com/Aditya152602/Advanced-House-Price-Prediction)

<br/>

*A full end-to-end machine learning pipeline that predicts residential house prices using the Ames Housing Dataset — featuring deep EDA, advanced feature engineering, and stacked ensemble models.*

</div>

---

## 🔍 Problem Statement

> *"Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad."*

Yet, these hidden factors often dominate price negotiations more than bedrooms or backyard size.

**The Challenge:** Using **79 explanatory variables** describing nearly every aspect of residential homes in Ames, Iowa — predict the final `SalePrice` of each home as accurately as possible.

**Evaluation Metric:** Root Mean Squared Log Error (RMSLE)
```
RMSLE = sqrt( (1/n) * Σ (log(ŷ+1) - log(y+1))² )
```

---

## ✨ Key Highlights

| Feature | Detail |
|---------|--------|
| 📊 Dataset | Ames Housing — 1460 training samples, 79 features |
| 🧹 Preprocessing | Missing value imputation, outlier removal, skewness correction |
| 🔧 Feature Engineering | Label encoding, rare category grouping, temporal features |
| 🤖 Models Used | Linear, Ridge, Lasso, Random Forest, XGBoost, Gradient Boosting |
| 🏆 Best Approach | Stacked Ensemble (Meta-learner) |
| 📉 Metric | RMSLE / Log-RMSE |

---

## 📁 Project Structure

```
Advanced-House-Price-Prediction/
│
├── 📁 data/
│   ├── train.csv                  # Training dataset (Kaggle)
│   ├── test.csv                   # Test dataset (Kaggle)
│   └── data_description.txt       # Feature documentation
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb   # Feature creation & transformation
│   ├── 03_Model_Training.ipynb    # Training multiple ML models
│   ├── 04_Hyperparameter_Tuning.ipynb # GridSearchCV / RandomizedSearchCV
│   └── 05_Ensemble_Stacking.ipynb # Final stacked model
│
├── 📁 models/
│   ├── lasso_model.pkl
│   ├── ridge_model.pkl
│   ├── xgboost_model.pkl
│   └── stacked_model.pkl
│
├── 📁 outputs/
│   ├── submission.csv             # Kaggle submission file
│   └── feature_importance.png    # Feature importance plot
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python --version     # Python 3.7+
pip --version        # pip 21+
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aditya152602/Advanced-House-Price-Prediction.git

# 2. Move into the project
cd Advanced-House-Price-Prediction

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook
```

### Run in Order

```bash
# Follow the notebook sequence for best results:
01_EDA.ipynb  →  02_Feature_Engineering.ipynb  →  03_Model_Training.ipynb  →  05_Ensemble_Stacking.ipynb
```

---

## 🧪 ML Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    END-TO-END ML PIPELINE                           │
│                                                                     │
│  Raw Data                                                           │
│     │                                                               │
│     ▼                                                               │
│  ┌──────────────────┐                                               │
│  │  DATA CLEANING   │  → Handle nulls, fix dtypes, drop outliers   │
│  └────────┬─────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────┐                                               │
│  │   EDA & VISUAL   │  → Distributions, correlations, heatmaps     │
│  └────────┬─────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────┐                                               │
│  │ FEATURE ENGINEER │  → Encode, transform, create new features    │
│  └────────┬─────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────────────────────────────┐                  │
│  │           MODEL TRAINING                     │                  │
│  │                                              │                  │
│  │  Linear │ Ridge │ Lasso │ Random Forest      │                  │
│  │  XGBoost │ GradBoost │ ElasticNet            │                  │
│  └────────────────────┬─────────────────────────┘                  │
│                       │                                             │
│                       ▼                                             │
│  ┌──────────────────────────────────────────────┐                  │
│  │       STACKING ENSEMBLE (Meta-learner)       │                  │
│  │  Best 5 models → Lasso meta-learner          │                  │
│  └────────────────────┬─────────────────────────┘                  │
│                       │                                             │
│                       ▼                                             │
│              📄 submission.csv  →  Kaggle Upload                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Exploratory Data Analysis

Key insights uncovered during EDA:

- 🏠 **SalePrice** is right-skewed → log-transformed for normality
- 📐 **GrLivArea** (above-ground living area) has the strongest positive correlation with price
- 📅 **YearBuilt** and **YearRemodAdd** are strong temporal predictors
- 🚗 **GarageArea**, **TotalBsmtSF** show high positive correlation
- ❌ Several features have **>15% missing values** — handled via domain-aware imputation

```python
# Log transformation to normalize target variable
import numpy as np
df['SalePrice'] = np.log1p(df['SalePrice'])
```

---

## 🔧 Feature Engineering Techniques

```
✅ Missing Value Imputation      →  Median (numerical) / Mode (categorical)
✅ Rare Label Encoding           →  Group categories with <1% frequency into "Other"
✅ Temporal Feature Extraction   →  Age of house = YrSold - YearBuilt
✅ Log Transformation            →  Fix skewed numerical features
✅ One-Hot Encoding              →  pd.get_dummies() for categorical variables
✅ Feature Scaling               →  StandardScaler for linear models
✅ New Features                  →  TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

---

## 🤖 Models & Results

| Model | CV RMSE | Notes |
|-------|---------|-------|
| Linear Regression | ~0.185 | Baseline |
| Ridge Regression | ~0.163 | L2 regularization |
| Lasso Regression | ~0.167 | L1, automatic feature selection |
| ElasticNet | ~0.166 | L1 + L2 combined |
| Random Forest | ~0.179 | Handles non-linearity |
| Gradient Boosting | ~0.163 | Strong single model |
| **XGBoost** | **~0.150** | **Best single model** |
| **Stacked Ensemble** | **~0.118** | **🏆 Best overall** |

> Lower RMSE = Better prediction accuracy

---

## 🏆 Stacking Strategy

```python
from sklearn.ensemble import StackingRegressor

# Base learners
base_models = [
    ('xgb',   XGBRegressor(...)),
    ('ridge',  RidgeCV(...)),
    ('lasso',  LassoCV(...)),
    ('gbr',    GradientBoostingRegressor(...)),
    ('rf',     RandomForestRegressor(...))
]

# Meta learner
meta_learner = LassoCV()

# Stack them up!
stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)
```

---

## 📦 Tech Stack

```
Language     →  Python 3.x
EDA & Viz    →  Pandas, NumPy, Matplotlib, Seaborn
ML Models    →  Scikit-Learn, XGBoost, LightGBM
Notebooks    →  Jupyter Notebook
Evaluation   →  RMSLE, Cross-Validation (K-Fold, n=5)
Dataset      →  Kaggle — Ames Housing Dataset
```

---

## 📈 Top Predictive Features

```
🥇 OverallQual     — Overall material and finish quality
🥈 GrLivArea       — Above grade (ground) living area sq ft
🥉 GarageCars      — Size of garage in car capacity
4️⃣  GarageArea      — Size of garage in square feet
5️⃣  TotalBsmtSF     — Total square feet of basement area
6️⃣  1stFlrSF        — First floor square feet
7️⃣  FullBath        — Full bathrooms above grade
8️⃣  TotRmsAbvGrd    — Total rooms above grade
9️⃣  YearBuilt       — Original construction date
🔟  YearRemodAdd    — Remodel date
```

---

## 🤝 Contributing

Contributions, ideas, and improvements are very welcome!

1. Fork the repository
2. Create your branch: `git checkout -b feature/improve-model`
3. Make changes with proper documentation
4. Commit: `git commit -m "Improve: added LightGBM to ensemble"`
5. Push and open a **Pull Request**

---

## 📚 References & Resources

| Resource | Link |
|----------|------|
| 🏠 Kaggle Competition | [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| 📖 Ames Housing Dataset Paper | [Dean De Cock — JSE Publication](http://jse.amstat.org/v19n3/decock.pdf) |
| 📘 XGBoost Docs | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
| 🔬 Feature Engineering Guide | [Baeldung ML Blog](https://www.baeldung.com/) |
| 🎓 Scikit-Learn Docs | [scikit-learn.org](https://scikit-learn.org/stable/) |

---

## 👨‍💻 Author

<div align="center">

**Aditya**

[![GitHub](https://img.shields.io/badge/GitHub-Aditya152602-black?style=flat-square&logo=github)](https://github.com/Aditya152602)

*Turning raw data into real-world predictions — one model at a time.* 🏡📊

</div>

---

## ⭐ Support

If this project helped you understand machine learning or real estate prediction — drop a **⭐ star**!
It motivates updates, new features, and helps others discover this repo.

---

<div align="center">

**Built with 🐍 Python | 📊 Data | 🤖 Machine Learning**

```python
print(f"Predicted Price: ${model.predict(your_house_features)[0]:,.0f} 🏡")
```

</div>
