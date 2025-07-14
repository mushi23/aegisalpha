#!/usr/bin/env python3
"""
LightGBM Hyperparameter Tuning Script
Performs RandomizedSearchCV on the current full technical feature set.
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import joblib
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="LightGBM Hyperparameter Tuning")
parser.add_argument('--data', type=str, default='merged_with_regime_features.csv', help='Input CSV file')
parser.add_argument('--feature_list', type=str, default='feature_list_full_technical.txt', help='Feature list file')
parser.add_argument('--n_iter', type=int, default=20, help='Number of random search iterations')
parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
args = parser.parse_args()

# Load data
print(f"📊 Loading data from {args.data}")
df = pd.read_csv(args.data)

# Create label if needed
if 'label' not in df.columns and 'return' in df.columns:
    df['label'] = (df['return'] > 0).astype(int)

# Load feature list
with open(args.feature_list, 'r') as f:
    features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
features = [f for f in features if f in df.columns]
print(f"✅ Using {len(features)} features: {features}")

X = df[features]
y = df['label']

# LightGBM parameter grid
param_dist = {
    'num_leaves': [15, 31, 63, 127],
    'max_depth': [3, 5, 7, 10, -1],
    'min_child_weight': [1, 3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 0.5],
    'reg_lambda': [0, 0.01, 0.1, 0.5]
}

# Scorer
scorer = make_scorer(f1_score, average='binary')

# Model
lgbm = LGBMClassifier(objective='binary', class_weight='balanced', random_state=42, verbosity=-1)

# CV
cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

# Randomized search
print("🔍 Starting RandomizedSearchCV...")
search = RandomizedSearchCV(
    lgbm,
    param_distributions=param_dist,
    n_iter=args.n_iter,
    scoring=scorer,
    n_jobs=-1,
    cv=cv,
    verbose=2,
    random_state=42,
    return_train_score=True
)
search.fit(X, y)

print("\n✅ Hyperparameter tuning complete!")
print(f"Best F1 Score: {search.best_score_:.4f}")
print(f"Best Params: {search.best_params_}")

# Save best model
joblib.dump(search.best_estimator_, 'lgbm_best_model.pkl')
print("✅ Best model saved as lgbm_best_model.pkl")

# Save results to CSV
results_df = pd.DataFrame(search.cv_results_)
results_df.to_csv('lgbm_hyperparam_tuning_results.csv', index=False)
print("✅ All tuning results saved as lgbm_hyperparam_tuning_results.csv") 