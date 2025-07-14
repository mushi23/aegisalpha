import pandas as pd
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import argparse

def get_feature_set(feature_set_name):
    """Get feature set based on name"""
    if feature_set_name == "basic":
        return [
            'regime_hmm', 'regime_gmm', 'bull_prob_hmm', 'bull_prob_gmm',
            'regime_duration_hmm', 'regime_duration_gmm'
        ]
    elif feature_set_name == "enhanced":
        return [
            'regime_hmm', 'regime_gmm', 'bull_prob_hmm', 'bull_prob_gmm',
            'regime_duration_hmm', 'regime_duration_gmm', 'regime_transition_hmm', 'regime_transition_gmm',
            'bull_prob_avg', 'bull_prob_ratio', 'bull_prob_std', 'bull_prob_ma5', 'bull_prob_ma10',
            'regime_volatility_hmm', 'regime_volatility_gmm', 'regime_momentum_hmm', 'regime_momentum_gmm',
            'regime_interaction', 'regime_stability', 'regime_consensus',
            'volatility_5', 'volatility_10', 'volatility_20', 'momentum_5', 'momentum_10', 'momentum_20',
            'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_width',
            'hour', 'day_of_week', 'month', 'quarter'
        ]
    elif feature_set_name == "no_bull_prob":
        # Remove bull_prob_hmm completely, rely on other regime features
        return [
            'regime_hmm', 'regime_gmm', 'bull_prob_gmm',
            'regime_duration_hmm', 'regime_duration_gmm', 'regime_transition_hmm', 'regime_transition_gmm',
            'bull_prob_avg', 'bull_prob_ratio', 'bull_prob_std', 'bull_prob_ma5', 'bull_prob_ma10',
            'regime_volatility_hmm', 'regime_volatility_gmm', 'regime_momentum_hmm', 'regime_momentum_gmm',
            'regime_interaction', 'regime_stability', 'regime_consensus',
            'volatility_5', 'volatility_10', 'volatility_20', 'momentum_5', 'momentum_10', 'momentum_20',
            'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_width',
            'hour', 'day_of_week', 'month', 'quarter'
        ]
    else:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train Corrective AI model with regime features")
parser.add_argument("--input", type=str, default="merged_with_regime_features.csv", help="Input CSV file")
parser.add_argument("--lgbm", action="store_true", help="Use LightGBM instead of Random Forest")
parser.add_argument("--feature_list", type=str, default=None, help="Path to feature list file (one feature per line)")
parser.add_argument("--feature_set", type=str, default="enhanced", choices=["basic", "enhanced", "no_bull_prob"], help="Feature set to use")
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.input)

# Create label: was the next return positive?
df['actual_return'] = df['close'].pct_change().shift(-1)
df['label'] = (df['actual_return'] > 0).astype(int)

# Add regime-based meta-features
df['return'] = df['close'].pct_change()
df['return_volatility'] = df['return'].rolling(window=5).std()

# Drop NaNs from pct_change
df.dropna(inplace=True)

# --- Feature Diagnostics ---
print("=== Regime/Volatility Feature Diagnostics ===")
if 'regime_hmm' in df.columns:
    print("regime_hmm value counts:")
    print(df['regime_hmm'].value_counts())
if 'regime_gmm' in df.columns:
    print("\nregime_gmm value counts:")
    print(df['regime_gmm'].value_counts())
if 'bull_prob_hmm' in df.columns:
    print("\nbull_prob_hmm stats:")
    print(df['bull_prob_hmm'].describe())
if 'bull_prob_gmm' in df.columns:
    print("\nbull_prob_gmm stats:")
    print(df['bull_prob_gmm'].describe())
if 'volatility_5' in df.columns:
    print("\nvolatility_5 stats:")
    print(df['volatility_5'].describe())
if 'momentum' in df.columns:
    print("\nmomentum stats:")
    print(df['momentum'].describe())
if 'regime_hmm_vol' in df.columns:
    print("\nregime_hmm_vol stats:")
    print(df['regime_hmm_vol'].describe())
if 'regime_gmm_vol' in df.columns:
    print("\nregime_gmm_vol stats:")
    print(df['regime_gmm_vol'].describe())
if 'regime_agreement' in df.columns:
    print("\nregime_agreement value counts:")
    print(df['regime_agreement'].value_counts())

# Regime transitions
if 'regime_hmm' in df.columns:
    regime_transitions = (df['regime_hmm'] != df['regime_hmm'].shift(1)).sum()
    print(f"\nRegime HMM transitions: {regime_transitions}")

# Correlation analysis
corr_cols = [c for c in ['return', 'regime_hmm', 'bull_prob_hmm', 'volatility_5', 'momentum', 'regime_hmm_vol', 'regime_gmm_vol', 'regime_agreement'] if c in df.columns]
if len(corr_cols) > 1:
    print("\nCorrelation with return:")
    print(df[corr_cols].corr())

# If a feature list is provided, use only those features
if args.feature_list:
    with open(args.feature_list, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip() in df.columns]
    print(f"\n📋 Using {len(selected_features)} features from {args.feature_list}")
else:
    # Use feature set based on argument
    feature_set_features = get_feature_set(args.feature_set)
    selected_features = [col for col in feature_set_features if col in df.columns]
    print(f"\n🔧 Using feature set: {args.feature_set} ({len(selected_features)} features)")
    print(f"   Features: {', '.join(selected_features)}")
    
    # Check for missing features
    missing_features = [col for col in feature_set_features if col not in df.columns]
    if missing_features:
        print(f"⚠️  Missing features: {', '.join(missing_features)}")

X = df[selected_features]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, stratify=None
)

# Conservative resampling: random undersample class 1, then moderate SMOTE for class 0
print("Class balance before resampling:")
print(pd.Series(y_train).value_counts())

# Dynamically determine available samples
n_class_1 = y_train.value_counts().get(1, 0)
n_class_0 = y_train.value_counts().get(0, 0)

# Step 1: Random undersample class 1 to at most 30k (or less if not available)
rus_target_1 = min(n_class_1, 30000)
rus = RandomUnderSampler(sampling_strategy={1: rus_target_1, 0: n_class_0}, random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)
print("Class balance after undersampling:")
print(pd.Series(y_rus).value_counts())

# Step 2: SMOTE for class 0 up to at most 20k (or less if not available)
smote_target_0 = min(y_rus.value_counts().get(0, 0), 20000)
smote = SMOTE(sampling_strategy={0: smote_target_0}, random_state=42)
X_res, y_res = smote.fit_resample(X_rus, y_rus)
print("Class balance after SMOTE:")
print(pd.Series(y_res).value_counts())

# Convert to numpy arrays for cross-validation
X_res_np = np.array(X_res)
y_res_np = np.array(y_res)

# Model selection based on CLI flag
if args.lgbm:
    print("\n🚀 Using LightGBM model...")
    try:
        from lightgbm import LGBMClassifier
        model_class = LGBMClassifier
        model_params = {
            'n_estimators': 100,
            'random_state': 42,
            'class_weight': 'balanced',
            'verbosity': -1,
            'objective': 'binary'
        }
        model_name = "LightGBM"
    except ImportError:
        print("❌ LightGBM not installed. Falling back to Random Forest.")
        print("Install with: pip install lightgbm")
        model_class = BalancedRandomForestClassifier
        model_params = {
            'n_estimators': 100,
            'random_state': 42,
            'sampling_strategy': 'auto',
            'replacement': False,
            'bootstrap': True
        }
        model_name = "Random Forest"
else:
    print("\n🌲 Using Random Forest model...")
    model_class = BalancedRandomForestClassifier
    model_params = {
        'n_estimators': 100,
        'random_state': 42,
        'sampling_strategy': 'auto',
        'replacement': False,
        'bootstrap': True
    }
    model_name = "Random Forest"

# Stratified K-Fold Cross-Validation with detailed logging
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"\nStratified 5-Fold Cross-Validation Results ({model_name}):")

# Store fold results for analysis
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_res_np, y_res_np), 1):
    X_fold_train, X_fold_val = X_res_np[train_idx], X_res_np[val_idx]
    y_fold_train, y_fold_val = y_res_np[train_idx], y_res_np[val_idx]
    
    # Train model
    model = model_class(**model_params)
    model.fit(X_fold_train, y_fold_train)
    
    # Get predictions and probabilities
    y_fold_pred = model.predict(X_fold_val)
    y_fold_proba = model.predict_proba(X_fold_val)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_fold_val, y_fold_pred, zero_division=0)
    recall = recall_score(y_fold_val, y_fold_pred, zero_division=0)
    f1 = f1_score(y_fold_val, y_fold_pred, zero_division=0)
    accuracy = accuracy_score(y_fold_val, y_fold_pred)
    
    # Store results
    fold_results.append({
        'fold': fold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'y_true': y_fold_val,
        'y_pred': y_fold_pred,
        'y_proba': y_fold_proba
    })
    
    print(f"\nFold {fold} Results:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Classification Report:")
    print(classification_report(y_fold_val, y_fold_pred))

# Calculate average metrics across folds
avg_precision = np.mean([r['precision'] for r in fold_results])
avg_recall = np.mean([r['recall'] for r in fold_results])
avg_f1 = np.mean([r['f1'] for r in fold_results])
avg_accuracy = np.mean([r['accuracy'] for r in fold_results])

print(f"\n📊 Cross-Validation Summary ({model_name}):")
print(f"  Average Precision: {avg_precision:.3f}")
print(f"  Average Recall: {avg_recall:.3f}")
print(f"  Average F1 Score: {avg_f1:.3f}")
print(f"  Average Accuracy: {avg_accuracy:.3f}")

# Plot precision/recall curves for each fold
plt.figure(figsize=(12, 8))
for i, result in enumerate(fold_results):
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, _ = precision_recall_curve(result['y_true'], result['y_proba'])
    plt.plot(recall_curve, precision_curve, alpha=0.7, label=f'Fold {i+1}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curves by Fold ({model_name})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'precision_recall_curves_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Precision-Recall curves saved as precision_recall_curves_{model_name.lower().replace(' ', '_')}.png")

# Final model training on all resampled data
model = model_class(**model_params)
model.fit(X_res, y_res)

# Evaluation on test set
# Default threshold
y_pred = model.predict(X_test)
print(f"\nTest set confusion matrix (default threshold) - {model_name}:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Multiple threshold analysis with realistic thresholds
print(f"\n=== {model_name} Threshold Analysis ===")
y_proba = model.predict_proba(X_test)[:, 1]

# Test multiple realistic thresholds
realistic_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
print(f"Testing realistic thresholds: {realistic_thresholds}")

for thresh in realistic_thresholds:
    y_pred_thresh = (y_proba > thresh).astype(int)
    
    # Calculate metrics
    tp = ((y_pred_thresh == 1) & (y_test == 1)).sum()
    fp = ((y_pred_thresh == 1) & (y_test == 0)).sum()
    tn = ((y_pred_thresh == 0) & (y_test == 0)).sum()
    fn = ((y_pred_thresh == 0) & (y_test == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Calculate prediction rate
    pred_rate = (tp + fp) / len(y_test) * 100
    
    print(f"\nThreshold {thresh}:")
    print(f"  Predictions: {tp + fp}/{len(y_test)} ({pred_rate:.1f}%)")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Confusion Matrix:")
    print(f"    [[{tn:4d} {fp:4d}]")
    print(f"     [{fn:4d} {tp:4d}]]")

# Find optimal threshold (max F1 score)
print(f"\n=== {model_name} Optimal Threshold Search ===")
thresholds = np.arange(0.05, 0.95, 0.05)
f1_scores = []
precisions = []
recalls = []
accuracies = []
prediction_rates = []

for thresh in thresholds:
    y_pred_thresh = (y_proba > thresh).astype(int)
    
    tp = ((y_pred_thresh == 1) & (y_test == 1)).sum()
    fp = ((y_pred_thresh == 1) & (y_test == 0)).sum()
    tn = ((y_pred_thresh == 0) & (y_test == 0)).sum()
    fn = ((y_pred_thresh == 0) & (y_test == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    pred_rate = (tp + fp) / len(y_test) * 100
    
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)
    prediction_rates.append(pred_rate)

# Find optimal threshold
optimal_idx = np.argmax(f1_scores)
optimal_thresh = thresholds[optimal_idx]

print(f"Optimal threshold (max F1): {optimal_thresh:.3f}")
print(f"  F1 Score: {f1_scores[optimal_idx]:.3f}")
print(f"  Precision: {precisions[optimal_idx]:.3f}")
print(f"  Recall: {recalls[optimal_idx]:.3f}")
print(f"  Prediction Rate: {prediction_rates[optimal_idx]:.1f}%")

# Export threshold analysis to CSV
threshold_df = pd.DataFrame({
    'threshold': thresholds,
    'precision': precisions,
    'recall': recalls,
    'f1_score': f1_scores,
    'accuracy': accuracies,
    'prediction_rate_pct': prediction_rates
})

csv_filename = f"threshold_analysis_{model_name.lower().replace(' ', '_')}.csv"
threshold_df.to_csv(csv_filename, index=False)
print(f"✅ Threshold analysis exported to {csv_filename}")

# Trading strategy recommendations
print(f"\n=== {model_name} Trading Strategy Recommendations ===")

# Conservative strategy (high precision, low recall)
conservative_idx = np.argmax(precisions)
conservative_thresh = thresholds[conservative_idx]
print(f"Conservative (max precision): threshold {conservative_thresh:.3f}")
print(f"  Precision: {precisions[conservative_idx]:.3f}")
print(f"  Recall: {recalls[conservative_idx]:.3f}")
print(f"  Prediction Rate: {prediction_rates[conservative_idx]:.1f}%")

# Balanced strategy (max F1)
print(f"Balanced (max F1): threshold {optimal_thresh:.3f}")
print(f"  F1 Score: {f1_scores[optimal_idx]:.3f}")
print(f"  Precision: {precisions[optimal_idx]:.3f}")
print(f"  Recall: {recalls[optimal_idx]:.3f}")

# Aggressive strategy (high recall, lower precision)
# Find threshold where recall > 0.8 and precision is reasonable
aggressive_candidates = [(i, t) for i, t in enumerate(thresholds) if recalls[i] > 0.8 and precisions[i] > 0.4]
if aggressive_candidates:
    aggressive_idx, aggressive_thresh = max(aggressive_candidates, key=lambda x: precisions[x[0]])
    print(f"Aggressive (recall > 0.8): threshold {aggressive_thresh:.3f}")
    print(f"  Precision: {precisions[aggressive_idx]:.3f}")
    print(f"  Recall: {recalls[aggressive_idx]:.3f}")
    print(f"  Prediction Rate: {prediction_rates[aggressive_idx]:.1f}%")
else:
    print("No aggressive strategy found (no threshold with recall > 0.8 and precision > 0.4)")

# Test optimal threshold
y_pred_optimal = (y_proba > optimal_thresh).astype(int)
print(f"\nTest set confusion matrix (optimal threshold {optimal_thresh:.3f}) - {model_name}:")
print(confusion_matrix(y_test, y_pred_optimal))
print(classification_report(y_test, y_pred_optimal))

# Save model with appropriate name
model_filename = "corrective_ai_lgbm.pkl" if args.lgbm else "corrective_ai_model.pkl"
joblib.dump(model, model_filename)
print(f"✅ {model_name} model saved as {model_filename} with optimal threshold {optimal_thresh:.3f}.")

# Save optimal threshold info separately
threshold_info = {
    'optimal_threshold': optimal_thresh,
    'feature_names': X.columns.tolist(),
    'model_name': model_name,
    'f1_score': f1_scores[optimal_idx],
    'precision': precisions[optimal_idx],
    'recall': recalls[optimal_idx]
}
threshold_filename = "corrective_ai_lgbm_threshold.json" if args.lgbm else "corrective_ai_threshold.json"
import json
with open(threshold_filename, 'w') as f:
    json.dump(threshold_info, f, indent=2)
print(f"✅ Threshold info saved as {threshold_filename}.")

# Feature importance
importances = model.feature_importances_
feat_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feat_names, importances)
plt.title(f"Corrective AI Feature Importance ({model_name})")
plt.grid(True)
plt.tight_layout()
plot_filename = "corrective_ai_lgbm_feature_importance.png" if args.lgbm else "corrective_ai_feature_importance.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Feature importance plot saved as '{plot_filename}'")

# SHAP Analysis for LightGBM models
if args.lgbm and model_name == "LightGBM":
    print(f"\n🔍 Running SHAP analysis for {model_name}...")
    try:
        import shap
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot (global importance) - Bar chart
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig("shap_feature_importance_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ SHAP feature importance bar plot saved")
        
        # Beeswarm plot (distribution of SHAP values)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"SHAP Values Distribution - {model_name}")
        plt.tight_layout()
        plt.savefig("shap_beeswarm_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ SHAP beeswarm plot saved")
        
        # Dependence plot for top feature
        feature_importance = np.abs(shap_values).mean(0)
        top_feature_idx = feature_importance.argmax()
        top_feature = X_test.columns[top_feature_idx]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_feature_idx, shap_values, X_test, show=False)
        plt.title(f"SHAP Dependence Plot - {top_feature}")
        plt.tight_layout()
        plt.savefig(f"shap_dependence_{top_feature}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP dependence plot for {top_feature} saved")
        
        # Force plot for a single sample (HTML)
        shap.initjs()
        sample_idx = 0  # First sample
        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_values[sample_idx], 
            X_test.iloc[sample_idx],
            show=False
        )
        shap.save_html("shap_force_plot.html", force_plot)
        print("✅ SHAP force plot (HTML) saved")
        
        # Feature importance ranking
        feature_importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'shap_importance': feature_importance
        }).sort_values('shap_importance', ascending=False)
        
        print(f"\n📊 SHAP Feature Importance Ranking:")
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<20} {row['shap_importance']:.4f}")
        
        # Save feature importance to CSV
        feature_importance_df.to_csv("shap_feature_importance.csv", index=False)
        print("✅ SHAP feature importance CSV saved")
        
        # Model conservatism analysis
        print(f"\n🔍 Model Conservatism Analysis:")
        print(f"  Expected value (baseline): {explainer.expected_value:.4f}")
        print(f"  Mean SHAP value: {np.mean(shap_values):.4f}")
        print(f"  SHAP value std: {np.std(shap_values):.4f}")
        
        # Check if model is biased towards one class
        positive_shap = (shap_values > 0).sum()
        negative_shap = (shap_values < 0).sum()
        total_shap = len(shap_values.flatten())
        
        print(f"  Positive SHAP contributions: {positive_shap/total_shap*100:.1f}%")
        print(f"  Negative SHAP contributions: {negative_shap/total_shap*100:.1f}%")
        
    except Exception as e:
        print(f"❌ SHAP analysis failed: {e}")
        print("   This is normal if SHAP is not properly installed or if there are memory constraints") 