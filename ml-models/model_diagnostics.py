import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import argparse

def load_model_and_data(model_path, data_path):
    """Load trained model and test data"""
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    # Prepare features (same as training)
    df['actual_return'] = df['close'].pct_change().shift(-1)
    df['label'] = (df['actual_return'] > 0).astype(int)
    df['return'] = df['close'].pct_change()
    df['return_volatility'] = df['return'].rolling(window=5).std()
    df.dropna(inplace=True)
    
    meta_features = [
        'return', 'return_volatility',
        'regime_hmm', 'bull_prob_hmm',
        'regime_gmm', 'bull_prob_gmm',
        'volatility_5', 'momentum',
        'regime_hmm_vol', 'regime_gmm_vol', 'regime_agreement'
    ]
    meta_features = [f for f in meta_features if f in df.columns]
    
    X = df[meta_features]
    y = df['label']
    
    # Use last 20% as test set (same as training)
    test_size = int(len(X) * 0.2)
    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]
    
    return model, X_test, y_test, meta_features

def analyze_probability_distribution(y_proba, model_name):
    """Analyze probability distribution for class 1"""
    print(f"\n=== {model_name} Probability Distribution Analysis ===")
    
    # Percentiles
    percentiles = np.percentile(y_proba, [5, 10, 25, 50, 75, 90, 95, 99])
    print(f"Probability percentiles for class 1:")
    for p, val in zip([5, 10, 25, 50, 75, 90, 95, 99], percentiles):
        print(f"  {p}th percentile: {val:.4f}")
    
    # Count predictions at different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"\nPredictions at different thresholds:")
    for thresh in thresholds:
        pred_count = (y_proba > thresh).sum()
        pred_pct = (pred_count / len(y_proba)) * 100
        print(f"  Threshold {thresh}: {pred_count} predictions ({pred_pct:.1f}%)")
    
    # Confidence analysis
    high_conf_0 = (y_proba < 0.1).sum()
    high_conf_1 = (y_proba > 0.9).sum()
    uncertain = ((y_proba >= 0.1) & (y_proba <= 0.9)).sum()
    
    print(f"\nConfidence breakdown:")
    print(f"  High confidence class 0 (< 0.1): {high_conf_0} ({high_conf_0/len(y_proba)*100:.1f}%)")
    print(f"  High confidence class 1 (> 0.9): {high_conf_1} ({high_conf_1/len(y_proba)*100:.1f}%)")
    print(f"  Uncertain (0.1-0.9): {uncertain} ({uncertain/len(y_proba)*100:.1f}%)")

def plot_threshold_analysis(y_test, y_proba, model_name):
    """Plot ROC curve and threshold analysis"""
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    
    # Threshold analysis
    thresholds = np.arange(0.1, 0.9, 0.05)
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        y_pred = (y_proba > thresh).astype(int)
        accuracy = (y_pred == y_test).mean()
        accuracies.append(accuracy)
        
        # Calculate precision, recall, F1
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'{model_name} - ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color='blue', lw=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'{model_name} - Precision-Recall Curve')
    ax2.grid(True)
    
    # Threshold vs Metrics
    ax3.plot(thresholds, accuracies, label='Accuracy', marker='o')
    ax3.plot(thresholds, f1_scores, label='F1 Score', marker='s')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Score')
    ax3.set_title(f'{model_name} - Threshold vs Metrics')
    ax3.legend()
    ax3.grid(True)
    
    # Precision vs Recall by Threshold
    ax4.plot(thresholds, precisions, label='Precision', marker='o')
    ax4.plot(thresholds, recalls, label='Recall', marker='s')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title(f'{model_name} - Precision vs Recall by Threshold')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Threshold analysis plots saved as '{model_name.lower().replace(' ', '_')}_threshold_analysis.png'")
    
    # Find optimal threshold (max F1)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[optimal_idx]
    print(f"\nOptimal threshold (max F1): {optimal_thresh:.3f}")
    print(f"  F1 Score: {f1_scores[optimal_idx]:.3f}")
    print(f"  Precision: {precisions[optimal_idx]:.3f}")
    print(f"  Recall: {recalls[optimal_idx]:.3f}")
    print(f"  Accuracy: {accuracies[optimal_idx]:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze model confidence and threshold sensitivity")
    parser.add_argument("--data", default="merged_with_regime_features.csv", help="Input data file")
    parser.add_argument("--rf_model", default="corrective_ai_model.pkl", help="Random Forest model file")
    parser.add_argument("--lgbm_model", default="corrective_ai_lgbm.pkl", help="LightGBM model file")
    args = parser.parse_args()
    
    # Analyze Random Forest
    try:
        print("🔍 Analyzing Random Forest model...")
        rf_model, X_test, y_test, features = load_model_and_data(args.rf_model, args.data)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        analyze_probability_distribution(rf_proba, "Random Forest")
        plot_threshold_analysis(y_test, rf_proba, "Random Forest")
    except Exception as e:
        print(f"❌ Error analyzing Random Forest: {e}")
    
    # Analyze LightGBM
    try:
        print("\n🔍 Analyzing LightGBM model...")
        lgbm_model, X_test, y_test, features = load_model_and_data(args.lgbm_model, args.data)
        lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
        analyze_probability_distribution(lgbm_proba, "LightGBM")
        plot_threshold_analysis(y_test, lgbm_proba, "LightGBM")
    except Exception as e:
        print(f"❌ Error analyzing LightGBM: {e}")
    
    print(f"\n✅ Model diagnostics complete!")
    print(f"📊 Check the generated PNG files for detailed threshold analysis")

if __name__ == "__main__":
    main() 