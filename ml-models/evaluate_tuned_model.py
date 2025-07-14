#!/usr/bin/env python3
"""
Evaluate Tuned LightGBM Model
Compares the tuned model with the previous version and generates comprehensive analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_models_and_data():
    """Load both models and prepare data"""
    print("🔄 Loading models and data...")
    
    # Load models
    try:
        tuned_model = joblib.load('lgbm_best_model.pkl')
        print("✅ Tuned model loaded")
    except:
        print("❌ Tuned model not found. Please run hyperparameter tuning first.")
        return None, None, None, None
    
    try:
        previous_model = joblib.load('corrective_ai_lgbm.pkl')
        print("✅ Previous model loaded")
    except:
        print("⚠️ Previous model not found. Will only evaluate tuned model.")
        previous_model = None
    
    # Load data
    df = pd.read_csv('merged_with_regime_features.csv')
    
    # Create label if needed
    if 'label' not in df.columns and 'return' in df.columns:
        df['label'] = (df['return'] > 0).astype(int)
    
    # Load feature list
    with open('feature_list_full_technical.txt', 'r') as f:
        features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df['label']
    
    # Train-test split (same as before)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, stratify=None
    )
    
    print(f"✅ Data prepared: {len(features)} features, {len(X_test)} test samples")
    
    return tuned_model, previous_model, X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model"""
    print(f"\n🔍 Evaluating {model_name}...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    [[{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f"     [{cm[1,0]:4d} {cm[1,1]:4d}]]")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'confusion_matrix': cm
    }

def compare_models(tuned_results, previous_results):
    """Compare model performances"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_data = []
    for results in [tuned_results, previous_results]:
        if results:
            comparison_data.append({
                'Model': results['model_name'],
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Calculate improvements
    if len(comparison_data) == 2:
        tuned = comparison_data[0]
        previous = comparison_data[1]
        
        print(f"\n📈 Improvements:")
        print(f"  F1 Score: {tuned['F1 Score']:.4f} vs {previous['F1 Score']:.4f} (+{tuned['F1 Score'] - previous['F1 Score']:.4f})")
        print(f"  Accuracy: {tuned['Accuracy']:.4f} vs {previous['Accuracy']:.4f} (+{tuned['Accuracy'] - previous['Accuracy']:.4f})")
        print(f"  Precision: {tuned['Precision']:.4f} vs {previous['Precision']:.4f} (+{tuned['Precision'] - previous['Precision']:.4f})")
        print(f"  Recall: {tuned['Recall']:.4f} vs {previous['Recall']:.4f} (+{tuned['Recall'] - previous['Recall']:.4f})")
    
    return comparison_df

def plot_comparison(tuned_results, previous_results):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tuned vs Previous Model Comparison', fontsize=16, fontweight='bold')
    
    # Prepare data
    models = []
    metrics = []
    values = []
    
    for results in [tuned_results, previous_results]:
        if results:
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                models.append(results['model_name'])
                metrics.append(metric.upper())
                values.append(results[metric])
    
    # Create comparison dataframe
    plot_df = pd.DataFrame({
        'Model': models,
        'Metric': metrics,
        'Value': values
    })
    
    # 1. Bar plot comparison
    sns.barplot(data=plot_df, x='Metric', y='Value', hue='Model', ax=axes[0,0])
    axes[0,0].set_title('Performance Metrics Comparison')
    axes[0,0].set_ylim(0, 1)
    
    # 2. Confusion matrices
    if tuned_results:
        sns.heatmap(tuned_results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title(f'Confusion Matrix - {tuned_results["model_name"]}')
    
    if previous_results:
        sns.heatmap(previous_results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Reds', ax=axes[1,0])
        axes[1,0].set_title(f'Confusion Matrix - {previous_results["model_name"]}')
    
    # 3. Probability distributions
    if tuned_results and previous_results:
        axes[1,1].hist(tuned_results['y_proba'], alpha=0.7, label=tuned_results['model_name'], bins=50)
        axes[1,1].hist(previous_results['y_proba'], alpha=0.7, label=previous_results['model_name'], bins=50)
        axes[1,1].set_title('Prediction Probability Distributions')
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('tuned_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Comparison plots saved as tuned_model_comparison.png")

def run_shap_analysis(model, X_test, model_name):
    """Run SHAP analysis for the model"""
    print(f"\n🔍 Running SHAP analysis for {model_name}...")
    
    try:
        import shap
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'shap_importance_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP importance plot saved")
        
        # Feature importance ranking
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'shap_importance': feature_importance
        }).sort_values('shap_importance', ascending=False)
        
        print(f"\n📊 Top 10 SHAP Feature Importance ({model_name}):")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<20} {row['shap_importance']:.4f}")
        
        # Save importance to CSV
        importance_df.to_csv(f'shap_importance_{model_name.lower().replace(" ", "_")}.csv', index=False)
        print(f"✅ SHAP importance CSV saved")
        
    except Exception as e:
        print(f"❌ SHAP analysis failed: {e}")

def main():
    """Main evaluation function"""
    print("🚀 Starting Tuned Model Evaluation")
    print("="*50)
    
    # Load models and data
    tuned_model, previous_model, X_test, y_test = load_models_and_data()
    
    if tuned_model is None:
        return
    
    # Evaluate tuned model
    tuned_results = evaluate_model(tuned_model, X_test, y_test, "Tuned LightGBM")
    
    # Evaluate previous model if available
    previous_results = None
    if previous_model:
        previous_results = evaluate_model(previous_model, X_test, y_test, "Previous LightGBM")
    
    # Compare models
    comparison_df = compare_models(tuned_results, previous_results)
    
    # Create comparison plots
    plot_comparison(tuned_results, previous_results)
    
    # Run SHAP analysis
    run_shap_analysis(tuned_model, X_test, "Tuned LightGBM")
    
    # Save comparison results
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print("\n✅ Model comparison results saved as model_comparison_results.csv")
    
    print("\n🎉 Tuned model evaluation complete!")

if __name__ == "__main__":
    main() 