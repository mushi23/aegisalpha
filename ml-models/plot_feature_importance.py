import argparse
import joblib
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model_path, top_n=10, output_csv=None, print_features=False):
    model = joblib.load(model_path)
    importances = model.feature_importances_
    features = model.feature_names_in_

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.gca().invert_yaxis()
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()

    if output_csv:
        importance_df.to_csv(output_csv, index=False)
        print(f"✅ Top-{top_n} features saved to {output_csv}")
    if print_features:
        print("Top features for training:")
        print(list(importance_df['Feature']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved model (.pkl)')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top features to show')
    parser.add_argument('--output_csv', type=str, default=None, help='CSV file to save top features')
    parser.add_argument('--print_features', action='store_true', help='Print Python list of top features')
    args = parser.parse_args()

    plot_feature_importance(args.model, args.top_n, args.output_csv, args.print_features) 