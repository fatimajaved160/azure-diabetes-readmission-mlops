
import argparse
import os
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to train parquet file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test parquet file")
    parser.add_argument("--model_output", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    mlflow.sklearn.autolog()

    train_df = pd.read_parquet(args.train_data)
    test_df = pd.read_parquet(args.test_data)

    target_col = "readmitted"
    feature_cols = [c for c in train_df.columns if c not in [target_col, "patient_nbr"]]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        print(classification_report(y_test, preds))

        macro_f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("macro_f1", macro_f1)
        print("Macro F1:", macro_f1)

        # Save model to the output directory Azure ML gives us
        os.makedirs(args.model_output, exist_ok=True)
        joblib.dump(model, os.path.join(args.model_output, "model.pkl"))
        print(f"Model saved to {args.model_output}")

if __name__ == "__main__":
    main()
