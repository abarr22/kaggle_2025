import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import itertools

# -------------------------------
def add_pairwise_interactions(df, columns=None, max_columns=10):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(columns) > max_columns:
        columns = columns[:max_columns]
    for col1, col2 in itertools.combinations(columns, 2):
        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    return df

# -------------------------------
def main():
    train_file = 'training_data.csv'
    test_file = 'testing_data.csv'
    sample_submission_file = 'sample_answers.csv'

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    df_sample_sub = pd.read_csv(sample_submission_file)

    # Identify target column.
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    possible_targets = train_cols - test_cols - {"Id"}
    if len(possible_targets) != 1:
        raise ValueError("Unable to uniquely identify the target column.")
    target_col = possible_targets.pop()
    print("Identified target column:", target_col)

    y = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])

    # Preprocessing
    X_train = X_train.fillna(X_train.median())
    df_test = df_test.fillna(df_test.median())

    # One-Hot Encoding
    combined = pd.concat([X_train, df_test], axis=0)
    combined = pd.get_dummies(combined)
    X_train = combined.iloc[:len(X_train), :].reset_index(drop=True)
    X_test = combined.iloc[len(X_train):, :].reset_index(drop=True)

    # Feature Engineering
    X_train = add_pairwise_interactions(X_train.copy(), max_columns=10)
    X_test = add_pairwise_interactions(X_test.copy(), max_columns=10)

    # Ensemble: train three models with different seeds.
    seeds = [42, 52, 62]
    preds = np.zeros((X_test.shape[0], len(seeds)))
    aucs = []
    for i, seed in enumerate(seeds):
        print(f"Training model with seed {seed}")
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              n_estimators=100, max_depth=3, learning_rate=0.1,
                              subsample=1.0, random_state=seed)
        # Split to get a local validation score.
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=seed, stratify=y)
        model.fit(X_tr, y_tr)
        y_val_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred)
        aucs.append(auc)
        print(f"Validation ROC AUC for seed {seed}: {auc:.4f}")
        preds[:, i] = model.predict_proba(X_test)[:, 1]

    # Average predictions from the ensemble.
    final_preds = preds.mean(axis=1)
    print("Average Validation ROC AUC across seeds:", np.mean(aucs))

    if 'target' in df_sample_sub.columns:
        df_sample_sub['target'] = final_preds
    elif 'Y' in df_sample_sub.columns:
        df_sample_sub['Y'] = final_preds
    else:
        pred_col = df_sample_sub.columns[1]
        df_sample_sub[pred_col] = final_preds

    submission_file = 'submission_ensemble.csv'
    df_sample_sub.to_csv(submission_file, index=False)
    print(f"Submission saved to {submission_file}")

if __name__ == '__main__':
    main()
