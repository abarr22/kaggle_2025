import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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

    # Load data
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

    # Preprocessing: fill missing values.
    X_train = X_train.fillna(X_train.median())
    df_test = df_test.fillna(df_test.median())

    # One-Hot Encoding (consistent across train/test)
    combined = pd.concat([X_train, df_test], axis=0)
    combined = pd.get_dummies(combined)
    X_train = combined.iloc[:len(X_train), :].reset_index(drop=True)
    X_test = combined.iloc[len(X_train):, :].reset_index(drop=True)

    # Feature Engineering: add pairwise interaction features.
    X_train = add_pairwise_interactions(X_train.copy(), max_columns=10)
    X_test = add_pairwise_interactions(X_test.copy(), max_columns=10)

    # Split a hold-out set for hyperparameter tuning.
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42, stratify=y)

    # Hyperparameter tuning with GridSearchCV (without early stopping)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    xgb_clf = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='roc_auc',
                               cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_tr, y_tr)  # Removed eval_set and early_stopping_rounds here
    print("Best hyperparameters:", grid_search.best_params_)

    # Validate the best model on the hold-out set.
    best_model = grid_search.best_estimator_
    y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    print("Validation ROC AUC:", val_auc)

    # Retrain the best model on the full training set with early stopping.
    X_full_tr, X_full_val, y_full_tr, y_full_val = train_test_split(
        X_train, y, test_size=0.1, random_state=42, stratify=y)
    best_model.set_params(early_stopping_rounds=10)
    best_model.fit(X_full_tr, y_full_tr, eval_set=[(X_full_val, y_full_val)], verbose=False)

    # Predict on test set.
    test_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Prepare submission.
    if 'target' in df_sample_sub.columns:
        df_sample_sub['target'] = test_pred_proba
    elif 'Y' in df_sample_sub.columns:
        df_sample_sub['Y'] = test_pred_proba
    else:
        pred_col = df_sample_sub.columns[1]
        df_sample_sub[pred_col] = test_pred_proba

    submission_file = 'submission_early_stopping.csv'
    df_sample_sub.to_csv(submission_file, index=False)
    print(f"Submission saved to {submission_file}")

if __name__ == '__main__':
    main()
