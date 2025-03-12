import sys

import numpy as np
import pandas as pd
import os
import math
import warnings
import logging
import joblib  # for saving models
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from skopt import BayesSearchCV  # Make sure to install via: pip install scikit-optimize

warnings.filterwarnings("ignore", category=UserWarning)  # quiet some XGBoost warnings

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("report.txt", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()
# -------------------------------
def kfold_train_and_evaluate_xgb(
        X,
        y,
        n_splits=5,
        hyper_tuning="default",
        param_grid=None,
        param_dist=None,
        search_spaces=None,
        save_dir=None
):
    """
    Performs K-fold cross-validation with XGBoost for classification, printing:
      - Fold-level best hyperparameters (if using a CV search)
      - Fold-level ROC AUC
      - Average ROC AUC across all folds
    Additionally, if save_dir is provided, saves the best model from each fold to that directory.

    Returns:
      best_params_per_fold : list
          Best hyperparameters from each fold (or None if not applicable).
      auc_avg : float
          Average ROC AUC across folds.
      fold_models : list
          List of best models from each fold.
    """

    def build_xgb_model():
        if hyper_tuning == "default":
            return XGBClassifier(random_state=42, eval_metric='logloss', objective='binary:logistic')
        elif hyper_tuning == "grid":
            if not param_grid:
                raise ValueError("param_grid must be provided for grid search.")
            base_xgb = XGBClassifier(random_state=42, eval_metric='logloss', objective='binary:logistic')
            return GridSearchCV(estimator=base_xgb, param_grid=param_grid,
                                scoring='roc_auc', cv=5, verbose=0)
        elif hyper_tuning == "random":
            if not param_dist:
                raise ValueError("param_dist must be provided for random search.")
            base_xgb = XGBClassifier(random_state=42, eval_metric='logloss', objective='binary:logistic')
            return RandomizedSearchCV(estimator=base_xgb, param_distributions=param_dist,
                                      n_iter=10, cv=5, scoring='roc_auc',
                                      random_state=42, verbose=0)
        elif hyper_tuning == "bayesian":
            if not search_spaces:
                raise ValueError("search_spaces must be provided for BayesSearchCV.")
            base_xgb = XGBClassifier(random_state=42, eval_metric='logloss', objective='binary:logistic')
            return BayesSearchCV(estimator=base_xgb, search_spaces=search_spaces,
                                 n_iter=16, cv=5, scoring='roc_auc',
                                 random_state=42, verbose=0)
        else:
            logger.info(f"[XGB] Unrecognized hyper_tuning='{hyper_tuning}'. Using default model.")
            return XGBClassifier(random_state=42, eval_metric='logloss', objective='binary:logistic')

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []
    best_params_per_fold = []
    fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = build_xgb_model()
        model.fit(X_train_fold, y_train_fold)  # For CV objects, this performs internal CV

        if hasattr(model, 'best_estimator_'):
            best_model = model.best_estimator_
            bp = model.best_params_
            if hasattr(bp, "items"):
                bp = dict(bp)
            logger.info("  Best Params: " + str(bp))
            best_params_per_fold.append(bp)
        else:
            best_model = model
            best_params_per_fold.append(None)

        fold_models.append(best_model)
        # Save the model for this fold if a save directory is provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            model_filename = os.path.join(save_dir, f"{hyper_tuning}_{fold_idx + 1}.pkl")
            joblib.dump(best_model, model_filename)
            logger.info(f"  Model saved to {model_filename}")

        y_pred = best_model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred)
        fold_aucs.append(auc)
        logger.info(f"  Fold ROC AUC: {auc:.3f}")

    auc_avg = np.mean(fold_aucs)
    logger.info("\n=== Overall K-Fold Results ===")
    logger.info(f"Average ROC AUC: {auc_avg:.3f}")

    return best_params_per_fold, auc_avg, fold_models


# -------------------------------
def get_xgb_model(best_params=None):
    """
    Returns an XGBClassifier with preset parameters.
    If best_params (a dict) is provided, updates the model parameters accordingly.
    """
    params = {"random_state": 42, "eval_metric": "logloss", "objective": "binary:logistic"}
    if best_params:
        params.update(best_params)
    return XGBClassifier(**params)


# -------------------------------
def main():
    # File paths – adjust as necessary
    train_file = 'training_data.csv'
    test_file = 'testing_data.csv'
    sample_submission_file = 'sample_answers.csv'

    # Load datasets
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    df_sample_sub = pd.read_csv(sample_submission_file)

    # Identify target column (assumes training data has one extra column)
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    possible_targets = train_cols - test_cols - {"Id"}
    if len(possible_targets) != 1:
        raise ValueError("Unable to uniquely identify the target column.")
    target_col = possible_targets.pop()
    logger.info("Identified target column:", target_col)

    y = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])

    # Basic preprocessing: fill missing values & one-hot encode
    X_train = X_train.fillna(X_train.median())
    df_test = df_test.fillna(df_test.median())
    combined = pd.concat([X_train, df_test], axis=0)
    combined = pd.get_dummies(combined)
    X_train = combined.iloc[:len(X_train), :].reset_index(drop=True)
    X_final_test = combined.iloc[len(X_train):, :].reset_index(drop=True)

    # Convert to NumPy arrays for CV
    X_train_np = X_train.values
    y_np = y.values

    # Define parameter sets for each hyper_tuning technique
    techniques = ["default", "grid", "random", "bayesian"]
    grid_example = {
        'n_estimators': [200, 300, 400],  # 200 (100) chosen a lot
        'max_depth': [2, 3],  # [1, 2, 3, 5] 2 and 3 chosen a lot
        'gamma': [0, 0.1, 0.5],  # determines best regularization parameter
        # 'reg_alpha': [0, 0.1, 1.0],  # determines best l1 parameter
        # 'reg_lambda': [1, 1.5, 2.0],  # determines best l2 parameter
        # 'subsample': [0.8, 1.0]  # optional
    }
    random_example = {
        'n_estimators': [175, 200, 225], # 200 chosen a lot
        'max_depth': [3, 4, 5],  # 5 (3) chosen a lot
        'learning_rate': [0.05, 0.075, 0.1],  # 0.05 (0.1) chosen a lot
        'gamma': [0.75, 0.1, 0.125],  # 0.1 chosen a lot
        'reg_alpha': [0, 0.05, 0.1],  # 0 chosen a lot
        'reg_lambda': [2.0, 2.5, 3.0],  # 2.0 chosen a lot
        'subsample': [0.7, 0.8, 0.9]  # 0.8 chosen a lot
    }
    bayesian_example = {
        'n_estimators': (350, 500),  # 356, 400
        'max_depth': (1, 4),  # 4, 3
        'learning_rate': (1e-3, 1e-1, 'log-uniform'),  # 0.0355 (0.055, 0.1)
        'gamma': (0, 0.5),  # 0.308 (0)
        'reg_alpha': (0, 1.0),  # 0.5926 (0.67, 0)
        'reg_lambda': (1, 2.5),  # 1.59, 2, 1.23
        'subsample': (0.6, 1.0, 'uniform')  # 0.859 (1.0)
    }
    params_dict = {
        "default": None,
        "grid": grid_example,
        "random": random_example,
        "bayesian": bayesian_example
    }

    # Loop over each hypertuning technique
    for tech in techniques:
        logger.info("\n==============================================")
        logger.info(f"Processing hypertuning technique: {tech.upper()}")
        logger.info("==============================================")
        # Set up directory for saving outer fold models
        save_dir = f"{tech}_outer_folds"
        # Stage i) K-fold CV training
        if tech == "default":
            bp_folds, kfold_auc, fold_models = kfold_train_and_evaluate_xgb(
                X_train_np, y_np, n_splits=5, hyper_tuning=tech, save_dir=save_dir
            )
        elif tech == "grid":
            bp_folds, kfold_auc, fold_models = kfold_train_and_evaluate_xgb(
                X_train_np, y_np, n_splits=5, hyper_tuning=tech, param_grid=params_dict[tech], save_dir=save_dir
            )
        elif tech == "random":
            bp_folds, kfold_auc, fold_models = kfold_train_and_evaluate_xgb(
                X_train_np, y_np, n_splits=5, hyper_tuning=tech, param_dist=params_dict[tech], save_dir=save_dir
            )
        elif tech == "bayesian":
            bp_folds, kfold_auc, fold_models = kfold_train_and_evaluate_xgb(
                X_train_np, y_np, n_splits=5, hyper_tuning=tech, search_spaces=params_dict[tech], save_dir=save_dir
            )
        logger.info(f"{tech.upper()} K-Fold ROC AUC: {kfold_auc:.3f}")

        # Choose best parameters from CV – here we simply pick the first fold's best params if available
        best_params = bp_folds[0] if bp_folds[0] is not None else None
        logger.info(f"{tech.upper()} Selected Best Hyperparameters: {best_params}")

        # Stage ii) Hold-out test evaluation
        X_tr, X_hold, y_tr, y_hold = train_test_split(X_train, y, test_size=0.2, random_state=42)
        model_hold = get_xgb_model(best_params)
        model_hold.fit(X_tr, y_tr)
        y_hold_pred = model_hold.predict_proba(X_hold)[:, 1]
        hold_auc = roc_auc_score(y_hold, y_hold_pred)
        logger.info(f"{tech.upper()} Hold-Out Test ROC AUC: {hold_auc:.3f}")

        # Stage iii) Final training on all training data and submission prediction
        final_model = get_xgb_model(best_params)
        final_model.fit(X_train, y)  # train on full training data
        y_final_pred = final_model.predict_proba(X_final_test)[:, 1]

        # Prepare aggregated submission file (from final model)
        if 'target' in df_sample_sub.columns:
            df_sample_sub['target'] = y_final_pred
        elif 'Y' in df_sample_sub.columns:
            df_sample_sub['Y'] = y_final_pred
        else:
            pred_col = df_sample_sub.columns[1]
            df_sample_sub[pred_col] = y_final_pred

        submission_filename = f"5fold_submission_{tech}.csv"
        df_sample_sub.to_csv(submission_filename, index=False)
        logger.info(f"{tech.upper()} Aggregated Submission saved to {submission_filename}")

        # Stage iv) Generate individual submission CSVs from each fold's model
        for i, model in enumerate(fold_models):
            # Retrain the saved model on full training data
            model.fit(X_train, y)
            y_fold_pred = model.predict_proba(X_final_test)[:, 1]
            # Reload sample submission for a fresh copy
            df_sub = pd.read_csv(sample_submission_file)
            if 'target' in df_sub.columns:
                df_sub['target'] = y_fold_pred
            elif 'Y' in df_sub.columns:
                df_sub['Y'] = y_fold_pred
            else:
                pred_col = df_sub.columns[1]
                df_sub[pred_col] = y_fold_pred
            fold_submission_filename = os.path.join(save_dir, f"{tech}_{i + 1}.csv")
            df_sub.to_csv(fold_submission_filename, index=False)
            logger.info(f"{tech.upper()} Fold {i + 1} Submission saved to {fold_submission_filename}")


if __name__ == "__main__":
    main()
