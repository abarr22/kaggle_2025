import sys
import numpy as np
import pandas as pd
import os
import math
import warnings
import logging
import joblib  # for saving models
import xgboost
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from skopt import BayesSearchCV  # Make sure to install via: pip install scikit-optimize
from sklearn.feature_selection import SelectFromModel

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
def preprocess_features(df_train, df_test):
    """
    Preprocesses training and testing features by:
      - Replacing sentinel missing values (-3, -1) with np.nan.
      - Filling missing values (median for numeric, mode for categorical).
      - Capping outliers at the 1st and 99th percentiles.
      - Adding interaction features from 'f2' and 'f3': difference and product.
      - Applying a log1p transform on highly skewed numeric features.
      - One-hot encoding the combined dataset.
    Returns:
      df_train_new, df_test_new: Preprocessed training and testing DataFrames.
    """
    # Replace sentinel missing values
    df_train = df_train.replace({-3: np.nan, -1: np.nan})
    df_test = df_test.replace({-3: np.nan, -1: np.nan})

    # Fill missing values for numeric columns using median and for non-numeric using mode
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df_train[col].median()
        df_train.loc[:, col] = df_train[col].fillna(median_val)
        if col in df_test.columns:
            df_test.loc[:, col] = df_test[col].fillna(df_test[col].median())

    categorical_cols = df_train.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        mode_val = df_train[col].mode()[0]
        df_train.loc[:, col] = df_train[col].fillna(mode_val)
        if col in df_test.columns:
            df_test.loc[:, col] = df_test[col].fillna(mode_val)

    # Cap outliers in numeric columns at the 1st and 99th percentiles
    for col in numeric_cols:
        df_train[col] = df_train[col].astype(float)
        df_test[col] = df_test[col].astype(float)
        lower = df_train[col].quantile(0.01)
        upper = df_train[col].quantile(0.99)
        df_train.loc[:, col] = df_train[col].clip(lower, upper)
        if col in df_test.columns:
            lower_test = df_test[col].quantile(0.01)
            upper_test = df_test[col].quantile(0.99)
            df_test.loc[:, col] = df_test[col].clip(lower_test, upper_test)

    # Add interaction features if f2 and f3 exist
    if 'f2' in df_train.columns and 'f3' in df_train.columns:
        df_train['f2_minus_f3'] = df_train['f2'] - df_train['f3']
        df_train['f2_times_f3'] = df_train['f2'] * df_train['f3']
        df_test['f2_minus_f3'] = df_test['f2'] - df_test['f3']
        df_test['f2_times_f3'] = df_test['f2'] * df_test['f3']

    # Apply log1p transform on numeric features that are highly skewed
    for col in numeric_cols:
        if df_train[col].min() >= 0:
            median_val = df_train[col].median()
            # If maximum is much larger than the median, assume heavy skew
            if median_val > 0 and (df_train[col].max() / median_val) > 10:
                df_train.loc[:, col] = np.log1p(df_train[col])
                if col in df_test.columns:
                    df_test.loc[:, col] = np.log1p(df_test[col])

    # Combine train and test for one-hot encoding
    combined = pd.concat([df_train, df_test], axis=0)
    combined = pd.get_dummies(combined)
    df_train_new = combined.iloc[:len(df_train), :].reset_index(drop=True)
    df_test_new = combined.iloc[len(df_train):, :].reset_index(drop=True)

    return df_train_new, df_test_new


# -------------------------------
def select_features(X, y):
    """
    Performs model-based feature selection using SelectFromModel with an XGBClassifier.
    Features with importance below the median importance are dropped.
    Returns the transformed feature matrix and the selector object.
    """
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', objective='binary:logistic')
    xgb_model.fit(X, y)
    selector = SelectFromModel(xgb_model, threshold="median", prefit=True)
    X_selected = selector.transform(X)
    return X_selected, selector


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
      fold_aucs : list
          List of ROC AUC values for each fold.
    """

    def build_xgb_model():
        if hyper_tuning == "default":
            # Increase n_estimators to allow early stopping
            return XGBClassifier(random_state=42, eval_metric='logloss', objective='binary:logistic', n_estimators=1000)
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

    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []
    best_params_per_fold = []
    fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        logger.info(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        if hyper_tuning == "default":
            # Use early stopping with a validation set for the default model
            model = build_xgb_model()
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
        else:
            model = build_xgb_model()
            model.fit(X_train_fold, y_train_fold)

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

    return best_params_per_fold, auc_avg, fold_models, fold_aucs


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
    logger.info("Identified target column: " + target_col)

    y = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])

    # Preprocess features with enhancements
    X_train, df_test_processed = preprocess_features(X_train, df_test)
    X_final_test = df_test_processed.copy()

    # Feature selection: remove low-importance features using SelectFromModel
    X_train, selector = select_features(X_train, y)
    # Transform test set with the same selector
    X_final_test = selector.transform(X_final_test)

    # Convert to NumPy arrays for CV
    X_train_np = X_train  # already a numpy array after selector.transform()
    y_np = y.values

    # Define parameter sets for each hypertuning technique
    techniques = ["default", "random", "bayesian"]

    # Parameter ranges for RandomizedSearchCV:
    random_example = {
        'n_estimators': [400, 450, 500, 550, 600],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'gamma': [0.05, 0.075, 0.1, 0.125, 0.15],
        'subsample': [0.85, 0.9, 0.95, 1.0],
        'colsample_bytree': [0.9, 0.95, 1.0],
        'reg_alpha': [0.2, 0.3, 0.4, 0.5],
        'reg_lambda': [2.0, 2.5, 3.0, 3.5]
    }

    # Parameter ranges for BayesSearchCV:
    bayesian_example = {
        'n_estimators': (400, 600),
        'max_depth': (4, 6),
        'learning_rate': (0.01, 0.1, 'log-uniform'),
        'gamma': (0.05, 0.15),
        'subsample': (0.85, 1.0, 'uniform'),
        'colsample_bytree': (0.9, 1.0, 'uniform'),
        'reg_alpha': (0.2, 0.5),
        'reg_lambda': (2.0, 3.0)
    }

    params_dict = {
        "default": None,
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
            bp_folds, kfold_auc, fold_models, fold_aucs = kfold_train_and_evaluate_xgb(
                X_train_np, y_np, n_splits=5, hyper_tuning=tech, save_dir=save_dir
            )
        elif tech == "random":
            bp_folds, kfold_auc, fold_models, fold_aucs = kfold_train_and_evaluate_xgb(
                X_train_np, y_np, n_splits=5, hyper_tuning=tech, param_dist=params_dict[tech], save_dir=save_dir
            )
        elif tech == "bayesian":
            bp_folds, kfold_auc, fold_models, fold_aucs = kfold_train_and_evaluate_xgb(
                X_train_np, y_np, n_splits=5, hyper_tuning=tech, search_spaces=params_dict[tech], save_dir=save_dir
            )
        logger.info(f"{tech.upper()} K-Fold ROC AUC: {kfold_auc:.3f}")

        # Choose best parameters from CV – here we simply pick the first fold's best params if available
        best_params = bp_folds[0] if bp_folds[0] is not None else None
        logger.info(f"{tech.upper()} Selected Best Hyperparameters: {best_params}")

        # Stage ii) Hold-out test evaluation
        X_tr, X_hold, y_tr, y_hold = train_test_split(X_train_np, y_np, test_size=0.2, random_state=42)
        model_hold = get_xgb_model(best_params)
        model_hold.fit(X_tr, y_tr)
        y_hold_pred = model_hold.predict_proba(X_hold)[:, 1]
        hold_auc = roc_auc_score(y_hold, y_hold_pred)
        logger.info(f"{tech.upper()} Hold-Out Test ROC AUC: {hold_auc:.3f}")

        # Stage iii) Final training on all training data using weighted ensemble of fold models
        meta_X_hold = np.column_stack([model.predict_proba(X_hold)[:, 1] for model in fold_models])
        from sklearn.linear_model import LogisticRegression
        meta_learner = LogisticRegression(solver='liblinear', random_state=42)
        meta_learner.fit(meta_X_hold, y_hold)

        meta_X_final = np.column_stack([model.predict_proba(X_final_test)[:, 1] for model in fold_models])
        y_final_pred_meta = meta_learner.predict_proba(meta_X_final)[:, 1]

        y_final_pred = y_final_pred_meta  # final prediction using weighted ensemble

        # Prepare aggregated submission file (from weighted ensemble)
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
            model.fit(X_train_np, y_np)
            y_fold_pred = model.predict_proba(X_final_test)[:, 1]
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
