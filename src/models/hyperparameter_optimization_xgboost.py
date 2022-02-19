import os
import xgboost as xgb
import pandas as pd
from src.features.build_features import preprocess_data
import optuna
from optuna.integration import XGBoostPruningCallback
import logging
import sys

# ignores all warnings
import warnings
warnings.filterwarnings("ignore")


############## LOADING PREPROCESSED DATA ##############

load_path = r"data/processed"

use_all_data = input("Use all data (y/n): ")
if use_all_data != 'y': preprocess_data()

event_X_train = pd.read_pickle(os.path.join(load_path, "event_X_train.pkl"))
event_X_test = pd.read_pickle(os.path.join(load_path, "event_X_test.pkl"))

y_train = pd.read_pickle(os.path.join(load_path, "y_train.pkl"))
y_test = pd.read_pickle(os.path.join(load_path, "y_test.pkl"))

############## DEFINING HYPERPARAMETERS ##############

activation='relu'
n_iter_no_change = 10
n_estimators = 1000
scale_factor = (y_train == 0).sum() / (y_train == 1).sum()

############## CREATING AND TRAINING MODEL ##############


def objective(trial):
    params = {
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'lr': trial.suggest_loguniform('lr', 1e-2, 1),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
    }
    model = create_model(params)

    model.fit(event_X_train, y_train,
                early_stopping_rounds=n_iter_no_change,
                eval_set=[(event_X_test, y_test)],
                eval_metric='logloss',
                callbacks=[XGBoostPruningCallback(trial, 'validation_0-logloss')],
                verbose=True)

    score = model.best_score

    return -score


def create_model(params):
    model = xgb.XGBClassifier(n_estimators=n_estimators,
            nthread=-1,
            random_state=1,
            max_depth=params['max_depth'],
            learning_rate=params['lr'], 
            alpha=params['alpha'],
            reg_lambda=params['lambda'],
            min_child_weight=params['min_child_weight'],
            colsample_bytree=params['colsample_bytree'],
            subsample=params['subsample'],
            scale_pos_weight=scale_factor)

    return model


def main():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "bayesian_opt_xgboost_v1"  # Unique identifier of the study.
    storage_name = f"sqlite:///models/{study_name}.db"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
        study_name=study_name,
        storage=storage_name,
        load_if_exists=False,
    )

    study.optimize(objective, n_trials=200)


if __name__ == "__main__":
    main()