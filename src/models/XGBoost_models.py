import matplotlib.pyplot as plt
import xgboost as xgb


class XGBoost_model:
    def __init__(self, scale_factor, **kwargs):
        self.scale_factor = scale_factor

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.make_model()

    def make_model(self):
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.lr,
            nthread=-1,
            random_state=1,
            max_depth=self.max_depth,
            alpha=self.alpha,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            scale_pos_weight=self.scale_factor,
        )

    def save(self, model_filepath):
        self.model.save_model(model_filepath)

    def plot_results(self):
        results = self.model.evals_result()

        plt.figure(figsize=(10, 7))
        plt.plot(results["validation_0"]["logloss"])
        plt.legend()
        plt.show()

    def load(self, model_filepath):
        bdt = xgb.Booster({"nthread": 4})  # init model
        bdt.load_model(model_filepath)  # load data

    def train(self, input_data, validation_data, y_train, y_test, class_weights):
        history = self.model.fit(
            input_data,
            y_train,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_set=[(validation_data, y_test)],
            eval_metric=["logloss", "auc"],
            verbose=True,
        )

        return history
