import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
import yaml
from sklearn.metrics import (
    PrecisionRecallDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from src.features.build_features import load_preprocessed_data
from tensorflow import keras

config = yaml.safe_load(open("src/config.yaml"))
plt.style.use(config["visuals"]["style"])

# maybe write code to automatically create a latex document or something based on these plots


def make_training_curves(history):
    """Generates training curves for a model

    Args:
        history: model training history
    """
    metrics = ["loss", "accuracy", "f1_score", "AUC"]
    _ = plt.figure(figsize=(20, 10))

    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ")
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label="Train")
        plt.plot(
            history.epoch, history.history["val_" + metric], linestyle="--", label="Val"
        )

        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.legend()


def make_significance(data, preds):
    interim_path = config["paths"]["interim_path"]
    X_test = pd.read_pickle(os.path.join(interim_path, "X_test.pkl"))
    test_weight = X_test["xs_weight"].values
    test_frac = len(data["y_test"]) / len(data["y_train"])

    n_thresholds = 50
    thresholds = np.linspace(0, 1, n_thresholds)
    significance = np.zeros(len(thresholds), dtype=float)

    lum = config["data"]["lum"]
    epsilon = 1e-5

    sg = np.zeros(len(thresholds))
    bg = np.zeros(len(thresholds))
    labels = data["y_test"].values

    for i, threshold in enumerate(thresholds):
        sg[i] = (
            sum(
                [
                    test_weight[j]
                    for j, (pred, label) in enumerate(zip(preds, labels))
                    if (pred >= threshold and label == 1)
                ]
            )
            * lum
            / test_frac
        )
        bg[i] = (
            sum(
                [
                    test_weight[j]
                    for j, (pred, label) in enumerate(zip(preds, labels))
                    if (pred >= threshold and label == 0)
                ]
            )
            * lum
            / test_frac
        )

    significance = sg / np.sqrt(bg + epsilon)

    plt.plot(thresholds, significance)
    plt.title("Significance as a function of Threshold")
    plt.ylabel("Significance")
    plt.xlabel("Threshold")

    return (thresholds, significance)


def make_discriminator(data, preds, model_name):
    labels = data["y_test"].values

    if "xgboost" in model_name:
        sg = [pred for label, pred in zip(labels, preds) if label == 1]
        bg = [pred for label, pred in zip(labels, preds) if label == 0]
    else:
        sg = [pred[0] for label, pred in zip(labels, preds) if label == 1]
        bg = [pred[0] for label, pred in zip(labels, preds) if label == 0]

    n_bins = 75
    alpha = 0.6

    fig, axs = plt.subplots(2, figsize=(14, 8))

    axs[0].yaxis.set_ticks([])
    axs[1].yaxis.set_ticks([])

    fig.suptitle("Discriminator Plots")
    axs[0].hist(
        sg, density=True, bins=n_bins, range=(0, 1), alpha=alpha, label="Signal"
    )
    axs[0].hist(
        bg, density=True, bins=n_bins, range=(0, 1), alpha=alpha, label="Background"
    )
    axs[1].hist(
        sg, density=False, bins=n_bins, range=(0, 1), alpha=alpha, label="Signal"
    )
    axs[1].hist(
        bg, density=False, bins=n_bins, range=(0, 1), alpha=alpha, label="Background"
    )

    axs[0].set_title("Normalised")
    axs[1].set_title("Unnormalised")
    plt.legend()


def make_confusion_matrix(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p, normalize="pred")
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=["background", "signal"],
        yticklabels=["background", "signal"],
        vmin=0,
        vmax=1,
    )

    plt.title(f"Normalised Confusion Matrix at {round(p, 3)}")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")


def make_roc_curve(data, preds):
    fpr, tpr, _ = roc_curve(data["y_test"], preds)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr)
    plt.axis([0, 1, 0, 1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


def make_pr_curve(data, preds):
    precision, recall, _ = precision_recall_curve(data["y_test"], preds)

    plt.figure(figsize=(8, 8))
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")


def calculate_metrics(data, preds, plot_path):
    thresholds, significance = make_significance(data, preds)
    index = significance.argmax()
    best_threshold = thresholds[index]
    best_significance = significance[index]

    auc_score = roc_auc_score(data["y_test"], preds)
    # accuracy = accuracy_score(y_test, preds)
    # f1 = f1_score(y_test, preds)

    metrics = {}
    metrics["Best Threshold"] = best_threshold
    metrics["Best Significance"] = best_significance
    metrics["AUC"] = auc_score
    # metrics["Accuracy"] = accuracy
    # metrics["F1 Score"] = f1

    with open(os.path.join(plot_path, "metrics.pickle"), "wb") as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return best_threshold


def make_summary_plot(shap_values, n_values, data):
    shap.summary_plot(
        shap_values,
        features=data["event_X_test"].head(n_values),
        feature_names=data["event_X_test"].columns,
        plot_size=(15, 10),
        show=False,
    )


def make_summary_plot_abs(shap_values, n_values, data):
    shap.summary_plot(
        np.abs(shap_values),
        features=data["event_X_test"].head(n_values),
        feature_names=data["event_X_test"].columns,
        plot_size=(15, 10),
        show=False,
    )


def make_bar_plots(shap_values, data):
    # max SHAP value bar plot
    _, axs = plt.subplots(ncols=1, nrows=2, figsize=(10, 15))
    cols = data["event_X_test"].columns

    shap_max = np.max(np.abs(shap_values), axis=0)
    shap_mean = np.mean(np.abs(shap_values), axis=0)
    shap_max, cols_max = zip(*sorted(zip(shap_max, cols)))
    shap_mean, cols_mean = zip(*sorted(zip(shap_mean, cols)))

    axs[0].barh(cols_mean, shap_mean)
    axs[1].barh(cols_max, shap_max)

    axs[0].set_xlabel("mean(|SHAP value|)")
    axs[1].set_xlabel("max(|SHAP value|)")


def make_dependence_plots(shap_values, n_values, data):
    plt.figure(figsize=(25, 20))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Dependence Plots", fontsize=40, y=0.95)

    for i, name in enumerate(data["event_X_train"].columns):
        ax = plt.subplot(4, 3, i + 1)
        shap.dependence_plot(
            name, shap_values, data["event_X_test"].head(n_values), ax=ax, show=False
        )


def save_plot(model_name: str, fig_name: str):
    """Saves a matplotlib figure in reports/figures in a
    subfolder named after the model name

    Args:
        model_name (str): name of the model + name of the subfolder
        fig_name (str): name of figure to be saved,
            without any filetype e.g. "bar_plot"
    """
    fig_path = config["paths"]["fig_path"]
    plot_path = os.path.join(fig_path, model_name)

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    plt.savefig(os.path.join(plot_path, f"{fig_name}.png"), bbox_inches="tight")


def make_shap_plots(model_name, model, data):
    n_bg = config["visuals"]["shap_n_bg"]
    n_values = config["visuals"]["shap_n_values"]

    if "xgboost" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data["event_X_test"].head(n_values).values)
    else:
        background = [
            data["event_X_train"].head(n_bg).values,
            data["object_X_train"][:n_bg],
        ]
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(
            [
                data["event_X_test"].head(n_values).values,
                data["object_X_test"][:n_values],
            ]
        )
        shap_values = shap_values[0][0]

    make_summary_plot(shap_values, n_values, data)
    save_plot(model_name, "shap_summary")
    make_summary_plot_abs(shap_values, n_values, data)
    save_plot(model_name, "shap_summary_abs")
    make_bar_plots(shap_values, data)
    save_plot(model_name, "shap_bar_plots")
    make_dependence_plots(shap_values, n_values, data)
    save_plot(model_name, "shap_dependence_plots")


def main(args):
    data = load_preprocessed_data()
    model_name = args.model_name
    model_path = os.path.join("models", model_name)

    if "xgboost" in model_name:
        model = xgb.Booster({"nthread": 4})  # init model
        model.load_model(model_path)  # load data
        preds = model.predict(xgb.DMatrix(data["event_X_test"].values))
    else:
        model = keras.models.load_model(
            model_path, custom_objects={"f1_score": f1_score}
        )
        preds = model.predict([data["event_X_test"], data["object_X_test"]])

    fig_path = config["paths"]["fig_path"]
    plot_path = os.path.join(fig_path, model_name)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    make_significance(data, preds)
    save_plot(model_name, "significance")
    best_threshold = calculate_metrics(data, preds, plot_path)
    make_confusion_matrix(data["y_test"], preds, p=best_threshold)
    save_plot(model_name, "confusion_matrix")
    make_discriminator(data, preds, model_name)
    save_plot(model_name, "discriminator_plots")
    make_roc_curve(data, preds)
    save_plot(model_name, "roc_curve")
    make_pr_curve(data, preds)
    save_plot(model_name, "pr_curve")
    make_shap_plots(model_name, model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural net")
    parser.add_argument(
        "--model_name",
        default="model_test.h5",
        help="Name of model (ends in .h5 for keras models or .model for xgboost)",
    )
    args = parser.parse_args()
    main(args)
