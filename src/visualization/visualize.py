"""
Make a series of plots for model evaluation and save them to a folder
in reports/model_name.

Usage: visualize.py --model_name {model_name} --make_shap

where:
--model_name: the name of the model to be evaluated, will look for it in the models folder
--make_shap: flag to generate shap plots for the model, this takes longer than the other plots
"""

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import xgboost as xgb
import yaml
from sklearn.metrics import (
    PrecisionRecallDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from src.features.build_features import (
    load_preprocessed_data,
    scale_event_data,
    scale_object_data,
)
from tensorflow import keras

config = yaml.safe_load(open(os.path.join(Path(__file__).parent.parent, "config.yaml")))


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
    X_test = data["X_test"]
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

    with plt.style.context(config["visuals"]["style"]):
        plt.plot(thresholds, significance)
        plt.ylabel("Significance")
        plt.xlabel("Threshold")

    return (thresholds, significance)


def make_discriminator(data, preds, model_name, best_threshold):
    labels = data["y_test"].values
    lum = config["data"]["lum"]
    weights = labels * lum

    if "xgboost" in model_name:
        sg = [
            pred * weights[i]
            for i, (label, pred) in enumerate(zip(labels, preds))
            if label == 1
        ]
        bg = [
            pred * weights[i]
            for i, (label, pred) in enumerate(zip(labels, preds))
            if label == 0
        ]
    else:
        sg = [pred[0] for label, pred in zip(labels, preds) if label == 1]
        bg = [pred[0] for label, pred in zip(labels, preds) if label == 0]

    n_bins = 200
    use_log = False
    histtype = 'stepfilled'
    alpha = 0.6
    linewidth = 2

    bg_color = 'tab:blue'
    bg_edge_color = 'blue'
    sg_color = 'tab:red'
    sg_edge_color = 'darkred'

    with plt.style.context(config["visuals"]["style"]):
        _, ax = plt.subplots(1, figsize=(10, 5))

        ax.hist(
            bg,
            density=False,
            bins=n_bins,
            range=(0, 1),
            label=r"$t\bar{t}$",
            histtype=histtype,
            alpha=alpha,
            color=bg_color,
            edgecolor=bg_edge_color,
            linewidth=linewidth,
            log=use_log,
        )
        ax.hist(
            sg,
            density=False,
            bins=n_bins,
            range=(0, 1),
            label="ttH",
            histtype=histtype,
            alpha=alpha,
            color=sg_color,
            edgecolor=sg_edge_color,
            linewidth=linewidth,
            log=use_log,
        )
        ax.axvline(best_threshold, color='black', linestyle='--')
    
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
        ax.set_xlabel("Signal Threshold")
        ax.set_ylabel("Event Density")

        plt.legend(loc=1)
        plt.tight_layout()


def make_confusion_matrix(labels, predictions, p=0.5, norm="pred"):
    cm = confusion_matrix(labels, predictions > p, normalize=norm)
    linewidth = 0

    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = linewidth

    ax = sns.heatmap(
        cm,
        cmap='OrRd',
        annot=True,
        linewidth=linewidth,
        square=True,
        linecolor='black',
        xticklabels=[r"$t\bar{t}$ (background)", "ttH (signal)"],
        yticklabels=[r"$t\bar{t}$ (background)", "ttH (signal)"],
        vmin=0,
        vmax=1,
    )

    plt.title(f"Discriminator Threshold = {round(p, 3)}", loc='center', fontsize=10)

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    plt.rcParams.update(plt.rcParamsDefault)


def make_roc_curve(data, preds):
    fpr, tpr, _ = roc_curve(data["y_test"], preds)

    with plt.style.context(config["visuals"]["style"]):
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "k--")
        plt.axis([0, 1, 0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")


def make_pr_curve(data, preds):
    precision, recall, _ = precision_recall_curve(data["y_test"], preds)
    
    with plt.style.context(config["visuals"]["style"]):
        plt.figure(figsize=(8, 8))
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.xlabel("Recall")
        plt.ylabel("Precision")


def calculate_metrics(data, preds, plot_path):
    thresholds, significance = make_significance(data, preds)
    index = significance.argmax()
    best_threshold = thresholds[index]
    best_significance = significance[index]
    auc_score = roc_auc_score(data["y_test"], preds)
    plt.clf()
    
    metrics = {
        "Best Threshold": best_threshold,
        "Best Significance": best_significance,
        "AUC": auc_score,
    }

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
    with plt.style.context(config["visuals"]["style"]):
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
    plt.clf()


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
    data["event_X_train"], data["event_X_test"] = scale_event_data(
        data["event_X_train"], data["event_X_test"]
    )
    data["object_X_train"], data["object_X_test"] = scale_object_data(
        data["object_X_train"], data["object_X_test"]
    )

    model_name = args.model_name
    model_path = os.path.join("models", model_name)

    if "xgboost" in model_name:
        model = xgb.Booster({"nthread": 4})  # init model
        model.load_model(model_path)  # load data
        preds = model.predict(xgb.DMatrix(data["event_X_test"].values))
    elif "FNN" in model_name:
        model = keras.models.load_model(model_path, compile=False)
        preds = model.predict(data["event_X_test"])
    elif "RNN" in model_name:
        model = keras.models.load_model(model_path, compile=False)
        preds = model.predict(data["object_X_test"])
    else:
        model = keras.models.load_model(model_path, compile=False)
        preds = model.predict([data["event_X_test"], data["object_X_test"]])

    fig_path = config["paths"]["fig_path"]
    plot_path = os.path.join(fig_path, model_name)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    make_significance(data, preds)
    save_plot(model_name, "significance")
    best_threshold = calculate_metrics(data, preds, plot_path)
    make_confusion_matrix(data["y_test"], preds, p=best_threshold, norm="pred")
    save_plot(model_name, "confusion_matrix_pred")
    make_confusion_matrix(data["y_test"], preds, p=best_threshold, norm="true")
    save_plot(model_name, "confusion_matrix_true")
    make_discriminator(data, preds, model_name, best_threshold)
    save_plot(model_name, "discriminator_plot")
    make_roc_curve(data, preds)
    save_plot(model_name, "roc_curve")
    make_pr_curve(data, preds)
    save_plot(model_name, "pr_curve")

    if args.make_shap:
        make_shap_plots(model_name, model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural net")
    parser.add_argument(
        "--model_name",
        default="model_test.h5",
        help="Name of model (ends in .h5 for keras models or .model for xgboost)",
    )
    parser.add_argument(
        "--make_shap",
        action="store_true",
        default=False,
        help="Name of model (ends in .h5 for keras models or .model for xgboost)",
    )
    args = parser.parse_args()
    main(args)
