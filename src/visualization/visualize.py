import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import pandas as pd
import pickle
import sys
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    PrecisionRecallDisplay,
    accuracy_score,
    f1_score,
)


############## LOADING MODEL AND DATA ##############

processed_path = r"data/processed"
interim_path = r"data/interim"
model_name = sys.argv[1]

event_X_train = pd.read_pickle(os.path.join(processed_path, "event_X_train.pkl"))
event_X_test = pd.read_pickle(os.path.join(processed_path, "event_X_test.pkl"))

y_train = pd.read_pickle(os.path.join(processed_path, "y_train.pkl"))
y_test = pd.read_pickle(os.path.join(processed_path, "y_test.pkl"))

object_X_train = np.load(os.path.join(processed_path, "object_X_train.npy"))
object_X_test = np.load(os.path.join(processed_path, "object_X_test.npy"))

X_test = pd.read_pickle(os.path.join(interim_path, "X_test.pkl"))

model_path = os.path.join("models", model_name)
model = keras.models.load_model(model_path, custom_objects={"f1_score": f1_score})
preds = model.predict([event_X_test, object_X_test])

############## GENERATING AND SAVING PLOTS ##############

plot_path = r"reports/figures"
plot_path = os.path.join(plot_path, model_name)
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


def make_significance():
    test_weight = X_test["xs_weight"].values
    test_frac = len(y_test) / len(y_train)

    thresholds = np.linspace(0, 1, 50)
    significance = np.zeros(len(thresholds), dtype=float)

    lum = 140e3
    epsilon = 1e-5

    sg = np.zeros(len(thresholds))
    bg = np.zeros(len(thresholds))
    labels = y_test.values

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
    plt.savefig(os.path.join(plot_path, "significance.png"))
    
    print("GENERATED SIGNIFICANCE PLOT")

    return (thresholds, significance)


def make_discriminator():
    labels = y_test.values

    sg = [pred[0] for label, pred in zip(labels, preds) if label == 1]
    bg = [pred[0] for label, pred in zip(labels, preds) if label == 0]

    n_bins = 75
    alpha = 0.6

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8))

    ax1.yaxis.set_ticks([])
    ax2.yaxis.set_ticks([])

    fig.suptitle("Discriminator Plots")
    ax1.hist(sg, density=True, bins=n_bins, range=(0, 1), alpha=alpha, label="Signal")
    ax1.hist(
        bg, density=True, bins=n_bins, range=(0, 1), alpha=alpha, label="Background"
    )
    ax2.hist(sg, density=False, bins=n_bins, range=(0, 1), alpha=alpha, label="Signal")
    ax2.hist(
        bg, density=False, bins=n_bins, range=(0, 1), alpha=alpha, label="Background"
    )

    ax1.set_title("Normalised")
    ax2.set_title("Unnormalised")
    plt.legend()

    print("GENERATED DISCRIMINATOR PLOT")

    plt.savefig(os.path.join(plot_path, "discriminator_plots.png"))


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

    print("GENERATED CONFUSION MATRIX")

    plt.savefig(os.path.join(plot_path, "confusion_matrix.png"))


def make_roc_curve():
    fpr, tpr, _ = roc_curve(y_test, preds)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr)
    plt.axis([0, 1, 0, 1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    print("GENERATED ROC CURVE")

    plt.savefig(os.path.join(plot_path, "roc_curve.png"))


def make_pr_curve():
    precision, recall, _ = precision_recall_curve(y_test, preds)

    plt.figure(figsize=(8, 8))
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    print("GENERATED PRECISION-RECALL CURVE")

    plt.savefig(os.path.join(plot_path, "pr_curve.png"))


def calculate_metrics():
    thresholds, significance = make_significance()
    index = significance.argmax()
    best_threshold = thresholds[index]
    best_significance = significance[index]
    
    auc_score = roc_auc_score(y_test, preds)
    #accuracy = accuracy_score(y_test, preds)
    #f1 = f1_score(y_test, preds)

    metrics = {}
    metrics["Best Threshold"] = best_threshold
    metrics["Best Significance"] = best_significance
    metrics["AUC"] = auc_score
    #metrics["Accuracy"] = accuracy
    #metrics["F1 Score"] = f1

    with open(os.path.join(plot_path, "metrics.pickle"), "wb") as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return best_threshold


best_threshold = calculate_metrics()
make_confusion_matrix(y_test, preds, p=best_threshold)
make_discriminator()
make_roc_curve()
make_pr_curve()

# maybe write code to automatically create a latex document or something based on these plots