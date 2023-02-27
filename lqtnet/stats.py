import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.utils import resample

from lqtnet import train


def baseline_chars(df):
    # age
    # sex
    # ethnicity
    # HR
    # QTc, confirmed?

    print("(N=%d)" % df.shape[0])

    print("Age: %d±%d" % (df.age.mean(), df.age.std()))
    num_female = df[df.sex == "Female"].shape[0]
    print("Female no (%%): %d (%.1f%%)" % (num_female, num_female / df.shape[0] * 100))
    print("Ethnicity:")
    print(
        pd.DataFrame(
            {
                "n": df.ethnicity.value_counts(),
                "%": df.ethnicity.value_counts(normalize=True).round(3) * 100,
            }
        )
    )
    print("HR: %d±%d" % (df.hr.mean(), df.hr.std()))

    # num_qt_known = df.qt.notna().sum()
    # print("QT known (%%): %d (%.1f%%)" % (num_qt_known,num_qt_known/df.shape[0]*100))

    num_qt_confirmed = df[df.qt_confirmed == True].shape[0]
    print(
        "QT confirmed (%%): %d (%.1f%%)"
        % (num_qt_confirmed, num_qt_confirmed / df.shape[0] * 100)
    )

    print(
        "QTc: %d±%d"
        % (
            df.query("qt_confirmed==True").qt.mean(),
            df.query("qt_confirmed==True").qt.std(),
        )
    )

    # num_qt_prolonged = df[df.qt_prolonged==True].shape[0]
    num_qt_prolonged = df.query("qt_confirmed==True and qt_prolonged==True").shape[0]
    print(
        "QT prolonged (%%): %d (%.1f%%)"
        % (num_qt_prolonged, num_qt_prolonged / df.shape[0] * 100)
    )


def lqts_carrier_true_label_and_probas(df):
    """
    LQTS1/2 carrier status label and predictions
    When predictions are already saved to the DataFrame

    Returns:
    y_true: nparray, 1 if LQTS1/2, 0 if control
    y_proba: nparray, 0-1 prediction for LQTS1/2 carrier status
    """

    # true labels
    y_true = df.lqts_type.to_numpy()
    type_1 = np.where(y_true == "Type 1", True, False)
    type_2 = np.where(y_true == "Type 2", True, False)
    y_true = np.logical_or(type_1, type_2).astype(int)

    # predictions
    y_probas = (1 - df.pred0).to_numpy()

    return y_true, y_probas


def lqts1_vs_lqts2_true_label_and_probas(df):
    """
    LQTS1 vs LQTS2 label and predictions
    When predictions are already saved to the DataFrame

    Returns:
    y_true: nparray, 0 if LQTS1, 1 if LQTS2
    y_proba: nparray, 0-1 prediction for LQTS2
    """

    # Filter everything not LQTS1 or LQTS2
    df_ = df.query("lqts_type=='Type 1' or lqts_type=='Type 2'")

    # true labels for LQTS1
    y_true = np.where(df_.lqts_type.to_numpy() == "Type 1", 0, 1)

    # predicted LQTS1 probabilities
    # proba is pred(LQTS1) / ( pred(LQTS1) + pred(LQTS2) )
    # if the model predicted control=0.33, lqts1=0.33, lqts2=0.33, the output pred(lqts1) is 0.5
    y_probas = (df_.pred2 / (df_.pred1 + df_.pred2)).to_numpy()

    return y_true, y_probas


def lqt1_probas(y_true, y_probas):
    """
    Calculate model predictions for LQTS1 vs LQTS2

    Input is [0/1,0/1,0/1] for [control,lqts1,lqts2]
    Exclude [1,0,0] control examples
    Only output lqts1 and lqts2 examples

    Returns:
    y_true = true genotype label lqts1/2
    y_pred = predicted probability for lqts1
    """

    # Select where true label is not 'Control'
    ix = np.where(y_true[:, 0] != 1)
    # y_true is label for whether ECG is 'LQTS1'
    y_true = y_true[ix, 1].flatten()

    # y_probas is probability for 'LQTS1'
    # if there is a case where the model guessed [0.33, 0.33, 0.33] (unsure if control/type 1/type 2)
    # and the true genotype was type 1, instead of saying the probability of type 1 is 0.33, really it
    # should be 0.5 (50/50% guess)
    y_probas = np.apply_along_axis(
        lambda a: a[0] / (a[0] + a[1]), 1, y_probas[ix, 1:3][0]
    )

    return y_true, y_probas


def roc_pr_curves(y_true, y_probas, thresh, title, labels, auc_text=None):
    """
    Plot confusion matrix, ROC curve, PR curve

    Need to pick threshold value first, for confusion matrix values, and also plots a
    point on the ROC and PR curves corresponding to the threshold.

    y_true: true labels (int 0 or 1)
    y_probas: predicted probabilities (float 0.0-1.0)
    thresh: float 0.0-1.0
    title: figure title
    labels: array(2) category labels
    """

    y_thresh = np.where(y_probas > thresh, 1, 0)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    plt.tight_layout(pad=3)
    fig.suptitle(title, y=1)

    # Confusion matrix
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_thresh)
    sen = cm[1][1] / (cm[1][0] + cm[1][1])
    spe = cm[0][0] / (cm[0][0] + cm[0][1])
    ppv = cm[1][1] / (cm[1][1] + cm[0][1])
    npv = cm[0][0] / (cm[0][0] + cm[1][0])

    print(f"Sen {sen:.2f}, Spe {spe:.2f}, PPV {ppv:.2f}, NPV {npv:.2f}")
    cm_disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    cm_disp.plot(cmap="Blues", ax=ax[0], colorbar=False)
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    # ROC curve
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_probas)
    auc = metrics.roc_auc_score(y_true, y_probas)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    display.plot(ax=ax[1])
    if auc_text:
        ax[1].legend(labels=[auc_text], loc="lower right")
    # threshold dot
    ix = np.where(thresh <= roc_thresholds)[0][-1]
    # ax[1].text(fpr[ix]+0.02,tpr[ix]-0.08, f"Sen {tpr[ix]:.2f}\nSpe {1-fpr[ix]:.2f}")
    ax[1].plot(fpr[ix], tpr[ix], marker=".", markersize=10)
    ax[1].text(
        fpr[ix] + 0.03,
        tpr[ix] - 0.01,
        f"Sen {sen:.2f}\nSpe {spe:.2f}",
        ha="left",
        va="top",
    )

    # PR curve
    precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true, y_probas)
    metrics.PrecisionRecallDisplay.from_predictions(y_true, y_probas, ax=ax[2])
    ax[2].set_ylabel("Precision")
    ax[2].set_xlabel("Recall")
    ax[2].get_legend().remove()
    # threshold dot
    ix = np.where(thresh < pr_thresholds)[0][0]
    ax[2].plot(recall[ix], precision[ix], marker=".", markersize=10)
    ax[2].text(
        recall[ix] - 0.03, precision[ix] - 0.01, f"PPV {ppv:.2f}", ha="right", va="top"
    )


def auc_ci(y_true, y_probas, n_samples):
    """
    Calculate AUC and 95% confidence interval (+/- 2 std around mean)
    Bootstrapping technique, sampling with replacement
    n_samples = number of samples to calculate
    """

    auc = []
    for _ in range(n_samples):
        y_true_sample, y_probas_sample = resample(y_true, y_probas, replace=True)
        auc.append(metrics.roc_auc_score(y_true_sample, y_probas_sample))

    percentiles = [2.5, 97.5]  # for 95% confidence interval
    ci = np.percentile(auc, percentiles)

    mean = np.mean(auc)

    result = f"AUC = {mean:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"
    return result


def youden_thresh(y_true, y_probas):
    """
    Pick best threshold based on Youden's J statistic

    y_true: true labels (int 0 or 1)
    y_probas: predicted probabilities (float 0.0-1.0)
    """

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas)
    J = tpr - fpr
    ix = np.argmax(J)
    return thresholds[ix]


def best_spe_thresh(y_true, y_probas):
    """
    Pick best threshold based on maximizing specificity

    y_true: true labels (int 0 or 1)
    y_probas: predicted probabilities (float 0.0-1.0)
    """

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas)
    ix = len(fpr) - np.argmax(np.flip(1 - fpr)) - 1
    return thresholds[ix]


def best_sen_thresh(y_true, y_probas):
    """
    Pick best threshold based on maximizing sensitivity

    y_true: true labels (int 0 or 1)
    y_probas: predicted probabilities (float 0.0-1.0)
    """

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas)
    ix = np.argmax(tpr)
    return thresholds[ix]
