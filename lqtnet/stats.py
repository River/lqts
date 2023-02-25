import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics

from lqtnet import train


def roc_pr_curves(y_true, y_probas, thresh, title, labels):
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

    y_thresh = np.where(y_probas>thresh, 1, 0)

    fig, ax = plt.subplots(1,3,figsize=(12,3.5))
    plt.tight_layout(pad=3)
    fig.suptitle(title, y=1)

    # Confusion matrix
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_thresh)
    print(f"Sen {cm[1][1]/(cm[1][0]+cm[1][1]):.2f}, Spe {cm[0][0]/(cm[0][0]+cm[0][1]):.2f}, \
          PPV {cm[1][1]/(cm[1][1]+cm[0][1]):.2f}, NPV {cm[0][0]/(cm[0][0]+cm[1][0]):.2f}")
    cm_disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    cm_disp.plot(cmap='Blues', ax=ax[0], colorbar=False)
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    # ROC curve
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_probas)
    auc = metrics.roc_auc_score(y_true, y_probas)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    display.plot(ax=ax[1])
    # threshold dot
    ix = np.where(roc_thresholds>=thresh)[0][-1]
    # ax[1].text(fpr[ix]+0.02,tpr[ix]-0.08, f"Sen {tpr[ix]:.2f}\nSpe {1-fpr[ix]:.2f}")
    ax[1].plot(fpr[ix],tpr[ix], marker='.', markersize=10)

    # PR curve
    precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true, y_probas)
    metrics.PrecisionRecallDisplay.from_predictions(y_true, y_probas, ax=ax[2])
    ax[2].set_ylabel("Precision")
    ax[2].set_xlabel("Recall")
    ax[2].get_legend().remove()
    # threshold dot
    ix = np.where(pr_thresholds<=thresh)[0][-1]
    ax[2].plot(recall[ix],precision[ix], marker='.', markersize=10)

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

    mean = np.mean(auc)
    std = np.std(auc)
    ci = (mean-2*std, mean+2*std)
    result = f"{mean:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"
    return(result)


def youden_thresh(y_true, y_probas):
    """
    Pick best threshold based on Youden's J statistic

    y_true: true labels (int 0 or 1)
    y_probas: predicted probabilities (float 0.0-1.0) 
    """

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas)
    J = tpr - fpr
    ix = np.argmax(J)
    return(thresholds[ix])

def best_spe_thresh(y_true, y_probas):
    """
    Pick best threshold based on maximizing specificity

    y_true: true labels (int 0 or 1)
    y_probas: predicted probabilities (float 0.0-1.0)  
    """

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas)
    ix = len(fpr)-np.argmax(np.flip(1-fpr))-1
    return(thresholds[ix])

def best_sen_thresh(y_true, y_probas):
    """
    Pick best threshold based on maximizing sensitivity

    y_true: true labels (int 0 or 1)
    y_probas: predicted probabilities (float 0.0-1.0)  
    """

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas)
    ix = np.argmax(tpr)
    return(thresholds[ix])

def lqt1_probas(y_true, y_probas):
    """
    Calculate model predictions for LQTS1 vs LQTS2

    Input is [0/1,0/1,0/1] for [control,lqts1,lqts2]
    Exclude [1,0,0] control examples
    Only output lqts1 and lqts2 examples
    y_true is the true genotype label lqts1/2
    y_pred is the predicted probability for lqts1
    """

    # Select where true label is not 'Control'
    ix = np.where(y_true[:,0]!=1)
    # y_true is label for whether ECG is 'LQTS1'
    y_true = y_true[ix, 1].flatten()
    # y_probas is probability for 'LQTS1'
    # if there is a case where the model guessed [0.33, 0.33, 0.33] (unsure if control/type 1/type 2)
    # and the true genotype was type 1, instead of saying the probability of type 1 is 0.33, really it 
    # should be 0.5 (50/50% guess)
    y_probas = np.apply_along_axis(lambda a: a[0]/(a[0]+a[1]), 1, y_probas[ix,1:3][0])

    return y_true, y_probas


if __name__ == "__main__":
    MODEL_PATH = 'models/2023.01.30-11/'
    METADATA_PATH = 'metadata/ecg_metadata_2023jan16_final.csv'
    ECG_SOURCE_DIR = 'ecgs/csv_normalized_2500/'

    (_, x_intval, x_extval, _, y_intval_true, y_extval_true) = train._import_data(
        metadata_path=METADATA_PATH,
        ecg_source_dir=ECG_SOURCE_DIR)

    model = train._load_model(MODEL_PATH)

    y_intval_pred = model.predict(x_intval)
    y_extval_pred = model.predict(x_extval)

    # Internal Validation
    # True = LQTS1/2
    # False = Control
    y_true = np.where(y_intval_true[:,0]==0, 1, 0)
    y_probas = 1 - y_intval_pred[:,0]

    t = best_sen_thresh(y_true, y_probas)
    roc_pr_curves(
        y_true, y_probas,
        thresh=t,
        title=f"LQTS1/2 carrier status (Internal Validation), Threshold={t:.2f}",
        labels=['Control', 'LQTS1/2'],
    )

    t = youden_thresh(y_true, y_probas)
    roc_pr_curves(
        y_true, y_probas,
        thresh=t,
        title=f"LQTS1/2 carrier status (Internal Validation), Threshold={t:.2f}",
        labels=['Control', 'LQTS1/2'],
    )
