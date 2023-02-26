import os
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow import keras

from lqtnet import import_ecgs
from lqtnet.networks import convnet, resnet


def _import_data(df, ecg_source_dir):
    """
    IMPORT DATA
    Edit the pandas df queries here to adjust which ECGs are included in training

    returns pandas dataframes:
        train: training dataset
        test: internal validation dataset
        ext: external validation dataset
    """

    train_df = df[(df.set == "Derivation") & (df.qc == "Good")]
    test_df = df[(df.set == "Internal validation") & (df.qc == "Good")]
    ext_df = df[(df.set == "External validation") & (df.qc == "Good")]

    x_train = import_ecgs.df_import_csv_to_numpy(train_df, from_dir=ecg_source_dir)
    x_test = import_ecgs.df_import_csv_to_numpy(test_df, from_dir=ecg_source_dir)
    x_ext = import_ecgs.df_import_csv_to_numpy(ext_df, from_dir=ecg_source_dir)

    y_train = import_ecgs.df_to_np_labels(train_df)
    y_test = import_ecgs.df_to_np_labels(test_df)
    y_ext = import_ecgs.df_to_np_labels(ext_df)

    return (x_train, x_test, x_ext, y_train, y_test, y_ext)


def _build_model(experiment):
    """
    Build model, either ConvNet or ResNet
    """
    if experiment.get_parameter("model_type") == "ResNet":
        model = resnet.ResNet(num_outputs=3)
        model.build((None, 2500, 8))
    elif experiment.get_parameter("model_type") == "ConvNet":
        model = convnet.build_model()

    model.compile(
        loss=LOSS,
        optimizer=OPTIMIZER,
        metrics=["accuracy"],
    )
    print(model.summary())
    return model


def _load_model(model_path: str):
    """
    Load saved model (for predictions later)
    """
    return keras.models.load_model(model_path)


def _train_and_save_model(
    experiment, model, x_train, y_train, x_test, y_test, model_save_dir=MODEL_SAVE_DIR
):
    """
    Train and save model
    """
    model.fit(
        x=x_train,
        y=y_train,
        epochs=experiment.get_parameter("epochs"),
        batch_size=experiment.get_parameter("batch_size"),
        verbose=1,
        validation_data=(x_test, y_test),
    )

    # Create new unique save path
    date = datetime.today().strftime("%Y.%m.%d")
    i = 1
    save_path = f"{model_save_dir}{date}-"
    while os.path.exists(f"{save_path}{i}"):
        i += 1
    save_path = f"{save_path}{i}"
    print(f"Saving model to {save_path}")
    model.save(f"{save_path}")

    return (model, save_path)


def _eval_model(experiment, model, x_test, y_test, x_ext, y_ext):
    """
    Evaluating model
    Need to pass in the CometML experiment object because I use their methods for logging
    confusion matrix and pyplot graphs
    """
    y_test_pred = model.predict(x_test)
    y_test_pred_max = np.argmax(y_test_pred, axis=1)

    y_ext_pred = model.predict(x_ext)
    y_ext_pred_max = np.argmax(y_ext_pred, axis=1)

    experiment.log_confusion_matrix(
        y_true=np.argmax(y_test, 1),
        y_predicted=y_test_pred_max,
        labels=["Control", "Type 1", "Type 2"],
        title="Internal validation",
        file_name="internal-validation.json",
        row_label="Actual Category",
        column_label="Predicted Category",
    )

    experiment.log_confusion_matrix(
        y_true=np.argmax(y_ext, 1),
        y_predicted=y_ext_pred_max,
        labels=["Control", "Type 1", "Type 2"],
        title="External validation",
        file_name="external-validation.json",
        row_label="Actual Category",
        column_label="Predicted Category",
    )

    def roc_curve(y_true, y_probas, title):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas)
        auc = metrics.roc_auc_score(y_true, y_probas)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
        fig, ax = plt.subplots()
        J = tpr - fpr
        ix = np.argmax(J)
        display.plot(ax=ax)
        plt.plot(fpr[ix], tpr[ix], marker=".", markersize=10)
        plt.text(
            fpr[ix] + 0.02,
            tpr[ix] - 0.08,
            f"Sen {round(tpr[ix],2)}, Spe {round(1-fpr[ix],2)}",
        )
        plt.title(title)
        experiment.log_figure(figure=plt, figure_name=title)

    def pr_curve(y_true, y_probas, title):
        f1_score = metrics.f1_score(y_true, np.around(y_probas))
        fig, ax = plt.subplots()
        metrics.PrecisionRecallDisplay.from_predictions(
            y_true, y_probas, ax=ax, name=f"F1 score = {f1_score:.3f}"
        )
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        plt.title(title)
        experiment.log_figure(figure=plt, figure_name=title)

    y_test_true, y_test_probas = y_test[:, 0], y_test_pred[:, 0]
    y_ext_true, y_ext_probas = y_ext[:, 0], y_ext_pred[:, 0]
    roc_curve(
        y_test_true, y_test_probas, "LQTS1/2 carrier status (Internal Validation)"
    )
    pr_curve(y_test_true, y_test_probas, "LQTS1/2 carrier status (Internal Validation)")
    roc_curve(y_ext_true, y_ext_probas, "LQTS1/2 carrier status (External Validation)")
    pr_curve(y_ext_true, y_ext_probas, "LQTS1/2 carrier status (External Validation)")

    """
    DISTINGUISHING LQTS TYPE 1 VS 2

    start off with selecting known examples of LQTS1/2 (which is already backwards)
    then using the predicted probability of LQTS1 alone to distinguish type 1 vs type 2

    if there is a case where the model guessed [0.33, 0.33, 0.33] (unsure if control/type 1/type 2)
    and the true genotype was type 1
    instead of saying the probability of type 1 is 0.33
    really it should be 0.5 (50/50% guess)
    """

    lqt_test_index = np.where(y_test[:, 0] != 1)
    y_true = y_test[lqt_test_index, 1].flatten()
    y_pred = np.apply_along_axis(
        lambda a: a[0] / (a[0] + a[1]), 1, y_test_pred[lqt_test_index, 1:3][0]
    )
    roc_curve(y_true, y_pred, title="LQTS type 1 vs type 2 (Internal Validation)")
    pr_curve(y_true, y_pred, title="LQTS type 1 vs type 2 (Internal Validation)")

    lqt_ext_index = np.where(y_ext[:, 0] != 1)
    y_true = y_ext[lqt_ext_index, 1].flatten()
    y_pred = np.apply_along_axis(
        lambda a: a[0] / (a[0] + a[1]), 1, y_ext_pred[lqt_ext_index, 1:3][0]
    )
    roc_curve(y_true, y_pred, title="LQTS type 1 vs type 2 (External Validation)")
    pr_curve(y_true, y_pred, title="LQTS type 1 vs type 2 (External Validation)")
