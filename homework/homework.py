import os
import glob
import gzip
import json
import pickle
import zipfile
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def load_data():

    dataframe_test = pd.read_csv(
        "./files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )

    dataframe_train = pd.read_csv(
        "./files/input/train_data.csv.zip",
        index_col=False,
        compression="zip",
    )

    return dataframe_train, dataframe_test


def clean_data(df):
    df_copy = df.copy()
    df_copy = df_copy.rename(columns={"default payment next month": "default"})
    df_copy = df_copy.drop(columns=["ID"])
    df_copy = df_copy.loc[df["MARRIAGE"] != 0]
    df_copy = df_copy.loc[df["EDUCATION"] != 0]
    df_copy["EDUCATION"] = df_copy["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    df_copy = df_copy.dropna()
    return df_copy


def split_data(df):
    return df.drop(columns=["default"]), df["default"]


def create_pipeline(x_train):
    categorical_columns = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_columns = list(set(x_train.columns).difference(categorical_columns))
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            (
                "scaler",
                StandardScaler(with_mean=True, with_std=True),
                numerical_columns,
            ),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("classifier", SVC(kernel="rbf", random_state=12345, max_iter=-1)),
        ]
    )

    return pipeline


def create_estimator(pipeline, x_train):

    param_grid = {
        "pca__n_components": [20, x_train.shape[1] - 2],
        "feature_selection__k": [12],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.1],
    }

    cv = StratifiedKFold(n_splits=10)

    scorer = make_scorer(balanced_accuracy_score)

    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring=scorer, cv=cv, n_jobs=-1
    )

    return grid_search


def _create_output_directory(output_directory):
    if os.path.exists(output_directory):
        for file in glob(f"{output_directory}/*"):
            os.remove(file)
        os.rmdir(output_directory)
    os.makedirs(output_directory)


def _save_model(path, estimator):
    _create_output_directory("files/models/")

    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def calculate_precision_metrics(dataset_type, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def calculate_confusion_matrix(dataset_type, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }


data_train, data_test = load_data()
data_train = clean_data(data_train)
data_test = clean_data(data_test)
x_train, y_train = split_data(data_train)
x_test, y_test = split_data(data_test)
pipeline = create_pipeline(x_train)

estimator = create_estimator(pipeline, x_train)
estimator.fit(x_train, y_train)

_save_model(
    os.path.join("files/models/", "model.pkl.gz"),
    estimator,
)

y_test_pred = estimator.predict(x_test)
test_precision_metrics = calculate_precision_metrics("test", y_test, y_test_pred)
y_train_pred = estimator.predict(x_train)
train_precision_metrics = calculate_precision_metrics("train", y_train, y_train_pred)

test_confusion_metrics = calculate_confusion_matrix("test", y_test, y_test_pred)
train_confusion_metrics = calculate_confusion_matrix("train", y_train, y_train_pred)

os.makedirs("files/output/", exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as file:
    file.write(json.dumps(train_precision_metrics) + "\n")
    file.write(json.dumps(test_precision_metrics) + "\n")
    file.write(json.dumps(train_confusion_metrics) + "\n")
    file.write(json.dumps(test_confusion_metrics) + "\n")