import os
import pickle
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from preprocessors.wav_preprocessor import preprocess_wav
from prerunners.form_getter import form_getter


def train_random_forest_midi(df: pd.DataFrame):

    df["form"] = df["canonical_title"].apply(lambda x: form_getter(x))

    X = np.array(df["note"].to_list())
    y_form = df["form"]
    y_composer = df["canonical_composer"]

    X_train = X[df["split"] == "train"]
    X_test = X[df["split"] == "test"]
    y_train_form = y_form[df["split"] == "train"]
    y_test_form = y_form[df["split"] == "test"]
    y_train_composer = y_composer[df["split"] == "train"]
    y_test_composer = y_composer[df["split"] == "test"]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run():
        rf_form = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_form.fit(X_train, y_train_form)
        y_pred_form = rf_form.predict(X_test)
        form_accuracy = accuracy_score(y_test_form, y_pred_form)
        print("Form Classification Accuracy:", form_accuracy)

        mlflow.log_metric("form_accuracy", float(form_accuracy))
        mlflow.sklearn.log_model(rf_form, "random_forest_form")

        rf_composer = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_composer.fit(X_train, y_train_composer)
        y_pred_composer = rf_composer.predict(X_test)
        composer_accuracy = accuracy_score(y_test_composer, y_pred_composer)
        print("Composer Classification Accuracy:", composer_accuracy)

        mlflow.log_metric("composer_accuracy", float(composer_accuracy))
        mlflow.sklearn.log_model(rf_composer, "random_forest_composer")

        os.makedirs("models", exist_ok=True)
        combined_model_path = os.path.join("models/", "random_forest_models_midi.pkl")
        with open(combined_model_path, "wb") as f:
            pickle.dump({"form_model": rf_form, "composer_model": rf_composer}, f)
        print(f"Combined RandomForest models saved to {combined_model_path}")
