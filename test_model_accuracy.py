import pytest
import numpy as np
from sklearn.metrics import precision_score, recall_score
from model.anomaly_model import AnomalyModel

def test_model_precision_and_recall(normal_data, anomaly_data):
    # data preparation
    X = np.vstack([normal_data, anomaly_data])
    y = np.array([1] * len(normal_data) + [-1] * len(anomaly_data))

    model = AnomalyModel()
    model.fit(normal_data)  # train with normal data
    preds = model.predict(X)

    precision = precision_score(y, preds)
    recall = recall_score(y, preds)

    assert precision > 0.9
    assert recall > 0.85
