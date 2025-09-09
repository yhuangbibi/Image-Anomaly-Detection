{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pytest\
import numpy as np\
from sklearn.metrics import precision_score, recall_score\
from model.anomaly_model import AnomalyModel\
\
def test_model_precision_and_recall(normal_data, anomaly_data):\
    # \uc0\u20934 \u22791 \u25968 \u25454 \
    X = np.vstack([normal_data, anomaly_data])\
    y = np.array([1] * len(normal_data) + [-1] * len(anomaly_data))\
\
    model = AnomalyModel()\
    model.fit(normal_data)  # \uc0\u29992 \u27491 \u24120 \u25968 \u25454 \u35757 \u32451 \
    preds = model.predict(X)\
\
    precision = precision_score(y, preds)\
    recall = recall_score(y, preds)\
\
    assert precision > 0.9\
    assert recall > 0.85\
}