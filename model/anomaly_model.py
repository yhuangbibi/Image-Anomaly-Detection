{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
from sklearn.ensemble import IsolationForest\
from sklearn.metrics import precision_score, recall_score\
\
class AnomalyModel:\
    def __init__(self):\
        # contamination = % of anomalies we expect in the dataset\
        self.model = IsolationForest(contamination=0.1, random_state=42)\
\
    def fit(self, X):\
        """Train the model on normal data"""\
        self.model.fit(X)\
\
    def predict(self, X):\
        """Predict anomalies: -1 = anomaly, 1 = normal"""\
        return self.model.predict(X)\
\
\
if __name__ == "__main__":\
    # --- Step 1: Generate fake data (as placeholder for images) ---\
    # "normal" data = cluster around (0,0), "anomalies" = around (5,5)\
    normal_data = np.random.normal(loc=0, scale=1, size=(200, 2))\
    anomaly_data = np.random.normal(loc=5, scale=1, size=(40, 2))\
\
    X_train = normal_data  # train only on normal data\
    X_test = np.vstack([normal_data, anomaly_data])\
    y_test = np.array([1] * len(normal_data) + [-1] * len(anomaly_data))  # 1=normal, -1=anomaly\
\
    # --- Step 2: Train model ---\
    model = AnomalyModel()\
    model.fit(X_train)\
\
    # --- Step 3: Predictions ---\
    preds = model.predict(X_test)\
\
    # --- Step 4: Evaluate ---\
    precision = precision_score(y_test, preds)\
    recall = recall_score(y_test, preds)\
\
    print("Precision:", precision)\
    print("Recall:", recall)\
    print("Sample predictions (first 20):", preds[:20])\
}