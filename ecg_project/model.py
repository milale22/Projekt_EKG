from pathlib import Path
import keras
import joblib
from keras import layers, ops
from keras.utils import register_keras_serializable
import numpy as np
from ecg_project.ecg_preprocess import make_model_inputs


# Ordner wo diese Datei liegt
HERE = Path(__file__).resolve().parent

# Projekt-Root (eine Ebene höher)
ROOT = HERE.parent
# artifacts/-Ordner im Projekt-Root
ARTIFACTS = ROOT / "artifacts"
# Pfad zum trainierten Modell
MODEL_PATH = ARTIFACTS / "best.keras"
# Pfad zu den Klassenlabels
CLASSES_PATH = ARTIFACTS / "classes.pkl"



# Laden der Klassenlabels
try:
    classes = joblib.load(CLASSES_PATH)  # ['CD','NORM','STTC']
except Exception:
    # Fallback, falls Datei nicht vorhanden oder nicht funktioniert
    classes = ['CD','NORM','STTC']


# wichtige Zeitpunkte hervorheben
@register_keras_serializable(package='Custom', name='attention')
class attention(layers.Layer):

    # jede Zeitstelle gewichtet zurückgeben
    def __init__(self, return_sequences=True, **kwargs):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences
        #super(attention, self).__init__()

    def build(self, input_shape):

        # input_shape = (batch, time, features) (Anzahl EKGs im Batch, Anzahl Zeitpunkte, Anzahl Features pro Zeitpunkt
        # Gewichte erstellen
        time = int(input_shape[1])
        feat = int(input_shape[2])

        # für jeden Zeitpunkt eine Zahl berechnet: Wichtigkeit; W eine Art Filter, die Kombination von Features bewertet
        self.W = self.add_weight(name="att_weight", shape=(feat, 1), initializer="glorot_uniform")

        # pro Zeitstelle einen Bias; z.B bestimmte Bereiche bevorzugen
        self.b = self.add_weight(name="att_bias", shape=(time, 1), initializer="zeros")
        super().build(input_shape)


    def call(self, x):
        # (batch, time, 1), eine Zahl pro Zeitpunkt: Wichtigkeit; Werte zwischen -1 und 1
        e = ops.tanh(ops.matmul(x, self.W) + self.b)
        # macht aus Scores Gewichte, Summe über alle Zeitpunkte gleich 1
        a = ops.softmax(e, axis=1)

        # Features an wichtigen Zeitpunkten werden verstärkt
        output = x * a

        # ganze Sequenz zurückgeben: (batch, time, features) mit Attention Gewichtung
        if self.return_sequences:
            return output
        # Wichtigkeits Zusammenfassung zu einem Vektor: (batch, features)
        return ops.sum(output, axis=1)

    # für Wiederherstellung des Layers
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"return_sequences": self.return_sequences})
        return cfg


model = keras.models.load_model(MODEL_PATH, compile= True, custom_objects= {"attention": attention})


def predict_ecg(ecg_data):

    # erwartete Inputs für das Modell
    x1, x2 = make_model_inputs(ecg_data)

    # Modellvorhersage
    preds = model.predict([x1, x2])

    # Sicherstellen, dass Klassenanzahl passt
    if len(classes) != preds.shape[1]:
        raise ValueError(
            f"Anzahl der Klassen stimmt nicht: {len(classes)} Klassen, aber Modell gibt {preds.shape[1]} Outputs"
        )
    # Index der größten Zahl und entsprechenden Klassennamen herausfinden
    pred_class = classes[np.argmax(preds)]
    # die zugehörige höchste Wahrscheinlichkeit
    confidence = float(np.max(preds))

    # Diagnose mit Wahrscheinlichkeit in %
    return pred_class, round(confidence * 100, 2)

