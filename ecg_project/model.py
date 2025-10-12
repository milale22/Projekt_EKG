#import keras.backend as K Problem wegen keras.ops 3 und soo
from pathlib import Path

import keras
import joblib
from keras import layers, ops
from keras.utils import register_keras_serializable
import numpy as np
from ecg_project.ecg_preprocess import make_model_inputs


#CLASSES_PATH = "../artifacts/classes.pkl"
HERE = Path(__file__).resolve().parent

# Projekt-Root (eine Ebene höher)
ROOT = HERE.parent
# artifacts/-Ordner im Projekt-Root
ARTIFACTS = ROOT / "artifacts"

MODEL_PATH = ARTIFACTS / "best.keras"

CLASSES_PATH = ARTIFACTS / "classes.pkl"



try:
    classes = joblib.load(CLASSES_PATH)  # ['CD','NORM','STTC']
except Exception:
    classes = ['CD','NORM','STTC']



@register_keras_serializable(package='Custom', name='attention')
#@keras.saving.register_keras_serializable(package="custom", name="attention")
class Attention(layers.Layer):

    def __init__(self, return_sequences=True, **kwargs):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences
        #super(attention, self).__init__()

    def build(self, input_shape):
        '''
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(attention, self).build(input_shape)
        '''

        # input_shape = (batch, time, features)
        time = int(input_shape[1])
        feat = int(input_shape[2])
        self.W = self.add_weight(
            name="att_weight", shape=(feat, 1), initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            name="att_bias", shape=(time, 1), initializer="zeros"
        )
        super().build(input_shape)


    def call(self, x):
        e = ops.tanh(ops.matmul(x, self.W) + self.b)
        a = ops.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return ops.sum(output, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"return_sequences": self.return_sequences})
        return cfg


#MODEL_PATH = "../artifacts/best.keras"
model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects={"attention": Attention})


def predict_ecg(ecg_data):

    x1, x2 = make_model_inputs(ecg_data)
    preds = model.predict([x1, x2])

    if len(classes) != preds.shape[1]:
        raise ValueError(
            f"Anzahl der Klassen stimmt nicht: {len(classes)} Klassen, aber Modell gibt {preds.shape[1]} Outputs"
        )
    pred_class = classes[np.argmax(preds)]

    confidence = float(np.max(preds))
    '''
    return {
        "vorhergesagte Klasse": pred_class,
        "Wahrscheinlichkeit": round(confidence * 100, 2) # vllt noch Prozentzeichen oder in der api?
    }
    '''
    return pred_class, round(confidence * 100, 2)


'''
dummy_ecg = np.zeros((1000, 1), dtype=np.float32)


X1 = dummy_ecg[np.newaxis, ...]  # (1,1000,1)

X2 = create_specto(dummy_ecg)[np.newaxis, ...]  # (1,33,13)

# Testvorhersage
probs = model.predict([X1, X2], verbose=0)
print("Test-Prediction shape:", probs.shape)
'''