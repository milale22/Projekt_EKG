from pathlib import Path

from fastapi import FastAPI, UploadFile

import keras
from pydantic import BaseModel
#from typing import List

import joblib

from ecg_project.ecg_preprocess import load_ecg_wfdb_zip, resampling, check_format
from ecg_project.model import predict_ecg
from fastapi import FastAPI, File, HTTPException



app = FastAPI()

'''
MODEL_PATH = "../artifacts/best.keras"
model = keras.models.load_model(MODEL_PATH)

SCALER_PATH = "../artifacts/scaler.pkl"
CLASSES_PATH = "../artifacts/classes.pkl"
'''

HERE = Path(__file__).resolve().parent

# Projekt-Root (eine Ebene höher)
ROOT = HERE.parent
# artifacts/-Ordner im Projekt-Root
ARTIFACTS = ROOT / "artifacts"

MODEL_PATH = ARTIFACTS / "best.keras"
SCALER_PATH = ARTIFACTS / "scaler.pkl"
CLASSES_PATH = ARTIFACTS / "classes.pkl"
model = keras.models.load_model(MODEL_PATH)


try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None

try:
    classes = joblib.load(CLASSES_PATH)  # ['CD','NORM','STTC']
except Exception:
    classes = ['CD','NORM','STTC']





class ECGOutput(BaseModel):
    diagnosis: str
    confidence: float


@app.post("/diagnosis", response_model=ECGOutput)
async def diagnosis_file(
        file: UploadFile = File(...),


):
    try:
        check_format(file.filename)

        content = await file.read()
        print("Erfolgreich geladen:", file.filename)




        signal, fs_in = load_ecg_wfdb_zip(content)

        data = resampling(signal, fs_in)

        #x_signal = sig_rs.reshape(1, -1, 1)


        '''
        y = predict_ecg(sig_rs)

        pred_class = classes[int(y.argmax())]
        pred_conf  = float(y.max())
        '''
        diag, conf = predict_ecg(data)


        #return ECGOutput(diagnosis=pred_class, confidence=pred_conf)
        return ECGOutput(diagnosis=diag, confidence=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Serverfehler: {e}")





#Test Endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}
