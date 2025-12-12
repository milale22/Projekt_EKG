import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Konsolenwarnungen verhindern
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path

from fastapi import FastAPI, UploadFile

from pydantic import BaseModel
from typing import List, Dict, Any

import joblib

from ecg_project.load_ecg import load_ecg_file #,  lead_to_fhir_observation
from ecg_project.ecg_preprocess import resampling
from ecg_project.model import predict_ecg
from fastapi import FastAPI, File, HTTPException



# FastAPI initialisieren
app = FastAPI()

'''
# Ordner wo diese Datei liegt
HERE = Path(__file__).resolve().parent

# Projekt-Root (eine Ebene höher)
ROOT = HERE.parent
# artifacts/-Ordner im Projekt-Root
ARTIFACTS = ROOT / "artifacts"
# Pfad zum trainierten Modell
MODEL_PATH = ARTIFACTS / "best.keras"
# Pfad zum Scaler
SCALER_PATH = ARTIFACTS / "scaler.pkl"
# Pfad zu den Klassenlabels
CLASSES_PATH = ARTIFACTS / "classes.pkl"


# Modell laden
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    model = None

# Laden des Scalers
try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Fehler beim Laden des Scalers: {e}")
    scaler = None

# Laden der Klassenlabels
try:
    classes = joblib.load(CLASSES_PATH)  # ['CD','NORM','STTC']
except Exception as e:
    print(f"Fehler beim Laden der Klassenlabels: {e}; Klassen wurden per Fallback definiert")
    classes = ['CD','NORM','STTC']
'''

'''
class ECGInput(BaseModel):
    resourceType: str = "Observation"
    status:
    category:
    code:
    effectiveDateTime:
    valueSampledData:
'''





# Output FHIR Struktur
class ECGOutput(BaseModel):
    resourceType: str = "DiagnosticReport"
    status: str = "final" # Pflichtfeld
    category: str = "EG"
    code: Dict[str, Any] # Pflichtfeld
    note: List[Dict[str, str]] #= []
    conclusion: str



# Endpoint zum Hochladen eines EKGs
@app.post("/diagnosis", response_model=ECGOutput)
async def diagnosis_file(
        file: UploadFile = File(...),


):
    try:


        # Inhalt der Datei einlesen
        content = await file.read()
        #print("Erfolgreich geladen:", file.filename)

        # Typ der Datei bestimmen und dem Typ entsprechend laden
        kind = load_ecg_file(file.filename, content)






        # EKG Daten aus Datei laden
        signal, fs_in, lead, unit = kind


        print("EKG-Daten geladen. Samplingrate:", fs_in, "Lead:", lead, "Unit:", unit)

        #fhir = lead_to_fhir_observation(signal, fs_in, lead, unit)
        #print(fhir)
        #print(json.dumps(fhir, indent=2)[:1000], "...")

        # Signal auf benötigte Sample Rate bringen
        data = resampling(signal, fs_in)
        print("EKG-Daten resampled. Neue Form:", data.shape)

        #print("len(signal):", len(signal), "fs:", fs_in, "dtype:", signal.dtype)
        print("erste 5:", signal[:5])


        # Diagnose vom Modell berechnen
        diag, conf = predict_ecg(data)


        # Ergebnis als FHIR DiagnosticReport zurückgeben
        return ECGOutput(
            code= {"text": "ECG analysis"},
            conclusion= diag,
            note= [{"text": f"Confidence: {conf:.2f}%"}]
        )
    except Exception as e:
        raise HTTPException(status_code= 500, detail= f"Serverfehler: {e}")





# Test Endpoint zum Überprüfen ob API läuft
@app.get("/")
async def root():
    return {"message": "EKG-Klassifikations-API läuft"}
