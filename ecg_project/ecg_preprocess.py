import tempfile
import zipfile
from io import BytesIO
from os import listdir, path
from pathlib import Path

import joblib
import numpy as np
import wfdb
from scipy.signal import spectrogram, resample

#SCALER_PATH = "../artifacts/scaler.pkl"
#CLASSES_PATH = "../artifacts/classes.pkl"

HERE = Path(__file__).resolve().parent

# Projekt-Root (eine Ebene höher)
ROOT = HERE.parent
# artifacts/-Ordner im Projekt-Root
ARTIFACTS = ROOT / "artifacts"


SCALER_PATH = ARTIFACTS / "scaler.pkl"
CLASSES_PATH = ARTIFACTS / "classes.pkl"


try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None

try:
    classes = joblib.load(CLASSES_PATH)  # ['CD','NORM','STTC']
except Exception:
    classes = ['CD','NORM','STTC']       # fallback

# Standardizing function
def standardizing(ecg_data):
    ecg2d = ecg_data.reshape(-1, 1)
    if scaler is not None:
        ecg_scaled = scaler.transform(ecg2d)
    else:
        mu = ecg2d.mean(axis=0, keepdims=True)
        sd = ecg2d.std(axis=0, keepdims=True) + 1e-8
        ecg_scaled = (ecg2d - mu) / sd
    return ecg_scaled
    #return ecg_scaled.reshape(ecg_data.shape).astype('float')

# Spectrogram function
def create_specto(ecg_data, fs=100, nperseg=40, noverlap=10, freq_bands=13):
    sig = np.asarray(ecg_data).squeeze()
    nper = min(nperseg, sig.shape[0])
    nover = min(noverlap, max(0, nper - 1))

    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nper, noverlap=nover)

    Sxx = Sxx[:freq_bands].transpose()
    Sxx = np.log(Sxx + 1e-8)


    return Sxx


def make_model_inputs(ecg_data):

    ecg_std = standardizing(ecg_data)

    if ecg_std.shape != (1000, 1):
        raise ValueError(
            f"Falsches Input-Shape: erwartet wird (1000,1), aber bekommen {ecg_data.shape}"
        )



    spec = create_specto(ecg_std)

    x1 = ecg_std[None, ...]
    x2 = spec[None, ...]

    return x1, x2


# 10 sekunden
def resampling(ecg_data, fs_in:int, sec=10, fs_out=100):
    ecg2d = ecg_data[:, None]
    len_in = sec * fs_in
    if len(ecg2d) > len_in:
       ecg2d = ecg2d[:len_in]     # cut die samples der ersten 10 sekunden


    # Resampling auf fs_out
    len_out = sec * fs_out
    ecg_resampled = resample(ecg2d, len_out)

    #return ecg_resampled.reshape(-1, 1)
    return ecg_resampled
    # TODO Fehler bei unter 10 sekunden: fehlermeldung ekg muss mindestens 10 sekunden lang sein


def check_format(filename:str):
    import os
    ext = os.path.splitext(filename)[1]
    if ext in (".csv", ".json", ".zip"):
        return{
            ".csv": "csv", ".json": "json", ".zip": "wfdb_zip"}[ext]

    raise ValueError(f"Dateiformat {ext} wird nicht unterstützt. Nur .csv, .json und .zip (mit .hea + .dat) sind erlaubt.")

# TODO: Prüfen auf die verschiedenene Formate, was bei jeweiligem Format getan werden muss

def load_ecg_file(filename:str, content:bytes):
    kind = check_format(filename)
    '''
    if kind == "csv":
        #load_ecg_csv(content)

    elif kind == "json":
        #load_ecg_json(content)

    elif kind == "wfdb_zip":
        load_ecg_wfdb_zip(content)
    '''
    if kind == "wfdb_zip":
        return load_ecg_wfdb_zip(content)

# TODO: CSV laden
#def load_ecg_csv(content:bytes):

# TODO: JSON laden
#def load_ecg_json(content:bytes):

def load_ecg_wfdb_zip(content:bytes):
    with tempfile.TemporaryDirectory() as tmpdirname:

        with zipfile.ZipFile(BytesIO(content)) as zip_ref:
            zip_ref.extractall(tmpdirname)

        names = listdir(tmpdirname)
        hea_bases = {path.splitext(n)[0] for n in names if n.lower().endswith(".hea")}
        dat_bases = {path.splitext(n)[0] for n in names if n.lower().endswith(".dat")}

        pairs = sorted(hea_bases & dat_bases)  # Schnittmenge
        if not pairs:
           raise ValueError("Kein .hea/.dat-Paar im ZIP gefunden.")

        base = path.join(tmpdirname, pairs[0])

        rec = wfdb.rdrecord(base)
        signal = rec.p_signal[:, 0]  # nur die erste Ableitung
        fs = rec.fs


        return signal, fs


