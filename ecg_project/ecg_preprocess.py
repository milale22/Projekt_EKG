from pathlib import Path
import joblib
import numpy as np
from scipy.signal import spectrogram, resample




# Ordner wo diese Datei liegt
HERE = Path(__file__).resolve().parent

# Projekt-Root (eine Ebene höher)
ROOT = HERE.parent
# artifacts/-Ordner im Projekt-Root
ARTIFACTS = ROOT / "artifacts"
# Pfad zum Scaler
SCALER_PATH = ARTIFACTS / "scaler.pkl"
# Pfad zu den Klassenlabels
CLASSES_PATH = ARTIFACTS / "classes.pkl"

# Laden des Scalers
try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None


# Laden der Klassenlabels
try:
    classes = joblib.load(CLASSES_PATH)  # ['CD','NORM','STTC']
except Exception:
    # Fallback falls Datei nicht vorhanden oder nicht funktioniert
    classes = ['CD','NORM','STTC']

# Funktion zum Standardisieren des EKGs
def standardizing(ecg_data):
    # ecg_data wird in 2D gebracht
    # -1: Länge automatisch selbst berechnen
    ecg2d = ecg_data.reshape(-1, 1)

    # geladene Scaler nutzen um Werte zu standardisieren
    if scaler is not None:
        ecg_scaled = scaler.transform(ecg2d)
    # falls Scaler nicht korrekt geladen wurde, manuelle Standardisierung
    else:
        # Mittelwert der Werte berechnen
        # axis = 0: über alle Zeilen/Samples; Ergbnis bleibt 2D (1,1)
        mu = ecg2d.mean(axis=0, keepdims=True)
        # Standardabweichung berechnen
        # + 1e-8 um Division durch 0 zu vermeiden
        sd = ecg2d.std(axis=0, keepdims=True) + 1e-8

        # Standardisierung = (Wert - Mittelwert)/Standardabweichung
        # Ergebnis: Verteilung mit Mittelwert ~0 und Standardabweichung ~1
        ecg_scaled = (ecg2d - mu) / sd

    # standardisierte EKG mit Shape (N,1) zurückgeben
    return ecg_scaled


# Funktion, die aus dem EKG ein Spektrogramm (Zeit-Frequenz-Darstellung) erzeugt
def create_specto(ecg_data, fs=100, nperseg=40, noverlap=10, freq_bands=13):
    # ecg_data in numpy array umwandeln und überflüssige Dimensionen entfernen
    # zb aus (1000,1) wird (1000,)
    sig = np.asarray(ecg_data).squeeze()
    # nper = tatsächliche Fenstergröße
    # Minimum aus gewünschter Fensterlänge (nperseg) und Signallänge nehmen, falls Signal kürzer als nperseg ist
    nper = min(nperseg, sig.shape[0])

    # nover = tatsächliche Überlappung zwischen Fenstern
    # darf nicht größer als nper-1 und nicht größer als noverlap sein
    nover = min(noverlap, max(0, nper - 1))

    # Spektrogramm berechnen
    # f = Frequenzachse, t = Zeitachsen Werte, Sxx = Spektrogramm
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nper, noverlap=nover)

    # nur die ersten freq_bands Frequenzbänder werden behalten
    # Sxx ursprünglich (Anzahl Frequenzen, Anzahl Zeitfenster)
    # durch Transpose: (Anzahl Zeitfenster, freq_bands)
    Sxx = Sxx[:freq_bands].transpose()
    # log anwenden, um große Werte abzuflachen und +1e-8 um log(0) zu verhindern
    Sxx = np.log(Sxx + 1e-8)

    # Spektrogramm zurückgeben
    return Sxx

# Funktion, die aus dem EKG die beiden Modell Inputs baut
# x1 = standardisierte Rohsignal, x2 = Spektrogramm
def make_model_inputs(ecg_data):

    # zuerst EKG standardisieren
    ecg_std = standardizing(ecg_data)

    # Prüfen, ob die Form genau (1000,1) ist
    # 1000 Samples und ein Kanal
    # Modell erwartet dieses Größe bei zehn Sekunden bei 100 Hz
    if ecg_std.shape != (1000, 1):
        raise ValueError(
            f"Falsches Input-Shape: erwartet wird (1000,1), aber bekommen {ecg_std.shape}"
        )


    # aus standardisierten Signalen Spektrogramm erzeugen
    spec = create_specto(ecg_std)

    # x1 = Rohsignal mit zusätzlicher Batch Dimension (1, 1000, 1)
    x1 = ecg_std[None, ...]
    # x2 = Spektrogramm mit zusätzlicher Batch Dimension (1, T, freq_bands)
    x2 = spec[None, ...]

    # Inputs zurückgeben
    return x1, x2


# Funktion zum Resampling: prüft auf Dauer, Umwandlung auf neue Abtastrate
def resampling(ecg_data, fs_in:int, sec=10, fs_out=100):
    # ecg_data in 2D: (N, 1)
    ecg2d = ecg_data[:, None]

    # len_in = wie viele Samples sollten 10 Sekunden bei der eingegangenen Abtastrate haben
    len_in = int(sec * fs_in)

    # wenn das Signal zu kurz ist (kürzer als 10 Sekunden)
    if len(ecg2d) < len_in:
        raise ValueError(
            f"EKG zu kurz, wird mindestens {sec} Sekunden benötigt, bekommt aber {len(ecg2d)} Sekunden."
        )

    # wenn das Signal zu lang ist (länger als 10 Sekunden)
    if len(ecg2d) > len_in:
        # auf die ersten len_in Samples reduzieren
        # nur die ersten sec Sekunden werden behalten
        ecg2d = ecg2d[:len_in]


    # Resampling auf neue Abtastrate fs_out
    # len_out = Anzahl der Samples am Ende
    len_out = int(sec * fs_out)
    # skaliert das Signal von len_in auf len_out Punkze
    ecg_resampled = resample(ecg2d, len_out)

    # resampelte Signal zurückgeben (len_out, 1)
    #return ecg_resampled.reshape(-1, 1)
    return ecg_resampled


