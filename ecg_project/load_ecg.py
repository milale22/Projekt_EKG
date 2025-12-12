import tempfile
import zipfile
from io import BytesIO
from os import listdir, path
import wfdb
import pandas as pd
import re
import numpy as np
import os


# Funktion, die Extension der Datei prüft
def check_format(filename:str):
    # holt die Dateiendung
    ext = os.path.splitext(filename)[1]
    # csv und zip -> erlaubt
    if ext in (".csv", ".zip"):
        return{
            ".csv": "csv", ".zip": "wfdb_zip"}[ext]

    # alles andere nicht erlaubt
    raise ValueError(f"Dateiformat {ext} wird nicht unterstützt. Nur .csv und .zip (mit .hea + .dat) sind erlaubt.")


# Funktion, die die EKG Datei lädt
def load_ecg_file(filename:str, content:bytes):
    # herausfinden, ob Datei csv oder zip ist
    kind = check_format(filename)

    # wenn csv, rufe die zugehörige csv Funktion auf
    if kind == "csv":
        return load_ecg_csv(filename)
    # wenn zip, rufe die zugehörige zip Funktion auf
    elif kind == "wfdb_zip":
        return load_ecg_wfdb_zip(content)



# Funktion, die EKG Daten aus csv lädt
def load_ecg_csv(path: str):

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if "\t" in first_line:
        sep = "\t"
    elif ";" in first_line:
        sep = ";"
    elif "," in first_line:
        sep = ","
    else:
        sep = r"\s+"


    df = pd.read_csv(path, sep=sep, engine="python")


    # Anzahl Zeilen und Spalten
    nrows, ncols = df.shape

    # Variablen, die in der Datei gesucht werden
    # Sampling Rate (Hz)
    fs   = None
    # Ableitung z.B Lead I
    lead = None
    # Einheit (µv)
    unit = None

    # alles in Text umwandeln und Leerzeichen entfernen
    as_str  = df.astype(str).applymap(lambda s: s.strip())
    # klein schreiben, Suche vereinfachen
    lower   = as_str.applymap(lambda s: s.lower())

    # Sample Rate / fs: überall suchen; Wert steht rechts daneben (gleiche Zeile, nächste Spalte)
    fs_keys = ("sample rate", "sampling frequency", "fs", "frequenz", "abtastrate", "frequency")
    # durch jede Zeile der csv gehen
    for i in range(nrows):
        # durch jede Spalte (bis zur vorletzten)
        for j in range(ncols - 1):
            cell = lower.iat[i, j]
            # wenn Zelle mit einem bekannten Schlüsselwort (in fs_keys) beginnt
            if any(cell.startswith(k) for k in fs_keys):
                # Zelle rechts daneben steht der Wert
                right = as_str.iat[i, j+1]
                # Zahl suchen
                m = re.search(r"\d+", right)
                # wenn Zahl gefunden, Sampling Rate übernehmen
                if m:
                    try:
                        fs = float(m.group(0))
                    except ValueError:
                        pass
                break
        if fs is not None:
            break

    # Lead + Unit: überall suchen; Wert steht recht daneben
    for i in range(nrows):
        for j in range(ncols - 1):
            cell = lower.iat[i, j]

            # rechts neben "lead" steht lead name
            if cell == "lead" or cell.startswith("lead "):
                val = as_str.iat[i, j+1]
                if val:
                    lead = val

            # rechts neben "unit" steht die Einheit
            if cell == "unit":
                val = as_str.iat[i, j+1]
                if val:
                    unit = val



    # Zeile mit µV/uv finden (egal welche Spalte)
    marker_row = None
    for i in range(nrows):
        # gibt es Spalte in der Zeile i in der uv vorkommt
        if any(("µv" in lower.iat[i, j]) or ("uv" in lower.iat[i, j]) for j in range(ncols)):
            # Zeilennummer merken
            marker_row = i
            # falls Unit noch nicht gefunden wurde
            if unit is None:
                # in der Zeile die Zelle mit uv finden und als Einheit nehmen
                for j in range(ncols):
                    if ("µv" in lower.iat[i, j]) or ("uv" in lower.iat[i, j]):
                        unit = as_str.iat[i, j]
                        break
            break


    # Datenblock bestimmen: alles ab marker_row+1 (rechte Spalten)
    # Fallback: erste Zeile, deren numerische Zellen >= 3 sind
    if marker_row is not None and marker_row + 1 < nrows:
        data_block = df.iloc[marker_row+1:, 1]
    else:
        # Fallback: erste Datenzeile finden
        start = 0
        for i in range(nrows):
            nums = pd.to_numeric(df.iloc[i, :], errors="coerce")
            # mind 3 numerische Zellen in der Zeile
            if np.isfinite(nums).sum() >= 3:
                start = i
                break
        data_block = df.iloc[start:, 1]

    # Defaults, falls etwas fehlt oder nicht gefunden wird
    if lead is None:
        lead = "Lead I"
    if fs is None:
        fs = 512.0
    if unit is None:
        unit = "µV"

    # EKG Spalte numerisch machen und NaN entfernen
    data_serie = pd.to_numeric(data_block, errors="coerce")

    signal = data_serie[data_serie.notna()].astype(float).to_numpy()

    return signal, fs, lead, unit


# Funktion um EKG Daten aus zip bestehend aus .hea und .dat zu laden
def load_ecg_wfdb_zip(content:bytes):
    # temporären Ordner erstellen
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Bytes der Zip laden und entpacken
        with zipfile.ZipFile(BytesIO(content)) as zip_ref:
            zip_ref.extractall(tmpdirname)

        # alle Dateien im Ordner
        names = listdir(tmpdirname)
        # Basisnamen ohne .hea (=Header) Endung
        hea_bases = {path.splitext(n)[0] for n in names if n.lower().endswith(".hea")}
        # Basisnamen ohne .dat (=Rohdaten) Endung
        dat_bases = {path.splitext(n)[0] for n in names if n.lower().endswith(".dat")}

        # Schnittmenge: Paare, die .hea UND .dat haben
        pairs = sorted(hea_bases & dat_bases)
        # wfdb besteht immer aus .hea und .dat
        if not pairs:
            raise ValueError("Kein .hea/.dat-Paar im ZIP gefunden.")

        # Pfad zum temp Ordner
        base = path.join(tmpdirname, pairs[0])

        # Record laden
        rec = wfdb.rdrecord(base)
        # nur die erste Ableitung nehmen (theoretisch noch prüfen ob erste Ableitung auch wirklich lead 1 ist)
        signal = rec.p_signal[:, 0]

        fs = rec.fs
        name = rec.sig_name
        units = rec.units

        lead_name= name[0]
        unit = units[0]


        return signal, fs, lead_name, unit


# Funktion, die Input in FHIR Format umwandelt
'''
def lead_to_fhir_observation(sig: np.ndarray, fs: float, lead_name: str, unit: str = "mV"):
    # wandelt ein Signal in eine FHIR-Observation um
    period_ms = 1000.0 / fs                    # Zeitintervall zwischen Samples (ms)
    #data_str = " ".join(f"{x:.6f}" for x in sig)  # als String
    data_str = " ".join(f"{x:.3f}" for x in sig)

    return {
        "resourceType": "Observation",
        "status": "final", # Pflichtfeld
        "code": {"text": f"ECG raw signal ({lead_name})"}, # Pflichtfeld
        # subject patient id
        # effectivedatetime zeit der aufnahme
        "valueSampledData": {
            "origin": {"value": 0, "unit": unit},
            "period": period_ms,
            "dimensions": 1,
            "data": data_str
        }
    }
'''

