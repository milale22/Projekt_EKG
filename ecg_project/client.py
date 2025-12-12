import requests
import os

print("Herzlich Willkommen zum EKG-Diagnose-Client!")
print("Hinweis: Dieses Programm ersetzt keine ärztliche Diagnose.")
print("Bitte eine EKG-Datei angeben (.zip oder .csv)")

# Nutzer gibt Datei an
file_name = input("Name/Pfad der Datei angeben: ")

# Prüfen ob Datei existiert
if not os.path.exists(file_name):
    print("Datei nicht gefunden!")
    exit(1)

# URL des Diagnose Endpunkts der FastAPI
url = "http://127.0.0.1:8000/diagnosis"


try:
    # Öffnen der Datei im Binärmodus und an API senden
    with open(file_name, "rb") as file:
        files = {"file": file}
        response = requests.post(url, files=files, timeout=10)

# Fehler: Server läuft nicht
except requests.exceptions.ConnectionError:
    print("Keine Verbindung zum Server gefunden!")
    exit(1)

# Fehler: Server braucht zu lange
except requests.exceptions.Timeout:
    print("Server antwortet nicht (Timeout).")
    exit(1)


# Prüfen ob Server Antwort gültig ist
if response.status_code != 200:
    print("Server Fehler: Statuscode {response.status_code}")
    print(response.text)
    exit(1)


# Prüfen ob JSON zurückgegeben wird
try:
    data = response.json()
    print("Antwort vom Server: ", data)
    print("NORM steht für normales EKG")
    print("CD steht für Leitungsstörung - Bitte an Arzt wenden")
    print("STTC steht für ST/T-Änderung - Bitte an Arzt wenden")

except ValueError:
    print("Server hat kein JSON zurückgegeben")
    exit(1)



