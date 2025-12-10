# Entwicklung eines Webdienstes zur Klassifikation von EKG-Daten mittels Deep Learning

> Dieses README ist bewusst kurz gehalten und dient der schnellen Übersicht.

Dieses Projekt befasst sich mit der Entwicklung eines Webdienstes zur automatischen Klassifikation von EKG-Daten mithilfe von Deep Learning. Ziel ist es EKG-Signale automatisiert auszuwerten und eine zuverlässige Diagnoseunterstützung bereitzustellen.

---

## Projektbeschreibung
Grundlage dieses Projekt ist ein auf Kaggle veröffentlichtes Notebook, das auf den PTB-XL-Datensatz basiert: 
https://www.kaggle.com/code/likith012/ecgnet-ptb-xl

PTB-XL-Datensatz: https://physionet.org/content/ptb-xl/1.0.3/

Das Modell wurde weitgehend übernommen und ebenfalls mit PTB-XL trainiert. 
Nutzer können entweder über das *Skript* oder über den Webdienst *\docs* eine geeignete EKG-Datei hochladen. 

Der Webdienst kann letztendlich drei verschiedene Diagnosen ausgeben:

- **NORM** = normales EKG

- **CD** (Conduction Disturbance) = Leitungsstörung

- **STTC** = ST/T-Änderung

Die jeweilige Diagnose wird zusammen mit ihrer zugehörigen Wahrscheinlichkeit ausgegeben.

---

## Projektstruktur

Das Projekt besteht aus mehreren Modulen:

- **api.py**: enthält die API des Webdienstes

- **client.py**: Skript zum Senden einer EKG-Datei an den Webdienst

- **ecg_preprocess**: enthält alle Vorverarbeitungsfunktionen wie Resampling, Standardisierung, etc.

- **load_ecg.py**: lädt und prüft die EKG-Dateien auf Gültigkeit

- **model.py**: enthält wichtige Modellaspekte und predict Funktion

---

  ## Nutzung

  FÜr die Nutzung werden die Dateien aus den Ordnern **artifacts** und **ecg_project** benötigt. Die im Ordner **artifacts** enthaltenen Dateien werden beim Start geladen.

  ### Start des Webdienstes

  In das Verzeichnis wechseln, das den Ordner **ecg_project** enthält und folgenden Befehl ausführen:

  **uvicorn ecg_project.api:app --reload**

  
  Bei einem erfolgreichem Start erscheint z.B.:
  
  INFO:     Started server process [77104]
  
  INFO:     Waiting for application startup.
  
  INFO:     Application startup complete.
  
  INFO:     127.0.0.1:52366 - "GET / HTTP/1.1" 200 OK


  Anschließend ist der Webdienst erreichbar unter:

  http://127.0.0.1:8000

  Dort sollte folgende Meldung erscheinen: 

  {"message": "EKG-Klassifikations-API läuft"}


  Die interaktive API-Doku kann nun über folgenden Link aufgerufen werden:

  http://127.0.0.1:8000/docs

  Dort kann eine EKG-Datei über den POST-Endpunkt hochgeladen und direkt getestet werden.

  Alternativ kann parallel das Skript gestartet werden:

  **python client.py**  
  
  Dabei muss der Pfad zur EKG-Datei angegeben werden, um eine Klassifikation zu erhalten. 

  --- 
  ## Hinweis
  Projekt dient ausschließlich zu Demonstrations- und Forschungszwecken und ist **KEIN** medizinisch zugelassenes Diagnosesystem
  
