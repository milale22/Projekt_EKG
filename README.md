# Entwicklung eines Webdienstes zur Klassifikation von EKG-Daten mittels Deep Learning

Dieses Projekt befasst sich mit der Entwicklung eines Webdienstes zur automatischen Klassifikation von EKG-Daten mithilfe von DL. Ziel ist es EKG-Signale automatisiert auszuwerten und eine zuverlässige Diagnoseunterstützung bereitzustellen.

## Projektbeschreibung
Grundlage des Projekt ist ein auf Kaggle veröffentlichtes Notebook, das auf den PTB-XL-Datensatz basiert: 
https://www.kaggle.com/code/likith012/ecgnet-ptb-xl?
Das Modell wurde weitgehend übernommen und ebenfalls mit PTB-XL trainiert. Nutzer können über das Skript oder über den Webdienst \docs ihre geeignet EKG-Datei hochladen. Der Webdienst kann letztendlich drei verschiedene Diganosen ausgeben:

- **NORM** = normales EKG

- **CD** = Leitungsstörung

- **STTC** = ST/T-Änderung

Die jeweilige Diagnose wird zusammen mit ihrer zugehörigen Wahrscheinlichkeit ausgegeben.

## Bestandteile

Das Projekt besteht aus mehreren Modulen:

- **api.py**:

- **client.py**: 

- **ecg_preprocess**:

- **load_ecg.py**:

- **client.py**:
