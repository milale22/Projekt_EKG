import requests
print("Herzlich Willkommen zum EKG-Diagnose-Client!")
# EKG Daten in Form von zip(.hea und .dat), json oder csv angeben print()

file_name = input("Name/Pfad der Datei angeben: ")

url = "http://127.0.0.1:8000/diagnosis"

files = {'file': open(file_name, 'rb')}

response = requests.post(url, files=files)

print("Antwort vom Server: ", response.json())