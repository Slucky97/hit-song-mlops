#On part d'une petite image python
FROM python:3.14-slim

#On définit un répertoire de travail dans le conteneur
WORKDIR /app

##On copie d'abord les dépendances
COPY requirements.txt .

#On installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

#On copie ensuite le reste du projet
COPY app.py .
COPY model.joblib .

#On expose le port sur lequel l'API écoute
EXPOSE 8000

#La commande qui lance l'API au démarrage du conteneur
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]