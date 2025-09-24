# Dockerfile pour conteneurisation
FROM python:3.11-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de travail
WORKDIR /app

# Copie des fichiers requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Création des répertoires nécessaires
RUN mkdir -p models cache logs

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV STARTUP_DELAY_SECONDS=30
ENV SOUND_ALERTS=false

# Port d'exposition (pour Railway)
EXPOSE 8080

# Commande de démarrage
CMD ["python", "tendance_globale.py"]