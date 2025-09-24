# Dockerfile pour Railway - Image Ubuntu stable
FROM ubuntu:22.04

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV STARTUP_DELAY_SECONDS=30
ENV SOUND_ALERTS=false

# Installation Python et dépendances système
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    gcc \
    g++ \
    curl \
    wget \
    build-essential \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de travail
WORKDIR /app

# Copie des fichiers requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Création des répertoires nécessaires
RUN mkdir -p models cache logs

# Port d'exposition (pour Railway)
EXPOSE 8080

# Commande de démarrage
CMD ["python", "server_bot.py"]
