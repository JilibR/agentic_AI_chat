import os
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ No MISTRAL KEY found .env")