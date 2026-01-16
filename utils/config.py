import os
from dotenv import load_dotenv

# Load var env from .env
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ No MISTRAL KEY found .env")