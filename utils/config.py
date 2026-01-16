import os
from dotenv import load_dotenv
import tomllib

# Load var env from .env
if load_dotenv():
    MISTRAL_API_KEY = os.getenv("MISTRAL_KEY")
    if not MISTRAL_API_KEY:
        print("⚠️ No MISTRAL KEY found .env")
else:
    with open('.streamlit/secrets.toml', 'rb') as f:
            config = tomllib.load(f)
    MISTRAL_API_KEY = config['MISTRAL_KEY']
