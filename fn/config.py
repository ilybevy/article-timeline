import os
from dotenv import load_dotenv

load_dotenv()


# =========================
# XAI / GROK CONFIG
# =========================
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_API_URL = os.getenv(
    "XAI_API_URL",
    "https://api.x.ai/v1/chat/completions"
)

GROK_MODEL = os.getenv(
    "GROK_MODEL",
    "grok-4-1-fast-reasoning"
)

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))


# =========================
# LLM PARAMETERS
# =========================
MAX_TOKENS_LABEL = 32
TEMPERATURE_LABEL = 0.3


# =========================
# PERIOD MAKER SETTINGS
# =========================
TOP_K_KEYWORDS = 10