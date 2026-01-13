import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

@dataclass
class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
    ANALYSIS_ENABLED: bool = os.getenv("ANALYSIS_ENABLED", "true").lower() == "true"
    ANALYSIS_SAMPLE_SIZE: int = int(os.getenv("ANALYSIS_SAMPLE_SIZE", "4000"))  # number of characters from tail of buffer
    ANALYSIS_DEBOUNCE_MS: int = int(os.getenv("ANALYSIS_DEBOUNCE_MS", "1200"))

    # Optional: toggle verbose logging for debugging (do not log secrets)
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()
