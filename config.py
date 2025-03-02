import os

class Config:
    # General settings
    APP_NAME = "AadhyaAI"
    VERSION = "1.0"
    DEBUG_MODE = True

    # File paths
    LOGS_DIR = os.path.join(os.getcwd(), "logs")
    DATA_DIR = os.path.join(os.getcwd(), "data")
    MODELS_DIR = os.path.join(os.getcwd(), "models")
    
    # Speech settings
    SPEECH_LANGUAGE = "en-US"
    VOICE_TYPE = "female"
    
    # AI Modules
    ENABLE_VISION = True
    ENABLE_MEMORY = True
    ENABLE_EMOTION_DETECTION = True
    ENABLE_HOLOGRAPHY = True
    ENABLE_REAL_WORLD_AI = True
    ENABLE_QUANTUM_AI = False  # Experimental
    
    # API keys (if required)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    # Security settings
    ENABLE_ENCRYPTION = True
    ENCRYPTION_KEY = os.getenv("AADHYA_ENCRYPTION_KEY", "default_secret_key")

# Ensure directories exist
for directory in [Config.LOGS_DIR, Config.DATA_DIR, Config.MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    print(f"{Config.APP_NAME} v{Config.VERSION} loaded successfully.")
