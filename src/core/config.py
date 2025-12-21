import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    
    # Validation
    if not SUPABASE_URL:
        print("Warning: SUPABASE_URL not set")
    if not SUPABASE_KEY:
        print("Warning: SUPABASE_KEY not set")

settings = Settings()
