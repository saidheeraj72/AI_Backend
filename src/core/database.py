from supabase import create_client, Client
from src.core.config import settings

# Initialize Supabase client
# Ensure URL and Key are available, otherwise this might fail at runtime
try:
    supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
except Exception as e:
    print(f"Failed to initialize Supabase client: {e}")
    supabase = None
