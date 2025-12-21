from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import user, superadmin, org_admin, permissions

app = FastAPI(
    title="Shreembo.com Backend API",
    description="API for user configuration and superadmin organization management.",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user.router, prefix="/user", tags=["User"])
app.include_router(superadmin.router, prefix="/superadmin", tags=["Superadmin"])
app.include_router(org_admin.router, prefix="/org-admin", tags=["OrgAdmin"])
app.include_router(permissions.router, prefix="/permissions", tags=["Permissions"])

@app.get("/")
async def root():
    return {"message": "Shreembo.com Backend API is running!"}
