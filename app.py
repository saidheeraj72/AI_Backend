from __future__ import annotations

import os

import uvicorn

from src.main import app as fastapi_app


app = fastapi_app


def main() -> None:
    """Run the FastAPI application defined in src.main."""
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
