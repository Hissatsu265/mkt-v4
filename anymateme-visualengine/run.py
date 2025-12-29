from pyngrok import ngrok
import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    public_url = ngrok.connect(
        addr=settings.PORT,
        proto="http",
        domain="wavier-unstoppably-gracia.ngrok-free.dev"
    )
    print(f"🌍 Public URL: {public_url}")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
# ===============================================
# import uvicorn
# from app.core.config import settings

# if __name__ == "__main__":
#     uvicorn.run(
#         "app.main:app",
#         host=settings.HOST,
#         port=settings.PORT,
#         reload=settings.RELOAD,
#         log_level=settings.LOG_LEVEL.lower()
#     )