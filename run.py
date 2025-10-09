# import uvicorn
# from config import API_HOST, API_PORT

# if __name__ == "__main__":
#     uvicorn.run(
#         "app.main:app",
#         host=API_HOST,
#         port=API_PORT,
#         reload=True,
#         log_level="info"
#     )
# ===================================
# from pyngrok import ngrok
# import uvicorn
# from config import API_HOST, API_PORT

# if __name__ == "__main__":
#     public_url = ngrok.connect(8000)
#     print("Public URL:", public_url)

#     uvicorn.run(
#         "app.main:app", 
#         host=API_HOST, 
#         port=API_PORT, 
#         reload=True,
#         log_level="info"
#     )
from pyngrok import ngrok
import uvicorn
from config import API_HOST, API_PORT

if __name__ == "__main__":
    public_url = ngrok.connect(
        API_PORT, 
        "http", 
        domain="hailee-unrepresentational-ronnie.ngrok-free.dev"
    )
    print("Public URL:", public_url)

    uvicorn.run(
        "app.main:app", 
        host=API_HOST, 
        port=API_PORT, 
        reload=True,
        log_level="info"
    )
