from fastapi.middleware.cors import CORSMiddleware

def add_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # React dev, tighten later
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


#later I have to replace * with the frontend domain
