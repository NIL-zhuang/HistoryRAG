import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag import __version__
from rag.server.api_server.chat_routes import chat_router
from rag.server.api_server.kb_routers import kb_router


def create_app():
    app = FastAPI(title="RAG API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(kb_router)
    return app


def run_api(host, port, reload=False, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            ssl_keyfile=kwargs.get("ssl_keyfile"),
            ssl_certfile=kwargs.get("ssl_certfile"),
        )
    else:
        uvicorn.run(app, host=host, port=port, reload=reload)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19198)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    app = create_app()
    run_api(**vars(parse_args()))
