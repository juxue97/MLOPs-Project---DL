from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.router import routerPred, routerTrain

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    try:
        return {"Health_Check_Status": "OK", "Connection": "Alive"}

    except Exception as e:
        raise Exception(f"Error starting http server : {e}")

app.include_router(routerPred)
app.include_router(routerTrain)
