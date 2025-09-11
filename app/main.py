from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .routers import qa, trials, eligibility


app = FastAPI(title="TrialWhisperer", version="0.1.0")


app.include_router(qa.router, prefix="/ask", tags=["qa"])
app.include_router(trials.router, prefix="/trial", tags=["trials"])
app.include_router(eligibility.router, prefix="/check-eligibility", tags=["eligibility"])


# Serve minimal UI
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")