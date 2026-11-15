from fastapi import APIRouter

from app.api.v1.analyses import router as analyses_router
from app.api.v1.auth import router as auth_router
from app.api.v1.system import router as system_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(analyses_router)
api_router.include_router(system_router)
