from fastapi import APIRouter

from app.auth.router import router as auth
from app.components.router import router as components
from app.exports.router import router as exports
from app.health.router import router as health
from app.projects.router import router as projects
from app.tasks.router import router as tasks
from app.verification.router import router as verification

router = APIRouter()
for topic in (auth, components, health, verification, projects, tasks, exports):
    router.include_router(topic)
