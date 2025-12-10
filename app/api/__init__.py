from .quiz_routes import router as quiz_router
from .admin_routes import router as admin_router

__all__ = ["quiz_router", "admin_router"]