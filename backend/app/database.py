import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# Database URL
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")


if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set."
    )


# ============================================================
# SQLAlchemy Engine
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# ============================================================
# Session Factory
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# ============================================================
# Declarative Base
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# Database Session Dependency
# ============================================================

def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()