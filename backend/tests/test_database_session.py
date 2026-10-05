from sqlalchemy import text

from backend.app.database import SessionLocal


def test_database_session():
    db = SessionLocal()

    try:
        result = db.execute(
            text("SELECT current_database()")
        )

        database_name = result.scalar()

        assert database_name == "campaign_roi_db"

    finally:
        db.close()