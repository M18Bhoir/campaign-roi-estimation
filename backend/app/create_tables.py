from backend.app.database import Base, engine

# Import models so SQLAlchemy registers them with Base.
from backend.app.models import Campaign, Prediction, ModelMetadata


def create_tables():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully.")