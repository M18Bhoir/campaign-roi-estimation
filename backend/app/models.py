import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


# ============================================================
# Campaign Model
# ============================================================

class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )

    campaign_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    campaign_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    marketing_channel: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    target_audience: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    duration_days: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    budget: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    marketing_spend: Mapped[float | None]=      mapped_column(
    Float,
    nullable=True,
    )

    impressions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    clicks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    conversions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    engagement_rate: Mapped[float] = mapped_column(
        Float,
        nullable=True,
    )

    ctr: Mapped[float] = mapped_column(
        Float,
        nullable=True,
    )

    conversion_rate: Mapped[float] = mapped_column(
        Float,
        nullable=True,
    )

    campaign_start_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    campaign_end_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="campaign",
        cascade="all, delete-orphan",
    )


# ============================================================
# Prediction Model
# ============================================================

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )

    campaign_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )

    predicted_roi: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    predicted_revenue: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    predicted_profit: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    roi_category: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    prediction_latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    campaign: Mapped["Campaign"] = relationship(
        back_populates="predictions",
    )


# ============================================================
# Model Metadata
# ============================================================

class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )

    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    algorithm: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    test_r2: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    cv_r2: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mae: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    rmse: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    training_samples: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    feature_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )