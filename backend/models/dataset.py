import uuid
from datetime import datetime

from sqlalchemy import Integer, String, DateTime, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class UserDataset(Base):
    __tablename__ = "user_datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    dataset_name: Mapped[str] = mapped_column(String, nullable=False)
    raw_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    columns: Mapped[list] = mapped_column(ARRAY(String), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
