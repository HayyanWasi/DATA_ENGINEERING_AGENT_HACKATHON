import uuid

from pydantic import BaseModel, Field


class ImportResponse(BaseModel):
    dataset_id: uuid.UUID
    row_count: int
    columns: list[str]


class DatabaseConnectionConfig(BaseModel):
    host: str
    port: int
    username: str
    password: str
    database: str
    db_type: str = Field(..., pattern=r"^(postgresql|mysql)$")
    table_name: str
