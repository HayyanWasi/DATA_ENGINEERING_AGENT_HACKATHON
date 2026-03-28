import asyncio
import io
import json
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

import httpx
import pandas as pd
from fastapi import HTTPException, UploadFile
from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.dataset import UserDataset
from schemas.dataset import DatabaseConnectionConfig

MAX_ROWS = 10_000


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def _parse_csv(contents: bytes) -> list[dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(contents), nrows=MAX_ROWS)
    # to_json handles NaN/NaT → null; json.loads converts back to Python dicts
    return json.loads(df.to_json(orient="records"))


def _parse_excel(contents: bytes) -> list[dict[str, Any]]:
    df = pd.read_excel(io.BytesIO(contents), nrows=MAX_ROWS)
    return json.loads(df.to_json(orient="records"))


async def parse_file(file: UploadFile) -> list[dict[str, Any]]:
    """Parse an uploaded CSV or Excel file into a list of dicts."""
    filename = (file.filename or "").lower()
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if filename.endswith(".csv"):
        return await asyncio.to_thread(_parse_csv, contents)
    elif filename.endswith((".xls", ".xlsx")):
        return await asyncio.to_thread(_parse_excel, contents)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Accepted: .csv, .xls, .xlsx",
        )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_input(raw: str) -> list[dict[str, Any]]:
    """Parse a raw JSON string into a list of dicts."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if isinstance(data, list):
        if not data:
            raise HTTPException(status_code=400, detail="JSON array is empty")
        if not all(isinstance(item, dict) for item in data):
            raise HTTPException(
                status_code=400, detail="JSON array must contain only objects"
            )
        return data[:MAX_ROWS]

    if isinstance(data, dict):
        if not data:
            raise HTTPException(status_code=400, detail="JSON object is empty")
        # Column-oriented format: {"col1": [v1, v2], "col2": [v3, v4]}
        if all(isinstance(v, list) for v in data.values()):
            df = pd.DataFrame(data)
            return json.loads(df.head(MAX_ROWS).to_json(orient="records"))
        # Single record wrapped in a list
        return [data]

    raise HTTPException(status_code=400, detail="JSON must be an array or object")


# ---------------------------------------------------------------------------
# URL / API fetching
# ---------------------------------------------------------------------------

async def fetch_from_url(url: str) -> list[dict[str, Any]]:
    """Fetch data from an external URL (API returning JSON, or a CSV/Excel file)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=400, detail=f"URL returned error: {e.response.status_code}"
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    content_type = response.headers.get("content-type", "")
    path = parsed.path.lower()

    # Try JSON first (most APIs return JSON)
    if "json" in content_type or path.endswith(".json"):
        return parse_json_input(response.text)

    # CSV file
    if "csv" in content_type or path.endswith(".csv"):
        return await asyncio.to_thread(_parse_csv, response.content)

    # Excel file
    if path.endswith((".xls", ".xlsx")) or "spreadsheet" in content_type:
        return await asyncio.to_thread(_parse_excel, response.content)

    # Default: try JSON, then CSV
    try:
        return parse_json_input(response.text)
    except HTTPException:
        pass

    try:
        return await asyncio.to_thread(_parse_csv, response.content)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not parse response as JSON or CSV. "
            "Ensure the URL returns JSON, CSV, or Excel data.",
        )


# ---------------------------------------------------------------------------
# External database fetching
# ---------------------------------------------------------------------------

def _fetch_external_data(config: DatabaseConnectionConfig) -> list[dict[str, Any]]:
    """Connect to an external DB, read up to MAX_ROWS, return list of dicts."""
    if config.db_type == "postgresql":
        url = (
            f"postgresql+psycopg2://{config.username}:{config.password}"
            f"@{config.host}:{config.port}/{config.database}"
        )
    else:  # mysql
        url = (
            f"mysql+pymysql://{config.username}:{config.password}"
            f"@{config.host}:{config.port}/{config.database}"
        )

    engine = create_engine(url, connect_args={"connect_timeout": 10})
    try:
        # Verify the table exists (prevents SQL injection via reflected metadata)
        insp = inspect(engine)
        tables = insp.get_table_names()
        if config.table_name not in tables:
            raise HTTPException(
                status_code=400,
                detail=f"Table '{config.table_name}' not found. "
                f"Available tables: {tables[:20]}",
            )

        metadata = MetaData()
        table = Table(config.table_name, metadata, autoload_with=engine)

        with engine.connect() as conn:
            result = conn.execute(select(table).limit(MAX_ROWS))
            records = [dict(row._mapping) for row in result]

        if not records:
            raise HTTPException(
                status_code=400, detail="External table returned no data"
            )
        return records

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Database connection failed: {e}"
        )
    finally:
        engine.dispose()


async def fetch_from_external_db(
    config: DatabaseConnectionConfig,
) -> list[dict[str, Any]]:
    """Async wrapper around the synchronous external DB fetch."""
    return await asyncio.to_thread(_fetch_external_data, config)


# ---------------------------------------------------------------------------
# Processing & storage
# ---------------------------------------------------------------------------

def _serialize_value(v: Any) -> Any:
    """Convert a value to a JSON-safe type."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v)


def process_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Normalize records to JSON-safe values and extract column metadata."""
    if not records:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    records = records[:MAX_ROWS]

    # Ensure every value is JSON-serializable
    clean: list[dict[str, Any]] = []
    for row in records:
        clean.append({k: _serialize_value(v) for k, v in row.items()})

    # Extract unique column names preserving first-seen order
    seen: set[str] = set()
    columns: list[str] = []
    for row in clean:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    return clean, columns, len(clean)


async def store_dataset(
    db: AsyncSession,
    user_id: str,
    dataset_name: str,
    records: list[dict[str, Any]],
    columns: list[str],
    row_count: int,
) -> uuid.UUID:
    """Persist the processed dataset into the user_datasets table."""
    dataset = UserDataset(
        user_id=user_id,
        dataset_name=dataset_name,
        raw_data=records,
        columns=columns,
        row_count=row_count,
    )
    db.add(dataset)
    await db.flush()
    return dataset.id
