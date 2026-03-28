from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from schemas.dataset import DatabaseConnectionConfig, ImportResponse
from services.data_import import (
    fetch_from_external_db,
    fetch_from_url,
    parse_file,
    parse_json_input,
    process_records,
    store_dataset,
)

router = APIRouter()


@router.post("/api/import-data", response_model=ImportResponse)
async def import_data(
    db: AsyncSession = Depends(get_db),
    user_id: str = Form(...),
    dataset_name: str = Form(...),
    import_type: str = Form(..., description="One of: file, json, database, url"),
    # --- File upload (required when import_type == "file") ---
    file: Optional[UploadFile] = File(None),
    # --- JSON input (required when import_type == "json") ---
    json_data: Optional[str] = Form(None),
    # --- URL import (required when import_type == "url") ---
    api_url: Optional[str] = Form(None),
    # --- External DB fields (required when import_type == "database") ---
    db_host: Optional[str] = Form(None),
    db_port: Optional[int] = Form(None),
    db_username: Optional[str] = Form(None),
    db_password: Optional[str] = Form(None),
    db_name: Optional[str] = Form(None),
    db_type: Optional[str] = Form(None),
    db_table: Optional[str] = Form(None),
):
    """
    Import data from one of four sources:
      - **file**: Upload a CSV or Excel file
      - **json**: Send a JSON array of objects or a column-oriented dict
      - **url**: Fetch data from an external API or file URL (JSON/CSV/Excel)
      - **database**: Connect to an external PostgreSQL or MySQL database
    """
    # ── 1. Parse records from the selected source ─────────────────────────
    if import_type == "file":
        if not file or not file.filename:
            raise HTTPException(
                status_code=400, detail="File is required for file import"
            )
        records = await parse_file(file)

    elif import_type == "json":
        if not json_data:
            raise HTTPException(
                status_code=400, detail="json_data is required for JSON import"
            )
        records = parse_json_input(json_data)

    elif import_type == "url":
        if not api_url:
            raise HTTPException(
                status_code=400, detail="api_url is required for URL import"
            )
        records = await fetch_from_url(api_url)

    elif import_type == "database":
        missing = [
            name
            for name, val in [
                ("db_host", db_host),
                ("db_port", db_port),
                ("db_username", db_username),
                ("db_password", db_password),
                ("db_name", db_name),
                ("db_type", db_type),
                ("db_table", db_table),
            ]
            if val is None
        ]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required database fields: {', '.join(missing)}",
            )
        if db_type not in ("postgresql", "mysql"):
            raise HTTPException(
                status_code=400,
                detail="db_type must be 'postgresql' or 'mysql'",
            )

        config = DatabaseConnectionConfig(
            host=db_host,       # type: ignore[arg-type]
            port=db_port,       # type: ignore[arg-type]
            username=db_username,  # type: ignore[arg-type]
            password=db_password,  # type: ignore[arg-type]
            database=db_name,      # type: ignore[arg-type]
            db_type=db_type,       # type: ignore[arg-type]
            table_name=db_table,   # type: ignore[arg-type]
        )
        records = await fetch_from_external_db(config)

    else:
        raise HTTPException(
            status_code=400,
            detail="import_type must be one of: file, json, url, database",
        )

    # ── 2. Process and store ──────────────────────────────────────────────
    records, columns, row_count = process_records(records)
    dataset_id = await store_dataset(
        db, user_id, dataset_name, records, columns, row_count
    )

    return ImportResponse(
        dataset_id=dataset_id,
        row_count=row_count,
        columns=columns,
    )
