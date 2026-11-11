
"""FastAPI endpoints for model promotion and archival."""

from fastapi import APIRouter, HTTPException

import src.registry.model_registry as registry


router = APIRouter(
    prefix="/models",
    tags=["Model Lifecycle"],
)

REGISTRY_DIR_OVERRIDE = None


def find_model_record(model_name, version):
    """Find a registered model."""

    records = registry.list_registered_models(
        registry_dir=REGISTRY_DIR_OVERRIDE
    )

    matches = [
        record
        for record in records
        if (
            record.get("model_name", "").lower()
            == model_name.lower()
            and str(record.get("version")) == str(version)
        )
    ]

    if not matches:
        raise HTTPException(
            status_code=404,
            detail="Model version not found.",
        )

    if len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail="Multiple matching model versions found.",
        )

    return matches[0]


@router.post("/{model_name}/{version}/promote")
def promote_model_endpoint(model_name: str, version: str):

    try:
        find_model_record(model_name, version)

        registry.promote_model(
            model_name=model_name,
            version=version,
            registry_dir=REGISTRY_DIR_OVERRIDE,
        )

        updated_record = find_model_record(
            model_name,
            version,
        )

        return {
            "status": "success",
            "message": "Model promoted successfully.",
            "model": updated_record,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@router.post("/{model_name}/{version}/archive")
def archive_model_endpoint(model_name: str, version: str):

    try:
        find_model_record(model_name, version)

        registry.archive_model(
            model_name=model_name,
            version=version,
            registry_dir=REGISTRY_DIR_OVERRIDE,
        )

        updated_record = find_model_record(
            model_name,
            version,
        )

        return {
            "status": "success",
            "message": "Model archived successfully.",
            "model": updated_record,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
