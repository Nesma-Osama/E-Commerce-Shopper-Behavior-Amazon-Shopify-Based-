import io
import json
import os
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile

try:
    from modelservice import service  # the Docker image copies this beside server.py.
except ImportError:
    from .modelservice import service


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()

    port = os.environ.get("PORT", "8000")
    internal_base = f"http://0.0.0.0:{port}"
    external_base = f"http://localhost:{port}"

    print("\n=== API Access ===")
    print(f"Internal: {internal_base}")
    print(f"External: {external_base}")
    print(f"Docs: {external_base}/docs")
    print("\n=== Endpoints ===")

    available_routes = []
    for route in app.routes:
        methods = sorted(m for m in (route.methods or set()) if m not in {"HEAD", "OPTIONS"})
        if not methods:
            continue
        for method in methods:
            endpoint_url = f"{external_base}{route.path}"
            available_routes.append(f"{method} {route.path}")
            print(f"{method} {endpoint_url}")

    print("\nAvailable routes:")
    for route_info in sorted(set(available_routes)):
        print(f"- {route_info}")

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _prediction_error_status(exc: ValueError) -> int:
    detail = str(exc)
    if detail.startswith("Unknown ") or "is not available" in detail:
        return 404
    return 400


def _classifier_response(model_name: str, data, meta):
    preds = service.predict_classifier(model_name, data)
    return {
        "model": model_name,
        "task": "classifier",
        "predictions": preds.tolist(),
        **meta,
    }


def _regressor_response(model_name: str, data, meta):
    preds = service.predict_regressor(model_name, data)
    return {
        "model": model_name,
        "task": "regressor",
        "predictions": preds.tolist(),
        **meta,
    }


async def _parse_uploaded_file(file: UploadFile):
    filename = file.filename or "uploaded_file"
    lower_name = filename.lower()
    content = await file.read()

    if lower_name.endswith(".csv"):
        try:
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        return df, {
            "input_type": "csv",
            "filename": filename,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        }

    if lower_name.endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(content), allow_pickle=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid NPY file: {exc}")

        arr = np.asarray(arr)
        if arr.size == 0:
            raise HTTPException(status_code=400, detail="NPY file is empty")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return arr.tolist(), {
            "input_type": "npy",
            "filename": filename,
            "rows": int(arr.shape[0]),
        }

    raise HTTPException(status_code=400, detail="file must be a .csv or .npy")


async def _parse_prediction_input(request: Request):
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        if isinstance(payload, dict):
            if "input" not in payload:
                raise HTTPException(status_code=400, detail="No input provided")
            data = payload["input"]
        elif isinstance(payload, list):
            data = payload
        else:
            raise HTTPException(status_code=400, detail="Invalid JSON format for input")

        if data is None:
            raise HTTPException(status_code=400, detail="No input provided")

        return data, {"input_type": "json"}

    if "multipart/form-data" in content_type:
        form = await request.form()
        file_obj = form.get("file")

        # Accept file uploads even when runtime uses a different UploadFile class.
        if file_obj is not None and hasattr(file_obj, "filename") and hasattr(file_obj, "read"):
            return await _parse_uploaded_file(file_obj)

        # Fallback: accept the first uploaded file from any form key.
        for value in form.values():
            if hasattr(value, "filename") and hasattr(value, "read"):
                return await _parse_uploaded_file(value)

        if file_obj is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid 'file' field. Send multipart/form-data with key 'file' "
                    "as a real uploaded file (.csv or .npy), not a text path."
                ),
            )

        raw_input = form.get("input")
        if raw_input:
            try:
                data = json.loads(raw_input)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON string in 'input' form field")

            if data is None:
                raise HTTPException(status_code=400, detail="No input provided")

            return data, {"input_type": "json_form"}

        raise HTTPException(status_code=400, detail="No input provided. Send JSON body or multipart file field named 'file'.")

    raise HTTPException(
        status_code=415,
        detail="Unsupported content type. Use application/json or multipart/form-data.",
    )



@app.post("/predict")
async def predict_default_classifier(request: Request):
    data, meta = await _parse_prediction_input(request)

    try:
        return _classifier_response("knn_model", data, meta)
    except ValueError as exc:
        raise HTTPException(status_code=_prediction_error_status(exc), detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="internal server error") from exc


@app.post("/predict/csv")
async def predict_csv(request: Request):
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="Use multipart/form-data with a .csv or .npy file upload.",
        )

    data, meta = await _parse_prediction_input(request)
    if meta.get("input_type") not in {"csv", "npy"}:
        raise HTTPException(
            status_code=400,
            detail="Send a multipart file field named 'file' containing a .csv or .npy file.",
        )

    try:
        return _classifier_response("knn_model", data, meta)
    except ValueError as exc:
        raise HTTPException(status_code=_prediction_error_status(exc), detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="internal server error") from exc


@app.post("/predict/classifier/{model_name}")
async def predict_classifier(model_name: str, request: Request):
    data, meta = await _parse_prediction_input(request)

    try:
        return _classifier_response(model_name, data, meta)
    except ValueError as exc:
        raise HTTPException(status_code=_prediction_error_status(exc), detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="internal server error") from exc


@app.post("/predict/regressor/{model_name}")
async def predict_regressor(model_name: str, request: Request):
    data, meta = await _parse_prediction_input(request)

    try:
        return _regressor_response(model_name, data, meta)
    except ValueError as exc:
        raise HTTPException(status_code=_prediction_error_status(exc), detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="internal server error") from exc
