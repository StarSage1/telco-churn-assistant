from __future__ import annotations

from typing import Union

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

try:
    from .chatbot import ChatService, ChatbotUnavailable, ollama_status
    from .bulk import process_bulk_file_assisted
    from .inference import predict_churn
    from .reports import create_customer_report
    from .schemas import ChatRequest, ChatResponse, CustomerInput, PredictionResponse
except ImportError:  # Support `uvicorn Api:app` when launched inside src/.
    from chatbot import ChatService, ChatbotUnavailable, ollama_status
    from bulk import process_bulk_file_assisted
    from inference import predict_churn
    from reports import create_customer_report
    from schemas import ChatRequest, ChatResponse, CustomerInput, PredictionResponse


app = FastAPI(
    title="Telco Churn Assistant API",
    description=(
        "Local conversational assistant and hybrid Logistic Regression + "
        "CatBoost churn predictor."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=[
        "Content-Disposition", "X-Total-Rows", "X-Scored-Rows",
        "X-Invalid-Rows", "X-Churn-Rate",
    ],
)

chat_service = ChatService()


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "service": "telco-churn-assistant",
        "ollama": await ollama_status(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_customer_churn(customer: CustomerInput) -> PredictionResponse:
    try:
        result = predict_churn(customer.model_dump())
        result.pop("risk_level", None)
        return PredictionResponse.model_validate(result)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@app.post("/report")
def generate_customer_report(customer: CustomerInput) -> Response:
    try:
        result = predict_churn(customer.model_dump())
        content = create_customer_report(customer.model_dump(), result)
        return Response(
            content=content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    'attachment; filename="churnsignal-customer-assessment.pdf"'
                )
            },
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/bulk-predict")
async def bulk_predict(file: UploadFile = File(...)) -> Response:
    filename = file.filename or "upload"
    try:
        content = await file.read()
        workbook, summary = await process_bulk_file_assisted(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=workbook,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                'attachment; filename="churnsignal-assessment-package.zip"'
            ),
            "X-Total-Rows": str(summary["total_rows"]),
            "X-Scored-Rows": str(summary["scored_rows"]),
            "X-Invalid-Rows": str(summary["invalid_rows"]),
            "X-Churn-Rate": f'{summary["churn_rate"]:.6f}',
        },
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await chat_service.process(request.session_id, request.message.strip())
    except ChatbotUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"{exc} Confirm Ollama is running and the configured model is installed."
            ),
        ) from exc


@app.delete("/chat/{session_id}")
def reset_chat(session_id: str) -> dict[str, Union[bool, str]]:
    removed = chat_service.reset(session_id)
    return {"session_id": session_id, "reset": removed}
