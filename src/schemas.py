from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    Senior_Citizen: Literal[0, 1]
    Is_Married: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    Phone_Service: Literal["Yes", "No"]
    Dual: Literal["Yes", "No", "No phone service"]
    Internet_Service: Literal["DSL", "Fiber optic", "No"]
    Online_Security: Literal["Yes", "No", "No internet service"]
    Online_Backup: Literal["Yes", "No", "No internet service"]
    Device_Protection: Literal["Yes", "No", "No internet service"]
    Tech_Support: Literal["Yes", "No", "No internet service"]
    Streaming_TV: Literal["Yes", "No", "No internet service"]
    Streaming_Movies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    Paperless_Billing: Literal["Yes", "No"]
    Payment_Method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    Monthly_Charges: float = Field(ge=0, allow_inf_nan=False)
    Total_Charges: float = Field(ge=0, le=100_000, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_service_combinations(self) -> "CustomerInput":
        if self.Phone_Service == "No" and self.Dual != "No phone service":
            raise ValueError("Dual must be 'No phone service' when phone service is No.")
        if self.Phone_Service == "Yes" and self.Dual == "No phone service":
            raise ValueError("Dual must be Yes or No when phone service is active.")

        internet_features = (
            self.Online_Security,
            self.Online_Backup,
            self.Device_Protection,
            self.Tech_Support,
            self.Streaming_TV,
            self.Streaming_Movies,
        )
        if self.Internet_Service == "No" and any(
            value != "No internet service" for value in internet_features
        ):
            raise ValueError(
                "Internet-dependent services must be 'No internet service' when internet is No."
            )
        if self.Internet_Service != "No" and any(
            value == "No internet service" for value in internet_features
        ):
            raise ValueError(
                "Internet-dependent services must be Yes or No when internet is active."
            )
        if self.tenure == 0 and self.Total_Charges > self.Monthly_Charges:
            raise ValueError(
                "Total charges look inconsistent for a customer with zero months of tenure."
            )
        return self


CustomerPatch = create_model(
    "CustomerPatch",
    __config__=ConfigDict(extra="forbid"),
    **{
        name: (Optional[field.annotation], None)
        for name, field in CustomerInput.model_fields.items()
    },
)


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    churn_probability: float
    logistic_probability: float
    catboost_probability: float
    threshold: float
    model_weights: dict[str, float]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=8, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=2_000)


class Explanation(BaseModel):
    risk_level: Literal["Low", "Medium", "High"]
    summary: str
    profile_signals: list[str]
    recommended_action: str
    note: str


class ChatResponse(BaseModel):
    session_id: str
    status: Literal["collecting", "complete", "clarification"]
    message: str
    collected_fields: dict[str, Any]
    missing_fields: list[str]
    requested_fields: list[str]
    completed_count: int
    required_count: int
    prediction: Optional[PredictionResponse] = None
    explanation: Optional[Explanation] = None
