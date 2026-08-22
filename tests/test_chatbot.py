import asyncio

from src.chatbot import (
    ChatService,
    _apply_relationship_rules,
    extract_customer_patch,
    extract_explicit_values,
)


COMPLETE_CUSTOMER = {
    "gender": "Female",
    "Senior_Citizen": 0,
    "Is_Married": "No",
    "Dependents": "No",
    "tenure": 3,
    "Phone_Service": "Yes",
    "Dual": "No",
    "Internet_Service": "Fiber optic",
    "Online_Security": "No",
    "Online_Backup": "No",
    "Device_Protection": "No",
    "Tech_Support": "No",
    "Streaming_TV": "Yes",
    "Streaming_Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless_Billing": "Yes",
    "Payment_Method": "Electronic check",
    "Monthly_Charges": 95.0,
    "Total_Charges": 285.0,
}


PREDICTION = {
    "prediction": 1,
    "prediction_label": "Churn",
    "churn_probability": 0.7709,
    "risk_level": "High",
    "logistic_probability": 0.7249,
    "catboost_probability": 0.7906,
    "threshold": 0.31,
    "model_weights": {"logistic_regression": 0.3, "catboost": 0.7},
}


def test_complete_profile_runs_prediction_and_explanation():
    async def extractor(message, current, requested):
        return COMPLETE_CUSTOMER

    service = ChatService(extractor=extractor, predictor=lambda customer: PREDICTION)
    response = asyncio.run(service.process("session_test", "complete profile"))

    assert response.status == "complete"
    assert response.completed_count == 19
    assert response.prediction.churn_probability == 0.7709
    assert response.explanation.risk_level == "High"
    assert "retention" in response.explanation.recommended_action.lower()


def test_conversation_keeps_asking_until_profile_is_complete():
    patches = [
        {
            "gender": "Male",
            "Senior_Citizen": 0,
            "Is_Married": "Yes",
            "Dependents": "No",
            "tenure": 20,
            "Phone_Service": "No",
            "Internet_Service": "No",
        },
        {
            "Contract": "One year",
            "Paperless_Billing": "No",
            "Payment_Method": "Mailed check",
            "Monthly_Charges": 20.0,
            "Total_Charges": 400.0,
        },
    ]

    async def extractor(message, current, requested):
        return patches.pop(0)

    service = ChatService(extractor=extractor, predictor=lambda customer: PREDICTION)
    first = asyncio.run(service.process("session_progress", "first details"))
    second = asyncio.run(service.process("session_progress", "remaining details"))

    assert first.status == "collecting"
    assert first.missing_fields
    assert first.collected_fields["Dual"] == "No phone service"
    assert first.collected_fields["Online_Security"] == "No internet service"
    assert second.status == "complete"


def test_conflicting_service_information_requires_clarification():
    values, issue = _apply_relationship_rules(
        {"Phone_Service": "No", "Dual": "No phone service"},
        {"Dual": "Yes"},
    )

    assert values["Dual"] == "Yes"
    assert issue is not None
    assert "clarify" in issue.lower()


def test_natural_profile_with_typos_is_normalized():
    values = extract_explicit_values(
        "a female cutomer with total charge of 5000.0 and tenur of 7 month "
        "with fiber optics internet and a senior citezen",
        [],
    )

    assert values["gender"] == "Female"
    assert values["Total_Charges"] == 5000.0
    assert values["tenure"] == 7
    assert values["Internet_Service"] == "Fiber optic"
    assert values["Senior_Citizen"] == 1


def test_ordered_yes_no_answers_follow_requested_field_order():
    values = extract_explicit_values(
        "yes\nno",
        ["Is_Married", "Dependents"],
    )

    assert values == {"Is_Married": "Yes", "Dependents": "No"}


def test_ordered_mixed_shorthand_answers_follow_question_order():
    requested = ["Contract", "Paperless_Billing", "Payment_Method"]

    assert extract_explicit_values("month\nyes\nelectronic", requested) == {
        "Contract": "Month-to-month",
        "Paperless_Billing": "Yes",
        "Payment_Method": "Electronic check",
    }
    assert extract_explicit_values("monthly\nyes\nelectronic", requested) == {
        "Contract": "Month-to-month",
        "Paperless_Billing": "Yes",
        "Payment_Method": "Electronic check",
    }


def test_common_typos_are_understood_for_single_follow_up():
    assert extract_explicit_values("Fiber optiv", ["Internet_Service"])[
        "Internet_Service"
    ] == "Fiber optic"
    assert extract_explicit_values("yrd", ["Online_Security"])[
        "Online_Security"
    ] == "Yes"


def test_bare_monthly_charge_has_no_arbitrary_upper_cap():
    values = extract_explicit_values("714.5", ["Monthly_Charges"])

    assert values["Monthly_Charges"] == 714.5


def test_multi_service_sentence_tracks_negation_and_positive_phrases():
    values = extract_explicit_values(
        "they dont have online security and have a online backup with a device "
        "protection and technical support",
        ["Online_Security", "Online_Backup", "Device_Protection", "Tech_Support"],
    )

    assert values["Online_Security"] == "No"
    assert values["Online_Backup"] == "Yes"
    assert values["Device_Protection"] == "Yes"
    assert values["Tech_Support"] == "Yes"


def test_labeled_complete_profile_preserves_explicit_yes_no_values():
    message = (
        "Gender: Female. Senior citizen: No. Married: Yes. Dependents: No. "
        "Tenure: 2 months. Phone service: Yes. Multiple phone lines: No. "
        "Internet service: Fiber optic. Online security: No. Online backup: No. "
        "Device protection: Yes. Technical support: No. Streaming TV: Yes. "
        "Streaming movies: Yes. Contract: Month-to-month. Paperless billing: Yes. "
        "Payment method: Electronic check. Monthly charges: 95.70. "
        "Total charges: 191.40."
    )

    values = extract_explicit_values(message, [])

    assert values == {
        **COMPLETE_CUSTOMER,
        "Is_Married": "Yes",
        "tenure": 2,
        "Device_Protection": "Yes",
        "Monthly_Charges": 95.7,
        "Total_Charges": 191.4,
    }


def test_negative_multiple_lines_phrasing_is_not_overridden_by_keyword():
    assert extract_explicit_values("multiple phone lines: No", [])["Dual"] == "No"
    assert (
        extract_explicit_values("has phone service but no multiple lines", [])["Dual"]
        == "No"
    )


def test_completed_profile_accepts_an_explicit_correction_without_ollama():
    profile = (
        "Gender: Female. Senior citizen: No. Married: No. Dependents: No. "
        "Tenure: 3 months. Phone service: Yes. Multiple phone lines: Yes. "
        "Internet service: Fiber optic. Online security: No. Online backup: No. "
        "Device protection: No. Technical support: No. Streaming TV: Yes. "
        "Streaming movies: Yes. Contract: Month-to-month. Paperless billing: Yes. "
        "Payment method: Electronic check. Monthly charges: 95. "
        "Total charges: 285."
    )
    service = ChatService(
        extractor=extract_customer_patch,
        predictor=lambda customer: PREDICTION,
    )

    first = asyncio.run(service.process("explicit_correction", profile))
    corrected = asyncio.run(
        service.process(
            "explicit_correction", "Correction: multiple phone lines is No."
        )
    )

    assert first.status == "complete"
    assert first.collected_fields["Dual"] == "Yes"
    assert corrected.status == "complete"
    assert corrected.collected_fields["Dual"] == "No"


def test_invalid_completed_field_is_requested_again_and_accepts_short_correction():
    async def extractor(message, current, requested):
        if not current:
            return {**COMPLETE_CUSTOMER, "tenure": 99}
        return extract_explicit_values(message, requested)

    service = ChatService(extractor=extractor, predictor=lambda customer: PREDICTION)
    first = asyncio.run(service.process("invalid_tenure", "complete profile"))

    assert first.status == "clarification"
    assert first.requested_fields == ["tenure"]
    assert first.missing_fields == ["tenure"]
    assert first.completed_count == 18
    assert "tenure" in first.message.lower()

    corrected = asyncio.run(service.process("invalid_tenure", "72"))

    assert corrected.status == "complete"
    assert corrected.collected_fields["tenure"] == 72


def test_relationship_conflict_requests_fields_and_accepts_ordered_correction():
    async def extractor(message, current, requested):
        if not current:
            return {
                **COMPLETE_CUSTOMER,
                "Phone_Service": "No",
                "Dual": "Yes",
            }
        return extract_explicit_values(message, requested)

    service = ChatService(extractor=extractor, predictor=lambda customer: PREDICTION)
    first = asyncio.run(service.process("phone_conflict", "complete profile"))

    assert first.status == "clarification"
    assert first.requested_fields == ["Phone_Service", "Dual"]
    assert first.missing_fields == ["Phone_Service", "Dual"]
    assert first.completed_count == 17

    corrected = asyncio.run(service.process("phone_conflict", "yes\nno"))

    assert corrected.status == "complete"
    assert corrected.collected_fields["Phone_Service"] == "Yes"
    assert corrected.collected_fields["Dual"] == "No"
