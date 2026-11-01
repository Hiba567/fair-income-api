"""Request and response schemas: input validation happens here."""

from pydantic import BaseModel, Field


class Applicant(BaseModel):
    age: int = Field(..., ge=16, le=100, examples=[38])
    education_num: int = Field(..., ge=1, le=16, description="Years of education (1-16)",
                               examples=[13])
    hours_per_week: int = Field(..., ge=1, le=99, examples=[40])
    capital_gain: float = Field(0, ge=0, examples=[0])
    capital_loss: float = Field(0, ge=0, examples=[0])
    workclass: str = Field(..., examples=["Private"])
    marital_status: str = Field(..., examples=["Married-civ-spouse"])
    occupation: str = Field(..., examples=["Prof-specialty"])


class Prediction(BaseModel):
    probability: float = Field(..., description="Probability that income is >50K")
    label: int = Field(..., description="1 if probability >= threshold, else 0")
    threshold: float
    model_version: str
