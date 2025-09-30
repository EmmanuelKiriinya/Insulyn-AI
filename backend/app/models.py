from typing import Union, Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from datetime import datetime

class DiabetesInput(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=200, description="Glucose level in mg/dL")
    blood_pressure: float = Field(..., ge=0, le=122, description="Blood pressure in mmHg")
    skin_thickness: float = Field(..., ge=0, le=99, description="Skin thickness in mm")
    insulin: float = Field(..., ge=0, le=846, description="Insulin level in mu U/ml")
    weight: float = Field(..., ge=20, le=200, description="Weight in kg")
    height: float = Field(..., ge=0.5, le=2.5, description="Height in meters")
    diabetes_pedigree_function: float = Field(..., ge=0.08, le=2.42, description="Diabetes pedigree function")
    age: int = Field(..., ge=21, le=81, description="Age in years")
    
    @validator('height')
    def validate_height(cls, v):
        if v <= 0:
            raise ValueError('Height must be positive')
        return v
    
    @property
    def bmi(self) -> float:
        """Calculate BMI from weight and height"""
        return round(self.weight / (self.height ** 2), 1)


class MLModelOutput(BaseModel):
    risk_label: str  # "high risk" or "low risk"
    probability: float
    feature_importances: Optional[Dict[str, float]] = None
    calculated_bmi: float


class LLMAdviceResponse(BaseModel):
    risk_summary: str
    clinical_interpretation: List[str]
    recommendations: Dict[str, Any]
    prevention_tips: List[str]
    monitoring_plan: List[str]
    clinician_message: str
    feature_explanation: str   # must be string, not dict
    safety_note: str



class CombinedResponse(BaseModel):
    timestamp: datetime
    ml_output: MLModelOutput
    llm_advice: LLMAdviceResponse
    bmi_category: str


class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User's message about diet or diabetes")
    conversation_context: Optional[str] = Field(None, description="Optional context about previous conversation")


class ChatResponse(BaseModel):
    response: str
    timestamp: datetime
    suggestions: list[str] = Field(default_factory=list)


class DietPlanRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    weight: float = Field(..., ge=30, le=200)
    height: float = Field(..., ge=1.0, le=2.5)
    dietary_preferences: Optional[str] = Field(None, description="e.g., vegetarian, low-carb, etc.")
    health_conditions: Optional[str] = Field(None, description="e.g., high BP, cholesterol, etc.")
    diabetes_risk: Optional[str] = Field(None, description="low/medium/high risk")


class DietPlanResponse(BaseModel):
    daily_calorie_target: int
    meal_plan: Dict[str, list[str]]
    foods_to_include: list[str]
    foods_to_avoid: list[str]
    portion_guidance: list[str]
    timing_recommendations: list[str]
