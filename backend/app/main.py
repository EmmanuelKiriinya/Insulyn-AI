# from fastapi import FastAPI, HTTPException, status
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List
# from datetime import datetime
# import logging
# import json

# from app.config import settings
# from app.models import DiabetesInput, CombinedResponse, MLModelOutput, LLMAdviceResponse, ChatMessage, ChatResponse, DietPlanRequest, DietPlanResponse
# from app.ml_model import diabetes_model, initialize_model
# from app.llm_chain import insulyn_llm, initialize_llm_service

# # Configure logging
# logging.basicConfig(
#     level=getattr(logging, settings.LOG_LEVEL),
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title=settings.PROJECT_NAME,
#     description="Insulyn AI - Diabetes Type 2 Detection, Prevention and Dietary Guidance System",
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize components on startup
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Starting Insulyn AI application...")
    
#     # Initialize ML model
#     model_loaded = initialize_model()
#     if not model_loaded:
#         logger.error("ML model failed to load")
#         raise RuntimeError("ML model initialization failed")
#     logger.info("ML model loaded successfully")
    
#     # Initialize LLM service
#     llm_ready = initialize_llm_service()
#     if not llm_ready:
#         logger.warning("LLM service initialization failed - running in limited mode")
#     else:
#         logger.info("LLM service initialized successfully")
    
#     logger.info("Insulyn AI application started successfully")

# # Health check endpoint
# @app.get("/")
# async def root():
#     return {"message": "Insulyn AI API is running", "status": "healthy"}

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow(),
#         "version": "1.0.0",
#         "services": {
#             "ml_model": diabetes_model.model is not None,
#             "llm_service": insulyn_llm.llm is not None
#         }
#     }

# # Main prediction endpoint (existing)
# @app.post("/api/v1/predict", response_model=CombinedResponse)
# async def predict_diabetes_risk(input_data: DiabetesInput):
#     # ... existing prediction code ...
#     pass

# # Chat endpoint for diabetes and diet discussions
# @app.post("/api/v1/chat", response_model=ChatResponse)
# async def chat_about_diabetes(chat_message: ChatMessage):
#     """
#     Chat with Insulyn AI about diabetes, diet, nutrition, and lifestyle.
    
#     Example questions:
#     - "What foods should I avoid if I'm prediabetic?"
#     - "Can you suggest a diabetes-friendly breakfast?"
#     - "How does exercise help with blood sugar control?"
#     - "What are the symptoms of type 2 diabetes?"
#     - "Can you give me a sample meal plan for a day?"
#     """
#     try:
#         response_text = insulyn_llm.chat_about_diabetes(
#             user_message=chat_message.message,
#             conversation_context=chat_message.conversation_context
#         )
        
#         # Generate follow-up suggestions
#         suggestions = generate_followup_suggestions(chat_message.message)
        
#         return ChatResponse(
#             response=response_text,
#             timestamp=datetime.utcnow(),
#             suggestions=suggestions
#         )
        
#     except Exception as e:
#         logger.error(f"Error in chat: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing chat message: {str(e)}"
#         )

# # Diet plan generation endpoint
# @app.post("/api/v1/diet-plan", response_model=DietPlanResponse)
# async def generate_diet_plan(diet_request: DietPlanRequest):
#     """
#     Generate a personalized diabetes-friendly diet plan.
    
#     This creates a customized meal plan based on:
#     - Age, weight, height (to calculate BMI and calorie needs)
#     - Dietary preferences (vegetarian, low-carb, etc.)
#     - Health conditions
#     - Diabetes risk level
#     """
#     try:
#         diet_data = {
#             'age': diet_request.age,
#             'weight': diet_request.weight,
#             'height': diet_request.height,
#             'dietary_preferences': diet_request.dietary_preferences,
#             'health_conditions': diet_request.health_conditions,
#             'diabetes_risk': diet_request.diabetes_risk
#         }
        
#         diet_plan_text = insulyn_llm.generate_diet_plan(diet_data)
        
#         # Parse the diet plan response (simplified - in production, use proper parsing)
#         return parse_diet_plan_response(diet_plan_text, diet_request)
        
#     except Exception as e:
#         logger.error(f"Error generating diet plan: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error generating diet plan: {str(e)}"
#         )

# # Clear chat history endpoint
# @app.post("/api/v1/chat/clear")
# async def clear_chat_history():
#     """
#     Clear the conversation history/memory.
#     Useful when starting a new conversation topic.
#     """
#     try:
#         insulyn_llm.clear_chat_memory()
#         return {"message": "Chat history cleared successfully", "timestamp": datetime.utcnow()}
#     except Exception as e:
#         logger.error(f"Error clearing chat history: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error clearing chat history: {str(e)}"
#         )

# # Get chat topics endpoint
# @app.get("/api/v1/chat/topics")
# async def get_chat_topics():
#     """
#     Get suggested topics for diabetes and diet discussions.
#     """
#     topics = [
#         "Foods to eat and avoid for diabetes",
#         "Meal planning and portion control",
#         "Exercise and physical activity recommendations",
#         "Blood sugar monitoring and targets",
#         "Symptoms and early warning signs",
#         "Medication and treatment options",
#         "Weight management strategies",
#         "Eating out and social situations",
#         "Stress management and sleep",
#         "Family history and genetic factors"
#     ]
#     return {"topics": topics}

# # Helper functions
# def generate_followup_suggestions(user_message: str) -> List[str]:
#     """Generate relevant follow-up question suggestions"""
#     user_message_lower = user_message.lower()
    
#     if any(word in user_message_lower for word in ['food', 'diet', 'eat', 'nutrition']):
#         return [
#             "Can you give me specific meal examples?",
#             "What about snacks between meals?",
#             "How do I read food labels for diabetes?"
#         ]
#     elif any(word in user_message_lower for word in ['exercise', 'workout', 'activity']):
#         return [
#             "What types of exercise are best?",
#             "How often should I exercise?",
#             "What about exercise and blood sugar levels?"
#         ]
#     elif any(word in user_message_lower for word in ['symptom', 'sign', 'feel']):
#         return [
#             "When should I see a doctor?",
#             "Are there emergency warning signs?",
#             "How is diabetes diagnosed?"
#         ]
#     else:
#         return [
#             "Can you explain this in simpler terms?",
#             "What are the next steps I should take?",
#             "Where can I find more resources?"
#         ]

# def parse_diet_plan_response(diet_plan_text: str, diet_request: DietPlanRequest) -> DietPlanResponse:
#     """Parse LLM diet plan response into structured format"""
#     # Calculate BMI
#     bmi = diet_request.weight / (diet_request.height ** 2)
    
#     # Estimate daily calorie target based on BMI and weight management
#     if bmi > 25:
#         calorie_target = int(diet_request.weight * 25)  # Weight loss
#     else:
#         calorie_target = int(diet_request.weight * 30)  # Maintenance
    
#     # Default meal plan structure (in production, parse from LLM response)
#     meal_plan = {
#         "breakfast": [
#             "Whole grain cereal with low-fat milk",
#             "Boiled eggs or protein source",
#             "Fresh fruits"
#         ],
#         "lunch": [
#             "Lean protein (chicken/fish/tofu)",
#             "Large portion of vegetables",
#             "Small portion of whole grains"
#         ],
#         "dinner": [
#             "Grilled fish or plant-based protein",
#             "Steamed vegetables",
#             "Small portion of complex carbs"
#         ],
#         "snacks": [
#             "Greek yogurt with berries",
#             "Handful of nuts",
#             "Vegetable sticks with hummus"
#         ]
#     }
    
#     foods_to_include = [
#         "Leafy green vegetables", "Whole grains", "Lean proteins",
#         "Legumes and beans", "Nuts and seeds", "Berries and low-sugar fruits"
#     ]
    
#     foods_to_avoid = [
#         "Sugary drinks and sodas", "Processed snacks", "White bread and pasta",
#         "Fried foods", "High-sugar fruits", "Full-fat dairy products"
#     ]
    
#     portion_guidance = [
#         "Use smaller plates for meals",
#         "Fill half your plate with non-starchy vegetables",
#         "Limit carbohydrate portions to 1/4 of your plate",
#         "Include protein in every meal"
#     ]
    
#     timing_recommendations = [
#         "Eat every 3-4 hours to maintain stable blood sugar",
#         "Have breakfast within 1 hour of waking up",
#         "Avoid eating 2-3 hours before bedtime",
#         "Space meals evenly throughout the day"
#     ]
    
#     return DietPlanResponse(
#         daily_calorie_target=calorie_target,
#         meal_plan=meal_plan,
#         foods_to_include=foods_to_include,
#         foods_to_avoid=foods_to_avoid,
#         portion_guidance=portion_guidance,
#         timing_recommendations=timing_recommendations
#     )

# # Model information endpoint (existing)
# @app.get("/api/v1/model/info")
# async def get_model_info():
#     """Get information about the ML model"""
#     return {
#         "model_loaded": diabetes_model.model_loaded,
#         "model_metadata": diabetes_model.model_metadata,
#         "feature_names": diabetes_model.feature_names,
#         "risk_threshold": 0.5
#     }

# # Error handlers (existing)
# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
#     return {
#         "detail": exc.detail,
#         "status_code": exc.status_code
#     }

# @app.exception_handler(Exception)
# async def general_exception_handler(request, exc):
#     logger.error(f"Unhandled exception: {exc}")
#     return {
#         "detail": "Internal server error",
#         "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
#     }

# from fastapi import FastAPI, HTTPException, status, Request
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List
# from datetime import datetime
# import logging

# from app.config import settings
# from app.models import (
#     DiabetesInput, CombinedResponse, MLModelOutput,
#     LLMAdviceResponse, ChatMessage, ChatResponse,
#     DietPlanRequest, DietPlanResponse
# )
# from app.ml_model import diabetes_model, initialize_model
# from app.llm_chain import insulyn_llm, initialize_llm_service

# # -------------------------------------------------
# # Logging
# # -------------------------------------------------
# logging.basicConfig(
#     level=getattr(logging, settings.LOG_LEVEL, "INFO"),
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # -------------------------------------------------
# # App creation
# # -------------------------------------------------
# app = FastAPI(
#     title=settings.PROJECT_NAME,
#     description="Insulyn AI - Diabetes Type 2 Detection, Prevention and Dietary Guidance System",
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # Allow frontend requests (adjust origins in production)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------------------------------
# # Startup
# # -------------------------------------------------
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Starting Insulyn AI application...")
#     if not initialize_model():
#         logger.error("❌ ML model failed to load")
#         raise RuntimeError("ML model initialization failed")
#     logger.info("✅ ML model loaded")

#     if initialize_llm_service():
#         logger.info("✅ LLM service initialized")
#     else:
#         logger.warning("⚠️ LLM service failed to initialize (limited mode)")

# # -------------------------------------------------
# # Health / Info
# # -------------------------------------------------
# @app.get("/")
# async def root():
#     return {"message": "Insulyn AI API is running", "status": "healthy"}

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow(),
#         "version": "1.0.0",
#         "services": {
#             "ml_model": diabetes_model.model is not None,
#             "llm_service": insulyn_llm.llm is not None
#         }
#     }

# # -------------------------------------------------
# # Prediction
# # -------------------------------------------------
# @app.post("/api/v1/predict", response_model=CombinedResponse)
# async def predict_diabetes_risk(input_data: DiabetesInput):
#     """
#     Run the diabetes risk model and optionally LLM advice.
#     """
#     try:
#         ml_output: MLModelOutput = diabetes_model.predict(input_data)
#         llm_output: LLMAdviceResponse = insulyn_llm.generate_advice(ml_output)
#         return CombinedResponse(
#             model_output=ml_output,
#             llm_advice=llm_output,
#             timestamp=datetime.utcnow()
#         )
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error generating prediction"
#         )

# # -------------------------------------------------
# # Chat
# # -------------------------------------------------
# @app.post("/api/v1/chat", response_model=ChatResponse)
# async def chat_about_diabetes(chat_message: ChatMessage):
#     """
#     Chat with Insulyn AI about diabetes, diet, or lifestyle.
#     """
#     try:
#         response_text = insulyn_llm.chat_about_diabetes(
#             user_message=chat_message.message,
#             conversation_context=chat_message.conversation_context
#         )
#         suggestions = generate_followup_suggestions(chat_message.message)
#         return ChatResponse(
#             response=response_text,
#             timestamp=datetime.utcnow(),
#             suggestions=suggestions
#         )
#     except Exception as e:
#         logger.error(f"Chat error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error processing chat message"
#         )

# @app.post("/api/v1/chat/clear")
# async def clear_chat_history():
#     try:
#         insulyn_llm.clear_chat_memory()
#         return {"message": "Chat history cleared", "timestamp": datetime.utcnow()}
#     except Exception as e:
#         logger.error(f"Clear history error: {e}")
#         raise HTTPException(status_code=500, detail="Error clearing chat history")

# @app.get("/api/v1/chat/topics")
# async def get_chat_topics():
#     return {
#         "topics": [
#             "Foods to eat and avoid for diabetes",
#             "Meal planning and portion control",
#             "Exercise recommendations",
#             "Blood sugar monitoring",
#             "Symptoms and early warning signs",
#             "Medication and treatment options",
#             "Weight management",
#             "Stress management and sleep"
#         ]
#     }

# # -------------------------------------------------
# # Diet plan
# # -------------------------------------------------
# @app.post("/api/v1/diet-plan", response_model=DietPlanResponse)
# async def generate_diet_plan(diet_request: DietPlanRequest):
#     try:
#         diet_data = {
#             "age": diet_request.age,
#             "weight": diet_request.weight,
#             "height": diet_request.height,
#             "dietary_preferences": diet_request.dietary_preferences,
#             "health_conditions": diet_request.health_conditions,
#             "diabetes_risk": diet_request.diabetes_risk
#         }
#         plan_text = insulyn_llm.generate_diet_plan(diet_data)
#         return parse_diet_plan_response(plan_text, diet_request)
#     except Exception as e:
#         logger.error(f"Diet plan error: {e}")
#         raise HTTPException(status_code=500, detail="Error generating diet plan")

# # -------------------------------------------------
# # Helpers
# # -------------------------------------------------
# def generate_followup_suggestions(user_message: str) -> List[str]:
#     msg = user_message.lower()
#     if any(k in msg for k in ["food", "diet", "eat", "nutrition"]):
#         return [
#             "Can you give me specific meal examples?",
#             "What about snacks between meals?",
#             "How do I read food labels for diabetes?"
#         ]
#     if any(k in msg for k in ["exercise", "workout", "activity"]):
#         return [
#             "What types of exercise are best?",
#             "How often should I exercise?",
#             "What about exercise and blood sugar?"
#         ]
#     if any(k in msg for k in ["symptom", "sign", "feel"]):
#         return [
#             "When should I see a doctor?",
#             "Are there emergency warning signs?",
#             "How is diabetes diagnosed?"
#         ]
#     return [
#         "Can you explain this in simpler terms?",
#         "What are the next steps I should take?",
#         "Where can I find more resources?"
#     ]

# def parse_diet_plan_response(plan_text: str, req: DietPlanRequest) -> DietPlanResponse:
#     bmi = req.weight / (req.height ** 2)
#     calorie_target = int(req.weight * (25 if bmi > 25 else 30))
#     # Basic placeholder meal plan
#     return DietPlanResponse(
#         daily_calorie_target=calorie_target,
#         meal_plan={
#             "breakfast": ["Whole grain cereal", "Boiled eggs", "Fresh fruits"],
#             "lunch": ["Lean protein", "Vegetables", "Whole grains"],
#             "dinner": ["Grilled fish", "Steamed vegetables", "Complex carbs"],
#             "snacks": ["Greek yogurt", "Nuts", "Veg sticks"]
#         },
#         foods_to_include=[
#             "Leafy greens", "Whole grains", "Lean proteins",
#             "Legumes", "Nuts", "Low-sugar fruits"
#         ],
#         foods_to_avoid=[
#             "Sugary drinks", "Processed snacks", "White bread",
#             "Fried foods", "High-sugar fruits", "Full-fat dairy"
#         ],
#         portion_guidance=[
#             "Fill half the plate with non-starchy vegetables",
#             "Limit carbs to a quarter of the plate",
#             "Include protein in every meal"
#         ],
#         timing_recommendations=[
#             "Eat every 3–4 hours",
#             "Have breakfast within 1 hour of waking",
#             "Avoid eating 2–3 hours before bed"
#         ]
#     )

# # -------------------------------------------------
# # Model info
# # -------------------------------------------------
# @app.get("/api/v1/model/info")
# async def get_model_info():
#     return {
#         "model_loaded": diabetes_model.model_loaded,
#         "model_metadata": diabetes_model.model_metadata,
#         "feature_names": diabetes_model.feature_names,
#         "risk_threshold": 0.5
#     }

# # -------------------------------------------------
# # Error handlers
# # -------------------------------------------------
# @app.exception_handler(HTTPException)
# async def http_exception_handler(request: Request, exc: HTTPException):
#     logger.error(f"HTTP {exc.status_code}: {exc.detail}")
#     return {"detail": exc.detail, "status_code": exc.status_code}

# @app.exception_handler(Exception)
# async def general_exception_handler(request: Request, exc: Exception):
#     logger.error(f"Unhandled exception: {exc}")
#     return {"detail": "Internal server error", "status_code": 500}

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from datetime import datetime
import logging
import json

from app.config import settings
from app.models import (
    DiabetesInput, CombinedResponse, MLModelOutput,
    LLMAdviceResponse, ChatMessage, ChatResponse,
    DietPlanRequest, DietPlanResponse
)
from app.ml_model import diabetes_model, initialize_model
from app.llm_chain import insulyn_llm, initialize_llm_service

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# App creation
# -------------------------------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Insulyn AI - Diabetes Type 2 Detection, Prevention and Dietary Guidance System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Startup
# -------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Insulyn AI application...")
    if not initialize_model():
        logger.error("❌ ML model failed to load")
        raise RuntimeError("ML model initialization failed")
    logger.info("✅ ML model loaded")

    if initialize_llm_service():
        logger.info("✅ LLM service initialized")
    else:
        logger.warning("⚠️ LLM service failed to initialize (limited mode)")

# -------------------------------------------------
# Health / Info
# -------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Insulyn AI API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "services": {
            "ml_model": diabetes_model.model is not None,
            "llm_service": insulyn_llm.llm is not None
        }
    }

# -------------------------------------------------
# Prediction
# -------------------------------------------------
@app.post("/api/v1/predict", response_model=CombinedResponse)
async def predict_diabetes_risk(input_data: DiabetesInput):
    try:
        # ------------------------------
        # Prepare features and compute BMI
        # ------------------------------
        features = input_data.dict()
        height = features.get("height")
        weight = features.get("weight")
        if not features.get("bmi"):
            if height and height > 0:
                features["bmi"] = round(weight / (height ** 2), 1)
            else:
                features["bmi"] = "not available"

        calculated_bmi = features["bmi"]
        bmi_category = (
            diabetes_model.get_bmi_category(calculated_bmi)
            if isinstance(calculated_bmi, (int, float))
            else "not available"
        )

        # ------------------------------
        # ML Prediction
        # ------------------------------
        risk_label, probability, feature_importances = diabetes_model.predict(features)
        ml_model_output = MLModelOutput(
            risk_label=risk_label,
            probability=probability,
            calculated_bmi=calculated_bmi,
            feature_importances=feature_importances,
        )

        # ------------------------------
        # Prepare patient data for LLM
        # ------------------------------
        patient_data = {
            "pregnancies": features.get("pregnancies", "not available"),
            "glucose": features.get("glucose", "not available"),
            "blood_pressure": features.get("blood_pressure", "not available"),
            "skin_thickness": features.get("skin_thickness", "not available"),
            "insulin": features.get("insulin", "not available"),
            "diabetes_pedigree_function": features.get("diabetes_pedigree_function", "not available"),
            "age": features.get("age", "not available"),
            "bmi": calculated_bmi,
            "bmi_category": bmi_category,
        }

        # ------------------------------
        # LLM Advice with retries
        # ------------------------------
        max_retries = 3
        llm_response = None

        for attempt in range(max_retries):
            llm_output = insulyn_llm.generate_advice(patient_data, ml_model_output.dict())

            # Try to parse JSON
            if isinstance(llm_output, dict):
                llm_raw_output = llm_output
            else:
                llm_text = str(llm_output).strip()
                if llm_text.startswith("```"):
                    llm_text = llm_text.strip("`").replace("json", "", 1).strip()
                try:
                    llm_raw_output = json.loads(llm_text)
                except json.JSONDecodeError:
                    logger.warning(f"❌ Attempt {attempt+1}: Could not parse LLM output as JSON")
                    continue  # retry

            # Normalize required fields
            required_fields = [
                "risk_summary",
                "clinical_interpretation",
                "recommendations",
                "prevention_tips",
                "monitoring_plan",
                "clinician_message",
                "feature_explanation",
                "safety_note",
            ]
            for field in required_fields:
                if field not in llm_raw_output or llm_raw_output[field] is None:
                    llm_raw_output[field] = "Not available"

            # Ensure list-type fields are lists
            for field in ["clinical_interpretation", "monitoring_plan", "prevention_tips"]:
                if not isinstance(llm_raw_output[field], list):
                    llm_raw_output[field] = [llm_raw_output[field]]

            # Ensure recommendations is a dict
            if not isinstance(llm_raw_output["recommendations"], dict):
                llm_raw_output["recommendations"] = {"note": str(llm_raw_output["recommendations"])}

            # Ensure feature_explanation is always a string
            if not isinstance(llm_raw_output["feature_explanation"], str):
                llm_raw_output["feature_explanation"] = str(llm_raw_output["feature_explanation"])

            # Validate with Pydantic
            try:
                llm_response = LLMAdviceResponse(**llm_raw_output)
                break  # ✅ success
            except Exception as e:
                logger.warning(f"❌ Attempt {attempt+1}: Validation failed - {e}")
                continue  # retry

        if not llm_response:
            logger.error("❌ All retries failed. Returning placeholders.")
            llm_response = LLMAdviceResponse(
                risk_summary="Not available",
                clinical_interpretation=["Not available"],
                recommendations={"note": "Not available"},
                prevention_tips=["Not available"],
                monitoring_plan=["Not available"],
                clinician_message="Not available",
                feature_explanation="Not available",
                safety_note="Not available",
            )

        # ------------------------------
        # Return combined response
        # ------------------------------
        return CombinedResponse(
            ml_output=ml_model_output,
            bmi_category=bmi_category,
            llm_advice=llm_response,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating prediction",
        )

# -------------------------------------------------
# Chat
# -------------------------------------------------
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_about_diabetes(chat_message: ChatMessage):
    try:
        response_text = insulyn_llm.chat_about_diabetes(
            user_message=chat_message.message,
            conversation_context=chat_message.conversation_context
        )
        suggestions = generate_followup_suggestions(chat_message.message)
        return ChatResponse(
            response=response_text,
            timestamp=datetime.utcnow(),
            suggestions=suggestions
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing chat message"
        )

@app.post("/api/v1/chat/clear")
async def clear_chat_history():
    try:
        insulyn_llm.clear_chat_memory()
        return {"message": "Chat history cleared", "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error(f"Clear history error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error clearing chat history")

@app.get("/api/v1/chat/topics")
async def get_chat_topics():
    return {
        "topics": [
            "Foods to eat and avoid for diabetes",
            "Meal planning and portion control",
            "Exercise recommendations",
            "Blood sugar monitoring",
            "Symptoms and early warning signs",
            "Medication and treatment options",
            "Weight management",
            "Stress management and sleep"
        ]
    }

# -------------------------------------------------
# Diet plan
# -------------------------------------------------
@app.post("/api/v1/diet-plan", response_model=DietPlanResponse)
async def generate_diet_plan(diet_request: DietPlanRequest):
    try:
        plan_text = insulyn_llm.generate_diet_plan(
            age=diet_request.age,
            weight=diet_request.weight,
            height=diet_request.height,
            dietary_preferences=diet_request.dietary_preferences,
            health_conditions=diet_request.health_conditions,
            diabetes_risk=diet_request.diabetes_risk
        )
        return parse_diet_plan_response(plan_text, diet_request)
    except Exception as e:
        logger.error(f"Diet plan error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating diet plan")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def generate_followup_suggestions(user_message: str) -> List[str]:
    msg = user_message.lower()
    if any(k in msg for k in ["food", "diet", "eat", "nutrition"]):
        return [
            "Can you give me specific meal examples?",
            "What about snacks between meals?",
            "How do I read food labels for diabetes?"
        ]
    if any(k in msg for k in ["exercise", "workout", "activity"]):
        return [
            "What types of exercise are best?",
            "How often should I exercise?",
            "What about exercise and blood sugar?"
        ]
    if any(k in msg for k in ["symptom", "sign", "feel"]):
        return [
            "When should I see a doctor?",
            "Are there emergency warning signs?",
            "How is diabetes diagnosed?"
        ]
    return [
        "Can you explain this in simpler terms?",
        "What are the next steps I should take?",
        "Where can I find more resources?"
    ]

def parse_diet_plan_response(plan_text: str, req: DietPlanRequest) -> DietPlanResponse:
    bmi = req.weight / (req.height ** 2)
    calorie_target = int(req.weight * (25 if bmi > 25 else 30))
    return DietPlanResponse(
        daily_calorie_target=calorie_target,
        meal_plan={
            "breakfast": ["Whole grain cereal", "Boiled eggs", "Fresh fruits"],
            "lunch": ["Lean protein", "Vegetables", "Whole grains"],
            "dinner": ["Grilled fish", "Steamed vegetables", "Complex carbs"],
            "snacks": ["Greek yogurt", "Nuts", "Veg sticks"]
        },
        foods_to_include=[
            "Leafy greens", "Whole grains", "Lean proteins",
            "Legumes", "Nuts", "Low-sugar fruits"
        ],
        foods_to_avoid=[
            "Sugary drinks", "Processed snacks", "White bread",
            "Fried foods", "High-sugar fruits", "Full-fat dairy"
        ],
        portion_guidance=[
            "Fill half the plate with non-starchy vegetables",
            "Limit carbs to a quarter of the plate",
            "Include protein in every meal"
        ],
        timing_recommendations=[
            "Eat every 3–4 hours",
            "Have breakfast within 1 hour of waking",
            "Avoid eating 2–3 hours before bed"
        ]
    )

# -------------------------------------------------
# Model info
# -------------------------------------------------
@app.get("/api/v1/model/info")
async def get_model_info():
    return {
        "model_loaded": diabetes_model.model_loaded,
        "model_metadata": diabetes_model.model_metadata,
        "feature_names": diabetes_model.feature_names,
        "risk_threshold": 0.5
    }

# -------------------------------------------------
# Error handlers
# -------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
