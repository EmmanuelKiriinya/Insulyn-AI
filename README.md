# Insulyn AI - Complete Project Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [API Documentation](#api-documentation)
4. [Installation Guide](#installation-guide)
5. [Configuration](#configuration)
6. [Machine Learning Model](#machine-learning-model)
7. [LLM Integration](#llm-integration)
8. [Database Schema](#database-schema)
9. [Error Handling](#error-handling)
10. [Testing](#testing)
11. [Deployment](#deployment)
12. [Future Enhancements](#future-enhancements)

---

## 🏥 Project Overview

### **What is Insulyn AI?**
Insulyn AI is a Type 2 Diabetes Detection and Management System that combines machine learning predictions with AI-powered clinical advice. The system helps users assess their diabetes risk and provides personalized dietary and lifestyle recommendations.

### **Key Features**
- 🧠 **Diabetes Risk Prediction**: ML model trained on clinical parameters
- 💬 **AI Chat Assistant**: Natural conversations about diabetes and diet
- 🍽️ **Personalized Diet Plans**: Custom meal plans based on user profile
- 📊 **BMI Calculation**: Automatic BMI calculation from weight and height
- 🎯 **Feature Importance**: Shows which factors contribute most to risk
- 🔒 **No Authentication Required**: Simple and accessible

### **Target Users**
- Individuals concerned about diabetes risk
- People with prediabetes seeking management strategies
- Healthcare providers for preliminary screening
- General public for diabetes awareness

---

## 🏗️ System Architecture

### **Technology Stack**
Frontend (Future) → FastAPI Backend → ML Model → Groq LLM
↓ ↓ ↓
React.js Python XGBoost Llama 3

text

### **Project Structure**
```
insulyn_ai/
├── app/
│ ├── init.py
│ ├── main.py # FastAPI application & endpoints
│ ├── models.py # Pydantic data models
│ ├── ml_model.py # Diabetes prediction model
│ ├── llm_chain.py # Groq LLM integration
│ └── config.py # Configuration settings
├── data/
│ └── best_model.pkl # Trained ML model
├── tests/ # Test files
├── requirements.txt # Python dependencies
├── .env # Environment variables
├── .gitignore # Git ignore rules
└── README.md # Project documentation
```


### **Data Flow**
User Input → FastAPI → ML Model → Risk Prediction → Groq LLM → Clinical Advice → Response



---

## 📚 API Documentation

### **Base URL**
http://localhost:8000


### **Endpoints**

#### 1. **Health Check**
```http
GET /
GET /health
Response:

json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "ml_model": true,
    "llm_service": true
  }
}
```

2. Diabetes Risk Prediction

http
POST /api/v1/predict
Request Body:

```
{
  "pregnancies": 2,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "weight": 85,
  "height": 1.65,
  "diabetes_pedigree_function": 0.627,
  "age": 50
}
```
Response:

```
{
  "timestamp": "2024-01-15T10:30:00Z",
  "ml_output": {
    "risk_label": "high risk",
    "probability": 0.85,
    "feature_importances": {
      "Glucose": 0.3,
      "BMI": 0.25,
      "Age": 0.2
    },
    "calculated_bmi": 31.2
  },
  "llm_advice": {
    "risk_summary": "high risk of type 2 diabetes",
    "clinical_interpretation": ["..."],
    "recommendations": {"immediate": [...], "lifestyle": [...]},
    "prevention_tips": ["..."],
    "monitoring_plan": ["..."],
    "clinician_message": "...",
    "feature_explanation": "...",
    "safety_note": "..."
  },
  "bmi_category": "Obese"
}
```
3. AI Chat
```
http
POST /api/v1/chat
Request Body:

```
```
{
  "message": "What foods should I avoid with prediabetes?",
  "conversation_context": "Recently diagnosed with prediabetes"
}
```

Response:


```
{
  "response": "For prediabetes, focus on whole foods...",
  "timestamp": "2024-01-15T10:30:00Z",
  "suggestions": [
    "Can you give me specific meal examples?",
    "What about snacks between meals?"
  ]
}
```
4. Personalized Diet Plan

```
http
POST /api/v1/diet-plan
Request Body:


{
  "age": 45,
  "weight": 85,
  "height": 1.75,
  "dietary_preferences": "Mediterranean",
  "health_conditions": "High blood pressure",
  "diabetes_risk": "medium"
}
```
5. Chat Management
```
http
POST /api/v1/chat/clear
GET /api/v1/chat/topics
GET /api/v1/model/info
```
### 🚀 Installation Guide
Prerequisites
Python 3.12

pip (Python package manager)

Groq API account

***Step-by-Step Setup***  
Clone and Setup Environment

```
# Create project directory
mkdir insulyn_ai
cd insulyn_ai

# Create virtual environment
python -m venv yourvenv
source yourvenv/bin/activate  # On Windows: venv\Scripts\activate

# Clone repository (when available)
git clone <repository-url> .
Install Dependencies

pip install -r requirements.txt

```
Configure Environment
```
bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
nano .env
Add ML Model
```
```
# Place your trained model in data directory
mkdir data
# Add diabetes_model.pkl to data/
Run Application

bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Verify Installation
```

curl http://localhost:8000/health
## Dependencies(requirements.txt)
```
aiohappyeyeballs==2.6.1
aiohttp==3.12.15
aiosignal==1.4.0
annotated-types==0.7.0
anyio==4.11.0
attrs==25.3.0
bcrypt==4.2.0
certifi==2025.8.3
cffi==2.0.0
charset-normalizer==3.4.3
click==8.3.0
colorama==0.4.6
cryptography==46.0.1
dataclasses-json==0.6.7
Deprecated==1.2.18
distro==1.9.0
ecdsa==0.19.1
fastapi==0.115.0
frozenlist==1.7.0
greenlet==3.2.4
groq==0.32.0
gunicorn==22.0.0
h11==0.16.0
httpcore==1.0.9
httpx==0.27.2
idna==3.10
joblib==1.4.2
jsonpatch==1.33
jsonpointer==3.0.0
langchain==0.2.17
langchain-community==0.2.19
langchain-core==0.2.43
langchain-groq==0.1.10
langchain-text-splitters==0.2.4
langsmith==0.1.147
limits==5.5.0
marshmallow==3.26.1
multidict==6.6.4
mypy_extensions==1.1.0
numpy==1.26.4
orjson==3.11.3
packaging==24.2
pandas==2.2.3
passlib==1.7.4
propcache==0.3.2
pyasn1==0.6.1
pycparser==2.23
pydantic==2.9.2
pydantic-settings==2.3.4
pydantic_core==2.23.4
PyMySQL==1.1.1
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
python-jose==3.3.0
python-json-logger==2.0.7
python-multipart==0.0.9
pytz==2025.2
PyYAML==6.0.3
requests==2.32.3
requests-toolbelt==1.0.0
rsa==4.9.1
scikit-learn==1.5.2
scipy==1.16.2
sentry-sdk==2.17.0
setuptools==80.9.0
six==1.17.0
slowapi==0.1.9
sniffio==1.3.1
starlette==0.38.6
structlog==24.2.0
tenacity==8.5.0
threadpoolctl==3.6.0
typing-inspect==0.9.0
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
uvicorn==0.32.0
wheel==0.45.1
wrapt==1.17.3
xgboost==3.0.0
yarl==1.20.1

```

## ⚙️ Configuration(.env)

```
Environment Variables (.env)
env
# FastAPI Settings
PROJECT_NAME="Insulyn AI"
API_V1_STR="/api/v1"

# ML Model
MODEL_PATH="./data/diabetes_model.pkl"

# Groq LLM
GROQ_API_KEY="your-groq-api-key-here"
LLM_MODEL_NAME="llama3-70b-8192"
LLM_TEMPERATURE=0.0

# Application
ENVIRONMENT="development"
DEBUG=True
LOG_LEVEL="INFO"

# Server
HOST="0.0.0.0"
PORT=8000
RELOAD=True
```