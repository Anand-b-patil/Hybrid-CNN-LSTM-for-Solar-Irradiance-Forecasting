import os
import logging
from datetime import datetime
from typing import List, Optional
import json
from pathlib import Path

import torch
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator

try:
    from config import settings
except ImportError:
    # Fallback settings if config.py doesn't exist
    class Settings:
        CORS_ORIGINS = ["http://localhost:3000"]
        UPLOAD_FOLDER = "uploads"
        MAX_UPLOAD_SIZE = 10 * 1024 * 1024
        ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".tif"]
        MAX_SEQUENCE_IMAGES = 100
        MIN_SEQUENCE_IMAGES = 10
        LOG_LEVEL = "INFO"
        APP_NAME = "Solar Forecasting API"
        APP_VERSION = "1.0.0"
    settings = Settings()

try:
    from werkzeug.utils import secure_filename
except Exception:
    # Fallback simple sanitizer
    def secure_filename(name: str) -> str:
        return os.path.basename(name)

from scripts.EfficientNet import EfficientNetRegression
from scripts.lstm_model import SolarLSTMForecasting, HybridCNNLSTM
from scripts.preprocess import IRImageProcessor
from scripts.gemini_service import get_gemini_service

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, getattr(settings, 'LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/api.log') if os.path.exists('logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=getattr(settings, 'APP_NAME', 'Solar Forecasting API'),
    version=getattr(settings, 'APP_VERSION', '1.0.0'),
    description="""Solar Irradiance Forecasting API using Hybrid EfficientNet-B0 + BiLSTM.
    
    This API provides three main endpoints:
    - Nowcasting: Current irradiance prediction from a single IR image
    - Forecasting: Future irradiance prediction from a sequence of values
    - Hybrid: Combined nowcasting and forecasting from image sequence
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware - Restricted origins for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, 'CORS_ORIGINS', ["http://localhost:3000"]),
    allow_credentials=getattr(settings, 'CORS_ALLOW_CREDENTIALS', True),
    allow_methods=getattr(settings, 'CORS_ALLOW_METHODS', ["GET", "POST"]),
    allow_headers=getattr(settings, 'CORS_ALLOW_HEADERS', ["*"]),
)

# GZip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

UPLOAD_FOLDER = getattr(settings, 'UPLOAD_FOLDER', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic Models for Request/Response Validation
class ForecastRequest(BaseModel):
    """Request model for LSTM forecasting endpoint"""
    irradiance_sequence: List[float] = Field(
        ...,
        description="Sequence of historical irradiance values",
        min_items=10,
        max_items=1000
    )
    
    @validator('irradiance_sequence')
    def validate_sequence(cls, v):
        if not all(isinstance(x, (int, float)) and x >= 0 for x in v):
            raise ValueError('All irradiance values must be non-negative numbers')
        return v

    class Config:
        schema_extra = {
            "example": {
                "irradiance_sequence": [450.2, 478.5, 490.1, 502.3, 515.7, 530.2, 545.8, 560.1, 575.3, 590.5]
            }
        }


class NowcastResponse(BaseModel):
    """Response model for nowcasting predictions"""
    nowcast_irradiance: float = Field(..., description="Predicted current irradiance in W/m²")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    model: str = Field(..., description="Model used for prediction")


class ForecastResponse(BaseModel):
    """Response model for forecasting predictions"""
    forecast_irradiance: List[float] = Field(..., description="Predicted future irradiance values in W/m²")
    forecast_horizon: int = Field(..., description="Number of future time steps predicted")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    model: str = Field(..., description="Model used for prediction")


class HybridResponse(BaseModel):
    """Response model for hybrid predictions"""
    nowcast_sequence: List[float] = Field(..., description="Nowcast irradiance values for input sequence")
    forecast_irradiance: List[float] = Field(..., description="Forecasted future irradiance values")
    sequence_length: int = Field(..., description="Length of input sequence")
    forecast_horizon: int = Field(..., description="Number of future time steps predicted")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    model: str = Field(..., description="Model used for prediction")


class AIExplanationRequest(BaseModel):
    """Request model for AI explanation"""
    prediction_data: dict = Field(..., description="Prediction data to explain")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction_data": {
                    "nowcast_irradiance": 456.7,
                    "timestamp": "2026-01-11T12:00:00",
                    "model": "CNN_Regression"
                }
            }
        }


class AIRecommendationRequest(BaseModel):
    """Request model for AI recommendations"""
    prediction_data: dict = Field(..., description="Prediction data for recommendations")
    user_context: Optional[str] = Field(None, description="User context (e.g., 'solar farm', 'residential')")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction_data": {
                    "forecast_irradiance": [450.2, 478.5, 490.1, 502.3],
                    "forecast_horizon": 4
                },
                "user_context": "residential solar system"
            }
        }


class AITrendAnalysisRequest(BaseModel):
    """Request model for trend analysis"""
    historical_data: List[float] = Field(..., description="Historical irradiance values", min_items=5)
    forecast_data: Optional[List[float]] = Field(None, description="Optional forecast values")
    
    class Config:
        schema_extra = {
            "example": {
                "historical_data": [450.2, 478.5, 490.1, 502.3, 515.7, 530.2],
                "forecast_data": [545.8, 560.1, 575.3, 590.5]
            }
        }


class AIQuestionRequest(BaseModel):
    """Request model for AI Q&A"""
    question: str = Field(..., description="User's question about solar forecasting", min_length=5)
    context_data: Optional[dict] = Field(None, description="Optional context data")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What does an irradiance value of 500 W/m² mean for my solar panels?",
                "context_data": {"current_irradiance": 500}
            }
        }


class AIResponse(BaseModel):
    """Response model for AI-generated content"""
    content: str = Field(..., description="AI-generated content")
    timestamp: str = Field(..., description="Response timestamp")
    model: str = Field(..., description="AI model used")


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="API health status")
    models: dict = Field(..., description="Status of loaded models")
    device: str = Field(..., description="Device being used (CPU/CUDA)")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")


class SolarForecastingAPI:
    """API class for solar irradiance nowcasting and forecasting"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.image_processor = IRImageProcessor(target_size=(240, 320))

        self.nowcast_model = None
        self.forecast_model = None
        self.hybrid_model = None

        self.load_models()

    def load_models(self):
        try:
            if os.path.exists('models/best_efficientnet_model.pth'):
                logger.info("Loading CNN nowcasting model...")
                self.nowcast_model = EfficientNetRegression().to(self.device)
                checkpoint = torch.load('models/best_efficientnet_model.pth', map_location=self.device )
                self.nowcast_model.load_state_dict(checkpoint.get("model_state_dict", {}), strict=False)
                self.nowcast_model.eval()
                logger.info("CNN model loaded successfully")

            lstm_ckpt_path = None
            for candidate in ['models/best_lstm_model.pth']:
                if os.path.exists(candidate):
                    lstm_ckpt_path = candidate
                    break

            if lstm_ckpt_path:
                logger.info("Loading LSTM forecasting model...")
                checkpoint = torch.load(lstm_ckpt_path, map_location=self.device, weights_only=False)
                config = checkpoint.get('config', {})

                state_dict_keys = checkpoint.get('model_state_dict', {}).keys()
                inferred_bidirectional = any('lstm.weight_ih_l0_reverse' in k for k in state_dict_keys) if state_dict_keys else True

                use_legacy_head = any(k.startswith('fc.') for k in state_dict_keys) if state_dict_keys else False

                self.forecast_model = SolarLSTMForecasting(
                    input_size=1,
                    hidden_size=config.get('lstm_hidden_size', 128),
                    num_layers=config.get('lstm_num_layers', 2),
                    output_size=config.get('forecast_horizon', 4),
                    bidirectional=config.get('bidirectional', inferred_bidirectional),
                    use_legacy_head=use_legacy_head
                ).to(self.device)

                self.forecast_model.load_state_dict(checkpoint.get('model_state_dict', {}), strict=False)
                self.forecast_model.eval()
                logger.info("LSTM model loaded successfully")

            hybrid_ckpt_path = None
            for candidate in ['models/best_hybrid_model.pth', 'models1/best_hybrid_model.pth']:
                if os.path.exists(candidate):
                    hybrid_ckpt_path = candidate
                    break

            if hybrid_ckpt_path:
                logger.info("Loading hybrid CNN-LSTM model...")
                checkpoint = torch.load(hybrid_ckpt_path, map_location=self.device, weights_only=False)
                config = checkpoint.get('config', {})

                self.hybrid_model = HybridCNNLSTM(
                    sequence_length=config.get('sequence_length', 20),
                    lstm_hidden_size=config.get('lstm_hidden_size', 128),
                    forecast_horizon=config.get('forecast_horizon', 4)
                ).to(self.device)

                self.hybrid_model.load_state_dict(checkpoint.get('model_state_dict', {}), strict=False)
                self.hybrid_model.eval()
                logger.info("Hybrid model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        processed_img = self.image_processor.process_single_image(image_path)
        img_tensor = torch.tensor(processed_img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(self.device)

    def nowcast_single_image(self, image_path: str):
        if self.nowcast_model is None:
            raise RuntimeError("CNN nowcasting model not loaded")

        img_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            prediction = self.nowcast_model(img_tensor)
            irradiance = float(prediction.squeeze().cpu().item())

        return {
            'nowcast_irradiance': round(irradiance, 2),
            'timestamp': datetime.now().isoformat(),
            'model': 'CNN_Regression'
        }

    def forecast_from_sequence(self, irradiance_sequence: List[float]):
        if self.forecast_model is None:
            raise RuntimeError("LSTM forecasting model not loaded")

        seq_tensor = torch.tensor(irradiance_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            forecast = self.forecast_model(seq_tensor)
            forecast_values = forecast.squeeze().cpu().numpy().tolist()

        return {
            'forecast_irradiance': [round(float(val), 2) for val in forecast_values],
            'forecast_horizon': len(forecast_values),
            'timestamp': datetime.now().isoformat(),
            'model': 'LSTM_Forecasting'
        }

    def hybrid_predict(self, image_paths: List[str]):
        if self.hybrid_model is None:
            raise RuntimeError("Hybrid CNN-LSTM model not loaded")

        image_tensors = []
        for img_path in image_paths:
            img_tensor = self.preprocess_image(img_path)
            image_tensors.append(img_tensor.squeeze(0))

        sequence_tensor = torch.stack(image_tensors).unsqueeze(0)
        with torch.no_grad():
            nowcasts, forecasts = self.hybrid_model(sequence_tensor)
            nowcast_values = nowcasts.squeeze().cpu().numpy().tolist()
            forecast_values = forecasts.squeeze().cpu().numpy().tolist()

        return {
            'nowcast_sequence': [round(float(val), 2) for val in nowcast_values],
            'forecast_irradiance': [round(float(val), 2) for val in forecast_values],
            'sequence_length': len(nowcast_values),
            'forecast_horizon': len(forecast_values),
            'timestamp': datetime.now().isoformat(),
            'model': 'Hybrid_CNN_LSTM'
        }


api = SolarForecastingAPI()

# Initialize Gemini AI Service
gemini_service = None
try:
    gemini_api_key = getattr(settings, 'GEMINI_API_KEY', '')
    if gemini_api_key:
        gemini_service = get_gemini_service(
            api_key=gemini_api_key,
            model_name=getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash'),
            temperature=getattr(settings, 'GEMINI_TEMPERATURE', 0.7),
            max_tokens=getattr(settings, 'GEMINI_MAX_TOKENS', 2048)
        )
        if gemini_service:
            logger.info("Gemini AI Service initialized successfully")
        else:
            logger.warning("Gemini AI Service initialization failed")
    else:
        logger.warning("GEMINI_API_KEY not set - AI features will be disabled")
except Exception as e:
    logger.error(f"Failed to initialize Gemini service: {e}")
    gemini_service = None


def _validate_file_extension(filename: str) -> bool:
    """Validate file extension against allowed types"""
    allowed = getattr(settings, 'ALLOWED_EXTENSIONS', ['.png', '.jpg', '.jpeg', '.tiff', '.tif'])
    return any(filename.lower().endswith(ext) for ext in allowed)


async def _save_upload_file(upload_file: UploadFile, dest_path: str) -> None:
    """Save uploaded file with size limit validation"""
    max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 10 * 1024 * 1024)
    
    # Validate file extension
    if not _validate_file_extension(upload_file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {getattr(settings, 'ALLOWED_EXTENSIONS', [])}"
        )
    
    # Read and validate file size
    contents = await upload_file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
        )
    
    # Save file
    with open(dest_path, 'wb') as fh:
        fh.write(contents)
    await upload_file.close()
    
    logger.info(f"File saved: {dest_path} ({len(contents)} bytes)")


@app.get('/', tags=["Frontend"], summary="Main page")
async def index(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse('index.html', {"request": request})


@app.get('/favicon.ico', tags=["Frontend"], include_in_schema=False)
async def favicon():
    """Serve a small inline SVG favicon to avoid 404s when browsers request /favicon.ico."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        '<rect width="16" height="16" fill="#f6c84c"/>'
        '<circle cx="8" cy="8" r="5" fill="#ffffff"/>'
        '</svg>'
    )
    return Response(content=svg, media_type='image/svg+xml')


@app.post(
    '/predict',
    response_model=NowcastResponse,
    tags=["Nowcasting"],
    summary="Predict current solar irradiance",
    description="Upload a single infrared sky image to get the current solar irradiance prediction."
)
async def predict_single(file: UploadFile = File(..., description="Infrared sky image")):
    """Nowcast endpoint: Predict current irradiance from a single image"""
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded or filename is empty"
        )

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        await _save_upload_file(file, filepath)
        logger.info(f"Processing single image prediction: {filename}")
        result = api.nowcast_single_image(filepath)
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
            
        logger.info(f"Prediction successful: {result['nowcast_irradiance']} W/m²")
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"Error in single prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    '/forecast',
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Forecast future solar irradiance",
    description="Provide a sequence of historical irradiance values to predict future values."
)
async def forecast_sequence(req: ForecastRequest):
    """Forecast endpoint: Predict future irradiance from historical sequence"""
    sequence = req.irradiance_sequence
    
    logger.info(f"Processing forecast for sequence of length {len(sequence)}")
    
    try:
        result = api.forecast_from_sequence(sequence)
        logger.info(f"Forecast successful: {result['forecast_horizon']} steps predicted")
        return JSONResponse(result)
        
    except RuntimeError as e:
        logger.error(f"Model not loaded: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in forecasting: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting failed: {str(e)}"
        )


@app.post(
    '/hybrid_predict',
    response_model=HybridResponse,
    tags=["Hybrid"],
    summary="Hybrid nowcasting and forecasting",
    description="Upload a sequence of infrared sky images for combined nowcasting and forecasting."
)
async def hybrid_predict(
    files: List[UploadFile] = File(
        ...,
        description="Sequence of infrared sky images (min 10, max 100)"
    )
):
    """Hybrid endpoint: Combined nowcasting and forecasting from image sequence"""
    min_images = getattr(settings, 'MIN_SEQUENCE_IMAGES', 10)
    max_images = getattr(settings, 'MAX_SEQUENCE_IMAGES', 100)
    
    if not files or len(files) < min_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'At least {min_images} images required for sequence prediction'
        )
    
    if len(files) > max_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Maximum {max_images} images allowed per request'
        )

    saved_paths: List[str] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info(f"Processing hybrid prediction with {len(files)} images")
        
        for i, file in enumerate(files):
            if not file.filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File at index {i} has no filename"
                )
            
            filename = secure_filename(file.filename)
            filename = f"{timestamp}_{i:03d}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            await _save_upload_file(file, filepath)
            saved_paths.append(filepath)

        result = api.hybrid_predict(saved_paths)

        # Cleanup
        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)

        logger.info(f"Hybrid prediction successful: {result['forecast_horizon']} steps forecasted")
        return JSONResponse(result)

    except HTTPException:
        # Cleanup on validation error
        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)
        raise
        
    except Exception as e:
        # Cleanup on processing error
        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)
        logger.error(f"Error in hybrid prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid prediction failed: {str(e)}"
        )


@app.get(
    '/health',
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check",
    description="Check API health status and model availability."
)
async def health_check():
    """Health check endpoint for monitoring and load balancers"""
    model_status = {
        'cnn_loaded': api.nowcast_model is not None,
        'lstm_loaded': api.forecast_model is not None,
        'hybrid_loaded': api.hybrid_model is not None
    }
    
    # Determine overall health
    all_models_loaded = all(model_status.values())
    health_status = 'healthy' if all_models_loaded else 'degraded'
    
    return JSONResponse({
        'status': health_status,
        'models': model_status,
        'device': str(api.device),
        'timestamp': datetime.now().isoformat(),
        'version': getattr(settings, 'APP_VERSION', '1.0.0')
    })


@app.get(
    '/api/training-data',
    tags=["Monitoring"],
    summary="Get training history",
    description="Retrieve training history and metrics from log files."
)
async def training_data():
    """Get training history and performance metrics"""
    try:
        logs_dir = getattr(settings, 'LOG_DIR', 'logs')
        result = {}
        
        if not os.path.isdir(logs_dir):
            logger.warning(f"Logs directory not found: {logs_dir}")
            return JSONResponse(result)

        for fname in os.listdir(logs_dir):
            if fname.endswith('.json'):
                path = os.path.join(logs_dir, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        result[fname] = json.load(fh)
                    logger.debug(f"Loaded training data from {fname}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {path}: {e}")
                    result[fname] = {"error": "Invalid JSON format"}
                except Exception as e:
                    logger.warning(f"Could not read {path}: {e}")
                    result[fname] = {"error": str(e)}

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Error reading training data logs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve training data: {str(e)}"
        )


# ==================== GEMINI AI ENDPOINTS ====================

@app.post(
    '/ai/explain',
    response_model=AIResponse,
    tags=["AI Insights"],
    summary="Get AI explanation of prediction",
    description="Generate natural language explanation of prediction results using Gemini AI."
)
async def ai_explain_prediction(req: AIExplanationRequest):
    """AI endpoint: Explain prediction results in natural language"""
    if not gemini_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini AI service not available. Check GEMINI_API_KEY configuration."
        )
    
    try:
        logger.info(f"Generating AI explanation for prediction")
        explanation = gemini_service.explain_prediction(req.prediction_data)
        
        return JSONResponse({
            'content': explanation,
            'timestamp': datetime.now().isoformat(),
            'model': getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')
        })
    
    except Exception as e:
        logger.error(f"Error generating explanation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate explanation: {str(e)}"
        )


@app.post(
    '/ai/recommend',
    response_model=AIResponse,
    tags=["AI Insights"],
    summary="Get AI recommendations",
    description="Generate actionable recommendations based on prediction data using Gemini AI."
)
async def ai_get_recommendations(req: AIRecommendationRequest):
    """AI endpoint: Get actionable recommendations based on predictions"""
    if not gemini_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini AI service not available. Check GEMINI_API_KEY configuration."
        )
    
    try:
        logger.info(f"Generating AI recommendations")
        recommendations = gemini_service.get_recommendations(
            req.prediction_data,
            req.user_context
        )
        
        return JSONResponse({
            'content': recommendations,
            'timestamp': datetime.now().isoformat(),
            'model': getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')
        })
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.post(
    '/ai/analyze-trends',
    response_model=AIResponse,
    tags=["AI Insights"],
    summary="Analyze irradiance trends",
    description="Analyze historical and forecast trends using Gemini AI to identify patterns."
)
async def ai_analyze_trends(req: AITrendAnalysisRequest):
    """AI endpoint: Analyze trends in irradiance data"""
    if not gemini_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini AI service not available. Check GEMINI_API_KEY configuration."
        )
    
    try:
        logger.info(f"Analyzing trends with AI")
        analysis = gemini_service.analyze_trends(
            req.historical_data,
            req.forecast_data
        )
        
        return JSONResponse({
            'content': analysis,
            'timestamp': datetime.now().isoformat(),
            'model': getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')
        })
    
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze trends: {str(e)}"
        )


@app.post(
    '/ai/ask',
    response_model=AIResponse,
    tags=["AI Insights"],
    summary="Ask AI about solar forecasting",
    description="Ask questions about solar irradiance, forecasting, or the predictions using Gemini AI."
)
async def ai_answer_question(req: AIQuestionRequest):
    """AI endpoint: Answer user questions about solar forecasting"""
    if not gemini_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini AI service not available. Check GEMINI_API_KEY configuration."
        )
    
    try:
        logger.info(f"Answering question with AI: {req.question[:50]}...")
        answer = gemini_service.answer_question(
            req.question,
            req.context_data
        )
        
        return JSONResponse({
            'content': answer,
            'timestamp': datetime.now().isoformat(),
            'model': getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')
        })
    
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}"
        )


@app.post(
    '/ai/smart-predict',
    tags=["AI Insights"],
    summary="Get prediction with AI insights",
    description="Combine nowcasting prediction with AI-powered explanation and recommendations."
)
async def ai_smart_predict(file: UploadFile = File(..., description="Infrared sky image")):
    """Smart prediction: Get nowcast with AI explanation and recommendations"""
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded or filename is empty"
        )

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        await _save_upload_file(file, filepath)
        logger.info(f"Processing smart prediction with AI: {filename}")
        
        # Get prediction
        prediction = api.nowcast_single_image(filepath)
        
        # Generate AI insights if available
        ai_insights = {}
        if gemini_service:
            try:
                explanation = gemini_service.explain_prediction(prediction)
                recommendations = gemini_service.get_recommendations(prediction)
                ai_insights = {
                    'explanation': explanation,
                    'recommendations': recommendations
                }
            except Exception as e:
                logger.warning(f"AI insights generation failed: {e}")
                ai_insights = {
                    'explanation': 'AI explanation not available',
                    'recommendations': 'AI recommendations not available'
                }
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
        
        result = {
            'prediction': prediction,
            'ai_insights': ai_insights,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Smart prediction successful: {prediction['nowcast_irradiance']} W/m²")
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"Error in smart prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Smart prediction failed: {str(e)}"
        )


if __name__ == '__main__':
    import uvicorn
    
    host = getattr(settings, 'HOST', '127.0.0.1')
    port = getattr(settings, 'PORT', 5000)
    debug = getattr(settings, 'DEBUG', False)
    
    logger.info(f"Starting Solar Forecasting API on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"API documentation available at: http://{host}:{port}/docs")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
