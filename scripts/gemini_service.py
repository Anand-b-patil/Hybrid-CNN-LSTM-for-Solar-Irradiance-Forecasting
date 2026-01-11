"""
Gemini AI Integration Service for Solar Irradiance Forecasting
Provides AI-powered insights, explanations, and recommendations
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old package if new one not available
        import google.generativeai as genai
        types = None
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False
        genai = None
        types = None

logger = logging.getLogger(__name__)


class GeminiAIService:
    """Service class for integrating Google Gemini AI with solar forecasting"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp", 
                 temperature: float = 0.7, max_tokens: int = 2048):
        """
        Initialize Gemini AI Service
        
        Args:
            api_key: Google AI API key
            model_name: Gemini model to use (gemini-2.0-flash-exp, gemini-1.5-flash, gemini-1.5-pro)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Genai package not installed. "
                "Install with: pip install google-genai"
            )
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not provided. "
                "Set it in .env file or as environment variable"
            )
        
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_new_api = types is not None
        
        # Configure Gemini - use new API if available
        if self.use_new_api:
            # New google.genai API
            self.client = genai.Client(api_key=api_key)
            logger.info(f"Using new google.genai API with model: {model_name}")
        else:
            # Old google.generativeai API (deprecated)
            genai.configure(api_key=api_key)
            
            # Initialize model with safety settings for old API
            self.generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
            }
            
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            logger.warning("Using deprecated google.generativeai API - consider upgrading to google.genai")
        
        logger.info(f"Gemini AI Service initialized with model: {model_name}")
    
    def _generate_content(self, prompt: str) -> str:
        """
        Generate content using appropriate API
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Generated text response
        """
        if self.use_new_api:
            # Use new google.genai API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=0.95,
                    top_k=40
                )
            )
            return response.text.strip()
        else:
            # Use old google.generativeai API
            response = self.model.generate_content(prompt)
            return response.text.strip()
    
    def _create_system_context(self) -> str:
        """Create system context for solar forecasting domain"""
        return """You are an AI assistant specialized in solar irradiance forecasting and renewable energy.
Your expertise includes:
- Understanding solar radiation patterns and atmospheric conditions
- Analyzing weather impacts on solar energy production
- Interpreting machine learning predictions for solar forecasting
- Providing actionable insights for solar energy management
- Explaining technical concepts in accessible language

Context: This system uses Hybrid CNN-LSTM models combining EfficientNet-B0 for sky image analysis 
with BiLSTM for time-series forecasting to predict solar irradiance (W/m²)."""
    
    def explain_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """
        Generate AI explanation for prediction results
        
        Args:
            prediction_data: Dictionary containing prediction results
            
        Returns:
            AI-generated explanation
        """
        try:
            context = self._create_system_context()
            
            # Format prediction data
            pred_type = prediction_data.get('model', 'Unknown')
            
            if 'nowcast_irradiance' in prediction_data:
                value = prediction_data['nowcast_irradiance']
                prompt = f"""{context}

Task: Explain this solar irradiance nowcasting prediction to a user.

Prediction Details:
- Model: {pred_type}
- Current Predicted Irradiance: {value} W/m²
- Timestamp: {prediction_data.get('timestamp', 'N/A')}

Provide a concise, user-friendly explanation that:
1. Interprets what this irradiance level means
2. Explains the expected solar energy generation potential
3. Mentions any practical implications (e.g., optimal for solar panels, cloudy conditions, etc.)
4. Keeps it under 150 words

Be clear, actionable, and avoid excessive technical jargon."""
            
            elif 'forecast_irradiance' in prediction_data:
                values = prediction_data['forecast_irradiance']
                horizon = prediction_data.get('forecast_horizon', len(values))
                prompt = f"""{context}

Task: Explain this solar irradiance forecast to a user.

Forecast Details:
- Model: {pred_type}
- Predicted Values: {values} W/m²
- Forecast Horizon: {horizon} time steps
- Timestamp: {prediction_data.get('timestamp', 'N/A')}

Provide a concise explanation that:
1. Summarizes the trend (increasing, decreasing, stable)
2. Highlights any significant changes
3. Explains practical implications for solar energy planning
4. Keeps it under 150 words

Be clear and actionable."""
            
            elif 'nowcast_sequence' in prediction_data:
                nowcasts = prediction_data['nowcast_sequence']
                forecasts = prediction_data['forecast_irradiance']
                prompt = f"""{context}

Task: Explain this hybrid nowcast/forecast prediction.

Hybrid Prediction Details:
- Model: {pred_type}
- Recent Nowcast Values: {nowcasts} W/m²
- Future Forecast: {forecasts} W/m²
- Sequence Length: {prediction_data.get('sequence_length', len(nowcasts))}
- Forecast Horizon: {prediction_data.get('forecast_horizon', len(forecasts))}

Provide a concise explanation that:
1. Summarizes the current pattern and future trend
2. Identifies any important transitions
3. Provides actionable insights for energy management
4. Keeps it under 150 words"""
            
            else:
                return "Unable to generate explanation: Invalid prediction data format"
            
            return self._generate_content(prompt)
        
        except Exception as e:
            logger.error(f"Error generating prediction explanation: {e}")
            return f"Error generating explanation: {str(e)}"
    
    def get_recommendations(self, prediction_data: Dict[str, Any], 
                           user_context: Optional[str] = None) -> str:
        """
        Generate AI recommendations based on predictions
        
        Args:
            prediction_data: Dictionary containing prediction results
            user_context: Optional context about user's application (e.g., "solar farm", "residential")
            
        Returns:
            AI-generated recommendations
        """
        try:
            context = self._create_system_context()
            user_info = f"\nUser Context: {user_context}" if user_context else ""
            
            prompt = f"""{context}{user_info}

Task: Provide actionable recommendations based on this solar irradiance prediction.

Prediction Data:
{json.dumps(prediction_data, indent=2)}

Generate specific, actionable recommendations that:
1. Address energy management strategies
2. Suggest optimal timing for high-power activities
3. Mention battery storage considerations if relevant
4. Provide weather-aware operational guidance
5. Keep the response practical and under 200 words

Focus on what the user should DO with this information."""
            
            return self._generate_content(prompt)
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return f"Error generating recommendations: {str(e)}"
    
    def analyze_trends(self, historical_data: List[float], 
                       forecast_data: Optional[List[float]] = None) -> str:
        """
        Analyze historical trends and patterns
        
        Args:
            historical_data: List of historical irradiance values
            forecast_data: Optional list of forecasted values
            
        Returns:
            AI-generated trend analysis
        """
        try:
            context = self._create_system_context()
            
            # Calculate basic statistics
            avg = sum(historical_data) / len(historical_data)
            max_val = max(historical_data)
            min_val = min(historical_data)
            
            forecast_info = ""
            if forecast_data:
                forecast_avg = sum(forecast_data) / len(forecast_data)
                forecast_info = f"\nForecast Average: {forecast_avg:.2f} W/m²\nForecast Values: {forecast_data}"
            
            prompt = f"""{context}

Task: Analyze these solar irradiance patterns and provide insights.

Historical Data Analysis:
- Values: {historical_data[-20:]} W/m² (last 20 points)
- Average: {avg:.2f} W/m²
- Maximum: {max_val:.2f} W/m²
- Minimum: {min_val:.2f} W/m²
- Data Points: {len(historical_data)}{forecast_info}

Provide analysis including:
1. Pattern identification (diurnal cycles, weather impacts, trends)
2. Variability assessment
3. Comparison of historical vs forecast (if available)
4. Practical insights for solar energy systems
5. Keep under 200 words

Be specific and data-driven."""
            
            return self._generate_content(prompt)
        
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return f"Error analyzing trends: {str(e)}"
    
    def answer_question(self, question: str, context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Answer user questions about solar forecasting
        
        Args:
            question: User's question
            context_data: Optional context data (predictions, historical data, etc.)
            
        Returns:
            AI-generated answer
        """
        try:
            system_context = self._create_system_context()
            
            data_context = ""
            if context_data:
                data_context = f"\n\nCurrent System Data:\n{json.dumps(context_data, indent=2)}"
            
            prompt = f"""{system_context}{data_context}

User Question: {question}

Provide a clear, accurate answer that:
1. Directly addresses the question
2. Uses the context data if relevant
3. Provides technical accuracy while remaining accessible
4. Includes specific examples when helpful
5. Keeps response under 250 words

Be helpful and informative."""
            
            return self._generate_content(prompt)
        
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error answering question: {str(e)}"
    
    def generate_report_summary(self, predictions: List[Dict[str, Any]], 
                               time_range: str = "today") -> str:
        """
        Generate a summary report of multiple predictions
        
        Args:
            predictions: List of prediction dictionaries
            time_range: Time range description (e.g., "today", "this week")
            
        Returns:
            AI-generated summary report
        """
        try:
            context = self._create_system_context()
            
            prompt = f"""{context}

Task: Create a comprehensive summary report of solar irradiance predictions.

Time Range: {time_range}
Number of Predictions: {len(predictions)}

Prediction Data:
{json.dumps(predictions[-10:], indent=2)}  # Last 10 predictions

Generate a professional summary report that includes:
1. Overall patterns and trends
2. Key statistics (averages, peaks, variability)
3. Notable events or anomalies
4. Performance implications for solar systems
5. Forward-looking insights
6. Keep under 300 words

Format as a structured report suitable for stakeholders."""
            
            return self._generate_content(prompt)
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"


# Singleton instance management
_gemini_service_instance: Optional[GeminiAIService] = None


def get_gemini_service(api_key: str, model_name: str = "gemini-2.0-flash-exp",
                       temperature: float = 0.7, max_tokens: int = 2048) -> Optional[GeminiAIService]:
    """
    Get or create Gemini AI service instance
    
    Args:
        api_key: Google AI API key
        model_name: Gemini model name
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        
    Returns:
        GeminiAIService instance or None if initialization fails
    """
    global _gemini_service_instance
    
    if _gemini_service_instance is None:
        try:
            _gemini_service_instance = GeminiAIService(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini service: {e}")
            return None
    
    return _gemini_service_instance
