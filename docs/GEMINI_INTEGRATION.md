# Gemini AI Integration Guide

## Overview
This project integrates Google's Gemini AI to provide intelligent insights, explanations, and recommendations for solar irradiance forecasting predictions.

## Features

### 🤖 AI-Powered Endpoints

1. **Explain Predictions** (`POST /ai/explain`)
   - Natural language explanations of prediction results
   - Interprets irradiance levels and their implications
   - Contextualizes predictions for better understanding

2. **Get Recommendations** (`POST /ai/recommend`)
   - Actionable recommendations based on predictions
   - Energy management strategies
   - Optimal timing suggestions for high-power activities

3. **Analyze Trends** (`POST /ai/analyze-trends`)
   - Pattern identification in historical data
   - Variability assessment
   - Comparison of historical vs forecast trends

4. **Ask Questions** (`POST /ai/ask`)
   - Interactive Q&A about solar forecasting
   - Context-aware responses using current predictions
   - Technical explanations in accessible language

5. **Smart Predict** (`POST /ai/smart-predict`)
   - Combined nowcasting with AI insights
   - Single endpoint for prediction + explanation + recommendations

## Setup

### 1. Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

### 2. Configure Environment

Create a `.env` file in the project root (copy from `.env.example`):

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your API key
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=2048
GEMINI_TIMEOUT=30
```

### 3. Install Dependencies

```bash
pip install -r requirements-fastapi.txt
# or
pip install google-generativeai
```

## Usage Examples

### 1. Explain a Prediction

```python
import requests

# First, get a prediction
with open('sky_image.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )
prediction = response.json()

# Then, get AI explanation
response = requests.post(
    'http://localhost:5000/ai/explain',
    json={'prediction_data': prediction}
)
explanation = response.json()
print(explanation['content'])
```

### 2. Get Recommendations

```python
import requests

# Get forecast
response = requests.post(
    'http://localhost:5000/forecast',
    json={
        'irradiance_sequence': [450.2, 478.5, 490.1, 502.3, 515.7, 
                                530.2, 545.8, 560.1, 575.3, 590.5]
    }
)
forecast = response.json()

# Get AI recommendations with context
response = requests.post(
    'http://localhost:5000/ai/recommend',
    json={
        'prediction_data': forecast,
        'user_context': 'residential solar system'
    }
)
recommendations = response.json()
print(recommendations['content'])
```

### 3. Analyze Trends

```python
import requests

response = requests.post(
    'http://localhost:5000/ai/analyze-trends',
    json={
        'historical_data': [450, 475, 500, 520, 510, 490, 470, 450],
        'forecast_data': [430, 410, 390, 370]
    }
)
analysis = response.json()
print(analysis['content'])
```

### 4. Ask Questions

```python
import requests

response = requests.post(
    'http://localhost:5000/ai/ask',
    json={
        'question': 'What does an irradiance value of 500 W/m² mean for my solar panels?',
        'context_data': {'current_irradiance': 500}
    }
)
answer = response.json()
print(answer['content'])
```

### 5. Smart Predict (All-in-One)

```python
import requests

with open('sky_image.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/ai/smart-predict',
        files={'file': f}
    )

result = response.json()
print("Prediction:", result['prediction'])
print("\nExplanation:", result['ai_insights']['explanation'])
print("\nRecommendations:", result['ai_insights']['recommendations'])
```

## API Documentation

Once the server is running, visit:
- Interactive API docs: http://localhost:5000/docs
- Alternative docs: http://localhost:5000/redoc

## Model Selection

### gemini-1.5-flash (Recommended)
- **Speed**: Fast response times
- **Cost**: Lower cost per request
- **Use case**: Real-time predictions, frequent queries
- **Quality**: Good for most explanations and recommendations

### gemini-1.5-pro
- **Speed**: Slower than flash
- **Cost**: Higher cost per request
- **Use case**: Complex analysis, detailed reports
- **Quality**: Superior reasoning and nuanced insights

Configure in `.env`:
```bash
GEMINI_MODEL=gemini-1.5-flash  # or gemini-1.5-pro
```

## Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `GEMINI_TEMPERATURE` | Controls randomness in responses | 0.7 | 0.0-1.0 |
| `GEMINI_MAX_TOKENS` | Maximum response length | 2048 | 1-8192 |
| `GEMINI_TIMEOUT` | API request timeout (seconds) | 30 | 10-60 |

**Temperature Guidelines:**
- `0.0-0.3`: Very deterministic, factual responses
- `0.4-0.7`: Balanced creativity and consistency
- `0.8-1.0`: More creative, varied responses

## Error Handling

### Common Issues

**1. "Gemini AI service not available"**
- Check if `GEMINI_API_KEY` is set in `.env`
- Verify the API key is valid
- Ensure `google-generativeai` is installed

**2. Rate Limiting**
- Gemini free tier has rate limits
- Implement request throttling for production
- Consider caching frequent queries

**3. Token Limits**
- Reduce `GEMINI_MAX_TOKENS` if responses are too long
- Split large analysis requests into smaller chunks

## Best Practices

### 1. API Key Security
```python
# ✅ Good: Use environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# ❌ Bad: Hardcode API keys
GEMINI_API_KEY = 'AIza...'  # Never do this!
```

### 2. Caching Responses
```python
# Cache similar requests to reduce API calls
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_explanation(prediction_hash):
    return gemini_service.explain_prediction(prediction_data)
```

### 3. Error Handling
```python
try:
    response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    logger.error(f"AI request failed: {e}")
    # Provide fallback response
```

## Cost Estimation

### Gemini API Pricing (as of Jan 2026)

**Free Tier:**
- 15 requests per minute
- 1 million tokens per day
- Rate limits may apply

**Estimated Tokens per Request:**
- Simple explanation: ~200-300 tokens
- Recommendations: ~250-400 tokens
- Trend analysis: ~300-500 tokens
- Q&A: ~150-300 tokens

**Daily Usage Example:**
- 100 predictions/day with explanations
- ~25,000 tokens/day
- Well within free tier limits

## Testing

### Test AI Endpoints

```bash
# Start the server
python Fast_api_app.py

# Test health check
curl http://localhost:5000/health

# Test AI explanation
curl -X POST http://localhost:5000/ai/explain \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_data": {
      "nowcast_irradiance": 456.7,
      "timestamp": "2026-01-11T12:00:00",
      "model": "CNN_Regression"
    }
  }'
```

## Troubleshooting

### Enable Debug Logging

```python
# In Fast_api_app.py or .env
LOG_LEVEL=DEBUG
```

### Check Gemini Service Status

```python
# Test in Python
from scripts.gemini_service import get_gemini_service

service = get_gemini_service(api_key="your_key")
if service:
    result = service.answer_question("Test question")
    print(result)
```

### Verify API Key

```bash
# Test with curl
curl -H "Content-Type: application/json" \
     -H "x-goog-api-key: YOUR_API_KEY" \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
     https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent
```

## Advanced Features

### Custom Prompts

Modify `scripts/gemini_service.py` to customize prompts:

```python
def explain_prediction(self, prediction_data: Dict[str, Any]) -> str:
    # Add your custom prompt engineering here
    custom_context = "Focus on technical details for expert users"
    prompt = f"{custom_context}\n\n{standard_prompt}"
    # ...
```

### Multi-Language Support

```python
# Add language parameter to requests
response = requests.post(
    'http://localhost:5000/ai/explain',
    json={
        'prediction_data': prediction,
        'language': 'es'  # Spanish
    }
)
```

## Support

- **Issues**: Report bugs on GitHub
- **Documentation**: See main README.md
- **Gemini Docs**: https://ai.google.dev/docs

## License

This integration follows the main project license. Gemini API usage is subject to Google's terms of service.
