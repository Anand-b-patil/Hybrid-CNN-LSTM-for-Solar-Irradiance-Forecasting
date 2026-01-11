"""
Example usage of Gemini AI integration with Solar Forecasting API

This script demonstrates how to use the AI-powered endpoints to get
intelligent insights about solar irradiance predictions.

Before running:
1. Set your GEMINI_API_KEY in .env file
2. Start the FastAPI server: python Fast_api_app.py
3. Run this script: python examples/gemini_ai_demo.py
"""

import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:5000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60 + "\n")


def example_1_smart_predict():
    """Example 1: Get prediction with AI explanation and recommendations"""
    print_section("Example 1: Smart Predict (All-in-One)")
    
    # Path to a sample infrared image
    # Replace with your actual image path
    image_path = Path("data/test/infrared/1547579790IR.png")
    
    if not image_path.exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   Please provide a valid infrared image path")
        return
    
    try:
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{BASE_URL}/ai/smart-predict",
                files={'file': ('image.png', f, 'image/png')},
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Smart Prediction Complete!\n")
            print(f"📊 Predicted Irradiance: {result['prediction']['nowcast_irradiance']} W/m²")
            print(f"\n🤖 AI Explanation:\n{result['ai_insights']['explanation']}")
            print(f"\n💡 AI Recommendations:\n{result['ai_insights']['recommendations']}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_explain_forecast():
    """Example 2: Get AI explanation of a forecast"""
    print_section("Example 2: Explain Forecast")
    
    # Sample irradiance sequence
    irradiance_sequence = [450.2, 478.5, 490.1, 502.3, 515.7, 530.2, 
                          545.8, 560.1, 575.3, 590.5, 600.2, 610.5,
                          620.3, 625.8, 630.1, 635.4, 640.2, 642.5,
                          645.1, 648.3]
    
    try:
        # First, get the forecast
        print("🔮 Getting forecast...")
        response = requests.post(
            f"{BASE_URL}/forecast",
            json={'irradiance_sequence': irradiance_sequence},
            timeout=30
        )
        
        if response.status_code == 200:
            forecast = response.json()
            print(f"✅ Forecast: {forecast['forecast_irradiance']}")
            
            # Now get AI explanation
            print("\n🤖 Getting AI explanation...")
            response = requests.post(
                f"{BASE_URL}/ai/explain",
                json={'prediction_data': forecast},
                timeout=30
            )
            
            if response.status_code == 200:
                explanation = response.json()
                print(f"\n{explanation['content']}")
            else:
                print(f"❌ Explanation error: {response.status_code}")
        else:
            print(f"❌ Forecast error: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def example_3_get_recommendations():
    """Example 3: Get AI recommendations with user context"""
    print_section("Example 3: AI Recommendations with Context")
    
    # Sample prediction data
    prediction_data = {
        'forecast_irradiance': [650.5, 680.2, 710.3, 735.1],
        'forecast_horizon': 4,
        'timestamp': '2026-01-11T12:00:00',
        'model': 'LSTM_Forecasting'
    }
    
    # User context
    user_contexts = [
        "residential solar system with 10kW capacity",
        "solar farm with battery storage",
        "commercial building with EV charging stations"
    ]
    
    try:
        for context in user_contexts:
            print(f"\n🏠 Context: {context}")
            print("-" * 60)
            
            response = requests.post(
                f"{BASE_URL}/ai/recommend",
                json={
                    'prediction_data': prediction_data,
                    'user_context': context
                },
                timeout=30
            )
            
            if response.status_code == 200:
                recommendations = response.json()
                print(recommendations['content'])
            else:
                print(f"❌ Error: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def example_4_analyze_trends():
    """Example 4: Analyze historical trends"""
    print_section("Example 4: Trend Analysis")
    
    # Historical data (morning to afternoon progression)
    historical_data = [
        50, 120, 230, 380, 490, 570, 620, 680, 710, 750,
        780, 790, 800, 795, 780, 750, 710, 650, 580, 480
    ]
    
    # Forecast (declining trend - evening)
    forecast_data = [380, 280, 180, 80]
    
    try:
        response = requests.post(
            f"{BASE_URL}/ai/analyze-trends",
            json={
                'historical_data': historical_data,
                'forecast_data': forecast_data
            },
            timeout=30
        )
        
        if response.status_code == 200:
            analysis = response.json()
            print(analysis['content'])
        else:
            print(f"❌ Error: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def example_5_ask_question():
    """Example 5: Ask AI questions about solar forecasting"""
    print_section("Example 5: Interactive Q&A")
    
    questions = [
        {
            'question': 'What does an irradiance value of 500 W/m² mean for my solar panels?',
            'context_data': {'current_irradiance': 500}
        },
        {
            'question': 'How does cloud cover affect solar irradiance predictions?',
            'context_data': None
        },
        {
            'question': 'When is the best time to charge my EV using solar power?',
            'context_data': {
                'forecast_irradiance': [600, 700, 800, 750],
                'ev_battery_capacity': 75
            }
        }
    ]
    
    try:
        for i, q in enumerate(questions, 1):
            print(f"\n❓ Question {i}: {q['question']}")
            print("-" * 60)
            
            response = requests.post(
                f"{BASE_URL}/ai/ask",
                json=q,
                timeout=30
            )
            
            if response.status_code == 200:
                answer = response.json()
                print(answer['content'])
            else:
                print(f"❌ Error: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def check_health():
    """Check API health and AI service availability"""
    print_section("Checking API Health")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API Status: {health['status']}")
            print(f"🖥️  Device: {health['device']}")
            print(f"📦 Version: {health['version']}")
            print("\n📊 Model Status:")
            for model, loaded in health['models'].items():
                status = "✅" if loaded else "❌"
                print(f"   {status} {model}: {'Loaded' if loaded else 'Not loaded'}")
            
            return health['status'] == 'healthy'
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("   Start with: python Fast_api_app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main function to run all examples"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Solar Forecasting API - Gemini AI Integration Demo      ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if API is running
    if not check_health():
        print("\n⚠️  Please start the API server first:")
        print("   python Fast_api_app.py")
        return
    
    print("\n🚀 Running AI Integration Examples...")
    
    # Run examples
    try:
        # example_1_smart_predict()  # Requires image file
        example_2_explain_forecast()
        example_3_get_recommendations()
        example_4_analyze_trends()
        example_5_ask_question()
        
        print_section("Demo Complete!")
        print("✅ All examples completed successfully!")
        print("\n📚 For more information, see:")
        print("   - docs/GEMINI_INTEGRATION.md")
        print("   - API Docs: http://localhost:5000/docs")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
