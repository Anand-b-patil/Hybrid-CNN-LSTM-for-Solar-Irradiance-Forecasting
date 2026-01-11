"""
Quick test script to verify Gemini AI integration
Run this to ensure everything is set up correctly
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if required packages are installed"""
    print("Testing imports...")
    try:
        import google.generativeai as genai
        print("✅ google-generativeai installed")
        return True
    except ImportError:
        print("❌ google-generativeai not installed")
        print("   Install with: pip install google-generativeai")
        return False


def test_api_key():
    """Test if API key is configured"""
    print("\nTesting API key configuration...")
    
    # Try to load from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️  python-dotenv not installed (optional)")
    
    api_key = os.getenv('GEMINI_API_KEY')
    
    if api_key and api_key != 'your_gemini_api_key_here':
        print(f"✅ GEMINI_API_KEY configured (length: {len(api_key)})")
        return True
    else:
        print("❌ GEMINI_API_KEY not configured")
        print("   1. Copy .env.example to .env")
        print("   2. Get API key from: https://makersuite.google.com/app/apikey")
        print("   3. Set GEMINI_API_KEY in .env file")
        return False


def test_gemini_service():
    """Test if Gemini service can be initialized"""
    print("\nTesting Gemini service initialization...")
    
    try:
        from scripts.gemini_service import get_gemini_service
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your_gemini_api_key_here':
            print("⚠️  Skipping service test (no API key)")
            return False
        
        service = get_gemini_service(api_key=api_key)
        
        if service:
            print("✅ Gemini service initialized successfully")
            return True
        else:
            print("❌ Gemini service initialization failed")
            return False
    
    except Exception as e:
        print(f"❌ Error initializing service: {e}")
        return False


def test_simple_query():
    """Test a simple AI query"""
    print("\nTesting simple AI query...")
    
    try:
        from scripts.gemini_service import get_gemini_service
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your_gemini_api_key_here':
            print("⚠️  Skipping query test (no API key)")
            return False
        
        service = get_gemini_service(api_key=api_key)
        
        if not service:
            print("❌ Service not available")
            return False
        
        # Test with a simple question
        question = "What is solar irradiance?"
        print(f"   Question: {question}")
        
        answer = service.answer_question(question)
        
        if answer and len(answer) > 0 and 'Error' not in answer:
            print(f"✅ Query successful!")
            print(f"   Answer preview: {answer[:100]}...")
            return True
        else:
            print(f"❌ Query failed: {answer}")
            return False
    
    except Exception as e:
        print(f"❌ Error during query: {e}")
        return False


def test_config_file():
    """Test if config.py has Gemini settings"""
    print("\nTesting config.py...")
    
    try:
        from config import settings
        
        # Check for Gemini attributes
        has_key = hasattr(settings, 'GEMINI_API_KEY')
        has_model = hasattr(settings, 'GEMINI_MODEL')
        
        if has_key and has_model:
            print("✅ Config file has Gemini settings")
            print(f"   Model: {settings.GEMINI_MODEL}")
            return True
        else:
            print("❌ Config file missing Gemini settings")
            return False
    
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def main():
    """Run all tests"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Gemini AI Integration - Verification Test          ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    results = {
        'Imports': test_imports(),
        'Config File': test_config_file(),
        'API Key': test_api_key(),
        'Service Init': test_gemini_service(),
        'Simple Query': test_simple_query()
    }
    
    print("\n" + "="*60)
    print(" Test Results Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print("\n" + "="*60)
    print(f" Total: {passed_count}/{total_count} tests passed")
    print("="*60)
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Gemini AI integration is ready to use.")
        print("\n📚 Next steps:")
        print("   1. Start the API: python Fast_api_app.py")
        print("   2. Run examples: python examples/gemini_ai_demo.py")
        print("   3. Visit API docs: http://localhost:5000/docs")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\n📚 Troubleshooting:")
        print("   - See docs/GEMINI_INTEGRATION.md for setup instructions")
        print("   - Ensure GEMINI_API_KEY is set in .env file")
        print("   - Run: pip install -r requirements-fastapi.txt")


if __name__ == "__main__":
    main()
