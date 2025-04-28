from flask import Flask, jsonify
from app import app

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    # Implement logic to fetch the forecast
    # For now, just a placeholder response
    forecast = {"date": "2025-04-14", "forecast": "Sunny", "irradiance": "800 W/m2"}
    return jsonify(forecast)

if __name__ == "__main__":
    app.run(debug=True)
