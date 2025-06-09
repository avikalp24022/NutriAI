from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

def load_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash-lite')

def predict_food_and_nutrition(image, model):
    # Get full path to nutrition data
    nutrition_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Final_key_value_pair.json')
    
    with open(nutrition_file_path, 'r', encoding='utf-8') as file:
        nutrition_data = json.load(file)
    
    class_names = list(nutrition_data.keys())
    
    prompt = f"""
    You are a culinary expert specializing in Singaporean cuisine. Given an image of a food item, first carefully examine its visual appearance, ingredients, and likely cooking techniques to infer how it was prepared. Based on this analysis, classify the dish into one of the following categories:
    Food classes: [{', '.join(class_names)}]
    If you are confident that the dish is not listed above, set dishName to your most promising guess based on its appearance and ingredients.
    After identifying the most promising dish, recall the ingredients used in making that dish and list them in the response accordingly.
    In case of Nutrition information use average values. Note that the nutrition information must be according to the portion size of the dish.
    """
    
    prompt += """Respond as a JSON String in the following format and nothing else:
        {
        "dishName": "the food dish name",
        "Ingredients": [ /* list of ingredients recalled; empty if not food */ ],
        "Nutrients": {
            "Protein": ["Protein (g)", "Protein"],
            "Total Fat": ["Total fat (g)", "Total Fat"],
            "Carbohydrates": ["Carbohydrate (g)", "Carbohydrates"],
            "Calories": ["Energy (kcal)", "Calories"],
            "Sugars": ["Sugar (g)", "Sugars"],
            "Sodium": ["Sodium (mg)", "Sodium"],
            "Per Serving Household Measure": ["Per Serving Household Measure"],
            "Iron": ["Iron (mg)", "Iron"],
            "Vitamin A": ["Vitamin A (mcg)", "Vitamin A"],
            "Vitamin C": ["Vitamin C (mg)", "Vitamin C"],
            "Vitamin D": ["Vitamin D (IU)", "Vitamin D"]
            }
        }
        Do not include any additional text or explanation.
    """
    response = model.generate_content([prompt, image])
    json_str = response.text.strip().replace("```json", "").replace("```", "")
    food_data = json.loads(json_str)
    return food_data

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[-1])
        image = Image.open(io.BytesIO(image_bytes))
        model = load_model()
        food_data = predict_food_and_nutrition(image, model)
        return jsonify(food_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
