import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import pandas as pd
import io
import matplotlib.pyplot as plt
import os

# Set page configuration for better mobile experience
st.set_page_config(
    page_title="Food Nutrition Analyzer",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# App title and description
st.title("Food Nutrition Analyzer")
st.write("Take a photo or upload a food image to get nutritional information")

# Create a sidebar for options
st.sidebar.header("Settings")
portion_size = st.sidebar.selectbox("Portion Size", ["small", "medium", "large"], index=1)

HUGGINGFACE_TOKEN = "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"
# Then modify your load_model function
@st.cache_resource
@st.cache_resource
def load_model():
    """Load pretrained Food101 model from Hugging Face"""
    try:
        model_name = "nateraw/food"  # Food101 specialized model
        
        # Use the hardcoded token for authentication
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, 
            token=HUGGINGFACE_TOKEN
        )
        
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            token=HUGGINGFACE_TOKEN
        )
        
        model.eval()
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data
def load_nutrition_data(file_path):
    """Load nutrition data from CSV and organize by food class"""
    try:
        # Read the data file
        nutrition_df = pd.read_csv(file_path)
        
        # Create a dictionary to store nutrition information by food label
        nutrition_map = {}
        
        # Group by food label and create entries for each food
        for label, group in nutrition_df.groupby('label'):
            # Convert numeric columns to appropriate types
            for col in ['weight', 'calories', 'protein', 'carbohydrates', 
                        'fats', 'fiber', 'sugars', 'sodium']:
                group[col] = pd.to_numeric(group[col], errors='coerce')
            
            # Store all portion sizes for each food
            nutrition_map[label] = group.to_dict('records')
        
        return nutrition_map
    except Exception as e:
        st.error(f"Error loading nutrition data: {e}")
        return {}

def predict_food_and_nutrition(image, model, feature_extractor, nutrition_map, portion_size='medium'):
    """Predict food class and map to nutritional information"""
    # Preprocess image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model = model.to('cuda')

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    predicted_class_idx = torch.argmax(probabilities).item()
    
    # Food101 class names
    class_names = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
        "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
        "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
        "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
        "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
        "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
        "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
        "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
        "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
        "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
        "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
        "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
        "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
        "mussels", "nachos", "omelette", "onion_rings", "oysters",
        "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
        "pho", "pizza", "pork_chop", "poutine", "prime_rib",
        "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
        "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
        "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
        "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
        "waffles"
    ]
    
    # Get food name and confidence
    predicted_food = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item()
    
    # Get nutritional information
    nutrition_info = nutrition_map.get(predicted_food, None)
    
    # Handle portion size selection for the nutrition data
    if nutrition_info is None:
        nutrition_result = {
            "message": "Nutritional information not available for this food"
        }
    else:
        # If multiple portion sizes are available, select based on preference
        if isinstance(nutrition_info, list) and len(nutrition_info) > 0:
            if portion_size == 'small':
                nutrition_result = nutrition_info[0]  # smallest portion
            elif portion_size == 'large':
                nutrition_result = nutrition_info[-1]  # largest portion
            else:  # medium (default)
                mid_idx = len(nutrition_info) // 2
                nutrition_result = nutrition_info[mid_idx]
            
            # Add information about available portion sizes
            available_portions = [item['weight'] for item in nutrition_info]
            nutrition_result['available_portions'] = available_portions
        else:
            nutrition_result = nutrition_info
    
    return {
        "food_name": predicted_food,
        "confidence": confidence,
        "nutrition": nutrition_result
    }

def process_image(image, model, feature_extractor, nutrition_map, portion_size):
    """Process image and return prediction results"""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Get prediction
    result = predict_food_and_nutrition(image, model, feature_extractor, 
                                       nutrition_map, portion_size)
    return result

def display_results(result, image):
    """Display prediction results in a user-friendly format"""
    # Display image and prediction
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption=f"Predicted: {result['food_name']}", use_column_width=True)
    
    with col2:
        st.subheader(f"Food: {result['food_name']}")
        st.progress(result['confidence'])
        st.write(f"Confidence: {result['confidence']:.2f}")
        
        # Display nutrition information
        st.subheader("Nutrition Information")
        if "message" in result['nutrition']:
            st.info(result['nutrition']['message'])
        else:
            # Create a nicely formatted nutrition card
            nutrition_card = f"""
            <div style="background-color:#e2e2e2;padding:20px;border-radius:10px;color:white;">
                <h3 style="color:white;">Nutrition Facts</h3>
                <hr style="border-top:1px solid white;">
                <p style="color:white;"><strong style="color:white;">Portion size:</strong> {result['nutrition']['weight']}g</p>
                <p style="color:white;"><strong style="color:white;">Calories:</strong> {result['nutrition']['calories']} kcal</p>
                <hr style="border-top:1px dashed white;">
                <p style="color:white;"><strong style="color:white;">Protein:</strong> {result['nutrition']['protein']}g</p>
                <p style="color:white;"><strong style="color:white;">Carbs:</strong> {result['nutrition']['carbohydrates']}g</p>
                <p style="color:white;"><strong style="color:white;">Fats:</strong> {result['nutrition']['fats']}g</p>
                <p style="color:white;"><strong style="color:white;">Fiber:</strong> {result['nutrition']['fiber']}g</p>
                <p style="color:white;"><strong style="color:white;">Sugars:</strong> {result['nutrition']['sugars']}g</p>
                <p style="color:white;"><strong style="color:white;">Sodium:</strong> {result['nutrition']['sodium']}mg</p>
            </div>
            """
            st.markdown(nutrition_card, unsafe_allow_html=True)

# Main application flow
def main():
    # Load model and nutrition data
    with st.spinner("Loading model and nutrition data..."):
        model, feature_extractor = load_model()
        nutrition_map = load_nutrition_data("nutrition.csv")
    
    # Check if model loaded correctly
    if model is None or feature_extractor is None:
        st.error("Failed to load the model. Please refresh and try again.")
        return
    
    # Tabs for camera or upload options
    tab1, tab2 = st.tabs(["Take Photo 📸", "Upload Image 📁"])
    
    # Camera input tab
    with tab1:
        st.subheader("Use Camera")
        st.write("Click the button below to access your device camera")
        
        # Camera input widget
        img_file_camera = st.camera_input("Take a picture of food")
        
        if img_file_camera is not None:
            with st.spinner("Analyzing food..."):
                try:
                    # Process the image
                    image = Image.open(img_file_camera)
                    result = process_image(image, model, feature_extractor, nutrition_map, portion_size)
                    
                    # Display results
                    display_results(result, image)
                except Exception as e:
                    st.error(f"Error processing image: {e}")
    
    # File upload tab
    with tab2:
        st.subheader("Upload Food Image")
        
        # File uploader widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            with st.spinner("Analyzing food..."):
                try:
                    # Process the image
                    image = Image.open(uploaded_file)
                    result = process_image(image, model, feature_extractor, nutrition_map, portion_size)
                    
                    # Display results
                    display_results(result, image)
                except Exception as e:
                    st.error(f"Error processing image: {e}")
    
    # Help section for mobile users
    with st.expander("Having trouble with the camera or upload?"):
        st.write("""
        ### Troubleshooting Tips:
        
        **For camera issues:**
        - Make sure you've granted camera permissions to your browser
        - Try using Chrome on Android or Safari on iOS devices
        - If the camera isn't working, try the Upload Image option instead
        
        **For upload issues:**
        - Make sure your image is in JPG, JPEG, or PNG format
        - Check that your image isn't too large (try under 5MB)
        - If you're on iOS, you may need to select "Choose" rather than "Take Photo"
        
        **About the app:**
        This app uses a machine learning model trained on the Food101 dataset to identify common food items and provide nutritional information.
        """)

if __name__ == "__main__":
    main()
