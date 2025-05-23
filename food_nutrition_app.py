import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io

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


# Then modify your load_model function
@st.cache_resource
def load_model():
    """Load and configure the Gemini Pro Vision model"""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-pro-vision')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


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

def predict_food_and_nutrition(image, model, nutrition_map, portion_size='medium'):
    """Predict food class and map to nutritional information using Gemini"""
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

    prompt = f"""
    Given the image of a food item, identify which of the following food classes it belongs to.
    Only return the single best match from this list. Do not add any extra text or explanation.

    Food classes: {', '.join(class_names)}
    """

    response = model.generate_content([prompt, image])
    predicted_food = response.text.strip().lower().replace(" ", "_")

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
        "food_name": predicted_food.replace("_", " ").title(),
        "nutrition": nutrition_result
    }

def process_image(image, model, nutrition_map, portion_size):
    """Process image and return prediction results"""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Get prediction
    result = predict_food_and_nutrition(image, model, nutrition_map, portion_size)
    return result

def display_results(result, image):
    """Display prediction results in a user-friendly format"""
    # Display image and prediction
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption=f"Predicted: {result['food_name']}", use_container_width=True)

    with col2:
        st.subheader(f"Food: {result['food_name']}")

        # Display nutrition information
        st.subheader("Nutrition Information")
        if "message" in result['nutrition']:
            st.info(result['nutrition']['message'])
        else:
            # Create a nicely formatted nutrition card
            nutrition_card = f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;color:black;">
                <h3 style="color:black;">Nutrition Facts</h3>
                <hr style="border-top:1px solid black;">
                <p><strong style="color:black;">Portion size:</strong> {result['nutrition']['weight']}g</p>
                <p><strong style="color:black;">Calories:</strong> {result['nutrition']['calories']} kcal</p>
                <hr style="border-top:1px dashed black;">
                <p><strong style="color:black;">Protein:</strong> {result['nutrition']['protein']}g</p>
                <p><strong style="color:black;">Carbs:</strong> {result['nutrition']['carbohydrates']}g</p>
                <p><strong style="color:black;">Fats:</strong> {result['nutrition']['fats']}g</p>
                <p><strong style="color:black;">Fiber:</strong> {result['nutrition']['fiber']}g</p>
                <p><strong style="color:black;">Sugars:</strong> {result['nutrition']['sugars']}g</p>
                <p><strong style="color:black;">Sodium:</strong> {result['nutrition']['sodium']}mg</p>
            </div>
            """
            st.markdown(nutrition_card, unsafe_allow_html=True)

# Main application flow
def main():
    """Main function to run the Streamlit app"""
    # Load model and nutrition data
    with st.spinner("Loading model and nutrition data..."):
        model = load_model()
        nutrition_map = load_nutrition_data("nutrition.csv")

    # Check if model loaded correctly
    if model is None:
        st.error("Failed to load the model. Please check your API key and refresh.")
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
                    result = process_image(image, model, nutrition_map, portion_size)

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
                    result = process_image(image, model, nutrition_map, portion_size)

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
        This app uses Google's Gemini Pro Vision model to identify common food items and provide nutritional information.
        """)

if __name__ == "__main__":
    main()
