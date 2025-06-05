import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io
import json


# Configuration for the Streamlit page
PAGE_TITLE = "Food Nutrition Analyzer"
PAGE_ICON = "🍔"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "collapsed"

# Set the configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# Title of the Streamlit app
st.title(PAGE_TITLE)

# Descriptive text to inform the user what the app does
st.write("Take a photo or upload a food image to get nutritional information")


@st.cache_resource
def load_model():
    """
    Load and configure the Generative AI model.

    This function configures the Generative AI model using the Gemini API key
    stored in Streamlit secrets. It initializes the model and caches it to avoid
    reloading with each app interaction.

    Returns:
        An instance of the configured GenerativeModel if successful, otherwise None.
    """
    try:
        # Configure the generative AI with the Gemini API key
        genai.configure(api_key="AIzaSyD1XnJp9s3WFF0G7zvLcXk4gNIIYPcdzgY")
        # genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

        # Initialize and return the GenerativeModel
        return genai.GenerativeModel('gemini-2.0-flash-lite')
    except Exception as error:
        # Display an error message if model loading fails
        st.error(f"Error loading model: {error}")
        return None


@st.cache_data
def load_nutrition_data(nutrition_file_path: str) -> dict:
    """
    Load nutrition data from a JSON file.
    
    The JSON file is expected to be a dictionary where each key is a dish name and
    the value is another dictionary with the following fields:
        - B-Carotene (mcg)
        - Calcium (mg)
        - Carbohydrate (g)
        - Cholesterol (mg)
        - Dietary fibre (g)
        - Energy (kcal)
        - Iron (mg)
        - Monounsaturated fat (g)
        - Phosphorus (mg)
        - Polyunsaturated fat (g)
        - Potassium (mg)
        - Protein (g)
        - Retinol (mcg)
        - Riboflavin (mg)
        - Saturated fat (g)
        - Selenium (mcg)
        - Sodium (mg)
        - Starch (g)
        - Sugar (g)
        - Thiamin (mg)
        - Total fat (g)
        - Vitamin A (mcg)
        - Vitamin C (mg)
        - Vitamin D (mcg)
        - Water (g)
        - Whole-grains (g)
        - Zinc (mcg)
    
    The function returns a dictionary where the keys are the same as the dish
    names in the JSON file, and the values are dictionaries with the same fields as
    above but with numeric fields converted to float where possible.
    """

    try:
        # Read JSON file as a dict
        with open(nutrition_file_path, 'r', encoding='utf-8') as file:
            nutrition_data = json.load(file)

        # Convert numeric fields to float where possible
        numeric_fields = [
            'B-Carotene',
            'Calcium',
            'Carbohydrate',
            'Cholesterol',
            'Dietary fibre',
            'Energy',
            'Iron',
            'Monounsaturated fat',
            'Phosphorus',
            'Polyunsaturated fat',
            'Potassium',
            'Protein',
            'Retinol',
            'Riboflavin',
            'Saturated fat',
            'Selenium',
            'Sodium',
            'Starch',
            'Sugar',
            'Thiamin',
            'Total fat',
            'Vitamin A',
            'Vitamin C',
            'Vitamin D',
            'Water',
            'Whole-grains',
            'Zinc'
        ]

        nutrition_map = {}

        # Iterate over each dish in the JSON file
        for dish, entry in nutrition_data.items():
            # Copy data so original isn't mutated
            dish_data = entry.copy()

            # Iterate over each numeric field
            for field in numeric_fields:
                # Get the value of the field
                val = dish_data.get(field, None)

                # Convert the value to float if it's not None
                try:
                    dish_data[field] = float(val) if val not in [None, "", " "] else None
                except Exception:
                    # Handle parse error
                    dish_data[field] = None

            # Add the dish to the nutrition map
            nutrition_map[dish.lower().replace(" ", "_")] = dish_data

        # Return the nutrition map
        return nutrition_map

    except Exception as e:
        # Handle any errors when loading the nutrition data
        st.error(f"Error loading nutrition data: {e}")
        return {}


def predict_food_and_nutrition(image, model):
    """
    Predict food class from an image and map to nutritional information.
    
    Parameters:
    - image: Image of the food item.
    - model: Pre-trained model used for class prediction.
    - nutrition_map: Dictionary mapping food class names to nutritional data.
    
    Returns:
    - predicted_food: The predicted food class name.
    - nutrition_info: Nutritional information for the predicted food class.
    """
    # List of food class names in the dataset
    with open('Final_key_value_pair.json', 'r', encoding='utf-8') as file:
        nutrition_data = json.load(file)
    # Extract class names from the nutrition data
    class_names = list(nutrition_data.keys())
    # class_names = [
    #     "Ayam penyet with rice",
    #     "BBQ Turkey Bacon Double Cheeseburger, Burger King",
    #     "Bak kut teh",
    #     "Bak kut teh, soup only",
    #     "Baked durian mooncake",
    #     "Ban mian soup",
    #     "Banana fritter",
    #     "Beans broad, coated with satay powder, deep fried",
    #     "Beef ball kway teow soup",
    #     "Beef burger with cheese",
    #     "Beef noodles soup",
    #     "Beef rendang",
    #     "Beef satay, without satay sauce",
    #     "Beef, rendang with soya sauce, simmered (Malay)",
    #     "Biscuit, savoury, salada style",
    #     "Braised duck rice",
    #     "Braised duck with yam rice",
    #     "Braised pork ribs with black bean sauce",
    #     "Brown rice laksa noodles, cooked",
    #     "Brown rice porridge, plain",
    #     "Bulgogi Gimbap (Korean Rice Roll with Spicy Beef)",
    #     "Caesar salad",
    #     "Cafe 26 oriental salad dressing",
    #     "Cafe 26 tangy salad dressing",
    #     "Carrot cake with egg, plain, mashed & fried",
    #     "Cereal prawn",
    #     "Char kway teow",
    #     "Char siew chee cheong fun",
    #     "Chee cheong fun",
    #     "Chee pah",
    #     "Cheesy BBQ Meltz, KFC",
    #     "Chendol",
    #     "Chendol, durian",
    #     "Chendol, mango",
    #     "Chicken briyani",
    #     "Chicken curry noodles",
    #     "Chicken satay, without peanut sauce",
    #     "Chilli crab",
    #     "Chwee kueh",
    #     "Claypot rice with salted fish,chicken and chinese sausages",
    #     "Claypot rice, with mixed vegetable",
    #     "Claypot rice, with prawn",
    #     "Claypot rice, with stewed beef",
    #     "Coconut kueh tutu",
    #     "Curry fish head",
    #     "Curry puff, beef",
    #     "Curry puff, chicken",
    #     "Curry puff, frozen, deep fried",
    #     "Curry puff, potato and mutton filling, deep fried",
    #     "Curry puff, potato, and spices, deep fried",
    #     "Curry puff, twisted",
    #     "Deep fried carrot cake",
    #     "Deep fried fish bee hoon soup with milk",
    #     "Dim sum, beancurd roll",
    #     "Dim sum, chicken feet with dark sauce, stewed",
    #     "Dim sum, dumpling, chives with minced prawn, steamed",
    #     "Dim sum, dumpling, yam, deep fried",
    #     "Dim sum, pork ribs",
    #     "Dim sum, pork tart, BBQ",
    #     "Dim sum, sharkfin dumpling",
    #     "Dim sum, siew mai, steamed",
    #     "Dim sum, turnip cake, steamed",
    #     "Dim sum, you tiao",
    #     "Dodol berdurian",
    #     "Double Cheeseburger, McDonalds'",
    #     "Drunken prawn",
    #     "Dry prawn noodles",
    #     "Duck rice, with skin removed",
    #     "Durian",
    #     "Durian Pancake",
    #     "Durian cake",
    #     "Durian fermented",
    #     "Durian ice kacang",
    #     "Durian pudding",
    #     "Durian puff",
    #     "Durian wafer",
    #     "Durian, Malaysian, mid-range",
    #     "Durian, raw",
    #     "Fast foods, salad, vegetable, tossed, without dressing, with chicken",
    #     "Fish ball mee pok, dry",
    #     "Fish ball noodles dry",
    #     "Fish finger, grilled or baked",
    #     "Fish head ban mian soup",
    #     "Fish ngoh hiang",
    #     "Fish satay snack",
    #     "Fried mee siam",
    #     "Fried plain carrot cake",
    #     "Fried vegetarian bee hoon, plain",
    #     "Frog leg claypot rice",
    #     "Fruit salad, canned in heavy syrup",
    #     "Fruit salad, canned in heavy syrup, drained",
    #     "Fruit salad, canned in pear juice",
    #     "Fruit salad, canned in pear juice, drained",
    #     "Fruit salad, canned in pineapple juice",
    #     "Fruit salad, canned in pineapple juice, drained",
    #     "Fruit salad, canned in syrup",
    #     "Fruit salad, canned in syrup, drained",
    #     "Grains based salad with 3 vegetable toppings, no dressing",
    #     "Grains based salad with chicken and 3 vegetable toppings, no dressing",
    #     "Grains based salad with fish and 3 vegetable toppings, no dressing",
    #     "Gravy, assam pedas",
    #     "Gravy, for Indian rojak",
    #     "Gravy, laksa",
    #     "Gravy, mee rebus",
    #     "Gravy, mee siam",
    #     "Grilled stingray with sambal",
    #     "Ham Salad, Subway",
    #     "Hokkien mee",
    #     "Hot and spicy beef noodles soup",
    #     "Ice kachang",
    #     "Indian rojak, tempeh, battered, fried",
    #     "Japanese Pork Ramen",
    #     "Japanese shio ramen",
    #     "Japanese shoyu ramen",
    #     "Kaya Toast with Butter",
    #     "Kopi",
    #     "Kopi C",
    #     "Kopi C siu dai",
    #     "Kopi O",
    #     "Kopi O siu dai",
    #     "Kopi siu dai",
    #     "Korean bulgogi beef with rice",
    #     "Kuih apam balik",
    #     "Kuih bangkit sagu",
    #     "Kuih koci pulut hitam",
    #     "Kuih sagu",
    #     "Kway chap",
    #     "Kway chap, noodles only",
    #     "Kway teow soup, with beef balls",
    #     "Laksa",
    #     "Laksa lemak, without gravy",
    #     "Laksa noodles, cooked",
    #     "Laksa yong tauhu",
    #     "Laksa, leaf, fresh",
    #     "Liver roll ngoh hiang",
    #     "Lontong goreng",
    #     "Lor mee",
    #     "Lor mee (NEW)",
    #     "Ma La Xiang Guo",
    #     "Mee goreng",
    #     "Mee goreng, mamak style",
    #     "Mee rebus",
    #     "Mee rebus, without gravy",
    #     "Mee siam",
    #     "Mee siam, without gravy",
    #     "Mee soto",
    #     "Mushrooom Fritter (Fried Mushroom)",
    #     "Mutton briyani",
    #     "Mutton curry puff",
    #     "Mutton satay, without satay sauce",
    #     "Nasi Lemak with chicken wing",
    #     "Nasi briyani, rice only",
    #     "Nasi lemak with fried egg only",
    #     "Nasi lemak, rice only",
    #     "Ngoh hiang, meat roll",
    #     "Ngoh hiang, meat roll, bung bung",
    #     "Ngoh hiang, mixed items",
    #     "Ngoh hiang, prawn fritter, crispy",
    #     "Ngoh hiang, prawn fritter, dough",
    #     "Ngoh hiang, sausage",
    #     "Ngoh hiang, yam meat roll",
    #     "Noodles, instant, chicken curry, with seasoning, uncooked",
    #     "Noodles, laksa, thick, dried",
    #     "Noodles, laksa, thick, wet",
    #     "Noodles, with prawn, tofu and vegetables, soup",
    #     "Omelette, oyster",
    #     "Otak",
    #     "Otak, shrimp",
    #     "Otak, sotong",
    #     "Oven Roasted Chicken Breast Salad, Subway",
    #     "Pandan chiffon cake",
    #     "Paru goreng",
    #     "Paste, hainanese chicken rice",
    #     "Paste, laksa, commercial",
    #     "Paste, mee rebus",
    #     "Paste, mee siam, commercial",
    #     "Peanut kueh tutu",
    #     "Penang laksa",
    #     "Penang prawn noodle",
    #     "Pig's liver soup",
    #     "Plain roti prata",
    #     "Pop corn, durian flavoured",
    #     "Popiah",
    #     "Popiah circular shape, skin only",
    #     "Popiah skin",
    #     "Pork satay, with satay sauce",
    #     "Potato curry puff",
    #     "Prawn cocktail",
    #     "Prawn noodles soup",
    #     "Pulut hitam with coconut milk",
    #     "Pulut hitam, served with coconut milk",
    #     "Red rice porridge, plain",
    #     "Rendang hati ayam",
    #     "Rice porridge, fish, dry",
    #     "Roast Beef Salad, Subway",
    #     "Roasted duck rice",
    #     "Roti john",
    #     "Salad with 3 vegetable toppings, no dressing",
    #     "Salad with chicken and 3 vegetable toppings, no dressing",
    #     "Salad with chicken and 4 vegetable toppings, no dressing",
    #     "Salad with fish and 3 vegetable toppings, no dressing",
    #     "Salad, ocean chef, Long John Silver's",
    #     "Salad, seafood, Long John Silver's",
    #     "Salad, vegetable, tossed, without dressing",
    #     "Sandwiches and burgers, roast beef sandwich with cheese",
    #     "Sardine curry puff",
    #     "Satay bee hoon",
    #     "Satay sauce",
    #     "Satay, beef, frozen",
    #     "Satay, chicken, canned",
    #     "Satay, chicken, frozen",
    #     "Satay, mutton, frozen",
    #     "Sauce, BBQ, McDonalds'",
    #     "Sauce, chee cheong fun",
    #     "Sausage, cocktail, chicken, boiled",
    #     "Sayur lodeh",
    #     "Shrimp chee cheong fun",
    #     "Soto ayam",
    #     "Soup, pig's liver , chinese spinach",
    #     "Spicy cucumber salad with coconut milk",
    #     "Subway Club Salad, Subway",
    #     "Sup tulang",
    #     "Sushi roll",
    #     "Sushi, california roll",
    #     "Sushi, raw tuna, roll",
    #     "Sushi, roll, cucumber",
    #     "Sushi, roll, futomaki",
    #     "Sushi, tuna salad",
    #     "Sweet Onion Chicken Teriyaki Salad, Subway",
    #     "Sweet potato ondeh ondeh",
    #     "Thai chicken feet salad",
    #     "Thai mango salad",
    #     "Thunder Tea Rice with Soup",
    #     "Traditional ondeh ondeh",
    #     "Tuna, salad, with thousand island dressing, canned",
    #     "Turkey Breast Salad, Subway",
    #     "Turtle soup",
    #     "Vadai with kacang hitam",
    #     "Vadai, kacang dal kuning",
    #     "Vegetable briyani",
    #     "Vegetarian brown rice porridge",
    #     "Vegetarian fried bee hoon",
    #     "Veggie Delite Salad, Subway",
    #     "Yong tau foo, beancurd skin, deep fried",
    #     "Yong tau foo, bittergourd with fish paste, boiled",
    #     "Yong tau foo, chilli sauce",
    #     "Yong tau foo, eggplant with fish paste",
    #     "Yong tau foo, fishmeat wrapped with taukee",
    #     "Yong tau foo, mixed items, noodles not included",
    #     "Yong tau foo, okra with fish paste",
    #     "Yong tau foo, pork skin, deep fried",
    #     "Yong tau foo, red chilli with fish",
    #     "Yong tau foo, red sauce",
    #     "Yong tau foo, squid roll",
    #     "Yong tau foo, taupok with fish paste",
    #     "Yong tau foo, tofu with fish paste",
    #     "You tiao",
    #     "Chicken murtabak",
    #     "Coleslaw, KFC",
    #     "Coleslaw, Long John Silver's",
    #     "Cucur badak",
    #     "Cucur udang",
    #     "Dressing, coleslaw, reduced fat, commercial",
    #     "Dressing, coleslaw, regular, commercial",
    #     "Dry minced pork and mushroom noodles",
    #     "Green wanton noodles dry",
    #     "Hong Kong wanton noodles, dry",
    #     "Kuih lompang",
    #     "Mushroom and minced pork noodles soup",
    #     "Paper thosai",
    #     "Rawa thosai",
    #     "Roasted duck",
    #     "Roasted duck rice",
    #     "Roasted duck without skin",
    #     "Roasted mock duck",
    #     "Thosai",
    #     "Thosai masala",
    #     "Vegetable murtabak",
    #     "Wanton noodles dry",
    #     "Wanton noodles soup",
    #     "gulai duan ubi",
    #     "Har cheong gai",
    #     "Nasi padang",
    #     "Economy rice (mixed rice)",
    #     "Taco"
    # ]

    # Create a prompt for the model to classify the food image
    prompt = f"""
    You are a culinary expert specializing in Singaporean cuisine. Given an image of a food item, first carefully examine its visual appearance, ingredients, and likely cooking techniques to infer how it was prepared. Based on this analysis, classify the dish into one of the following categories:
    Food classes: {', '.join(class_names)}
    After identifying the most promising dish, recall the ingredients used in making that dish and list them in the response accordingly.
    """
    prompt+="""Respond as a JSON String in the following format and nothing else:
        {
        "dishName": "the food dish name",
        "Ingredients": [ /* list of ingredients recalled; empty if not food */ ],
        "Nutrients": {
            "Protein": x,        // in grams (integer)
            "Total Fat": y,            // in grams (integer)
            "Carbohydrates": z   // in grams (integer)
            "Calories:": a, // in kcal (integer)
            "Sugars": b, // in grams (integer)
            "Sodium": c, // in mg (integer)
            "Per Serving Household Measure" : z // in grams (integer)
            }
        }
        Do not include any additional text or explanation.
    """

    # Use the model to generate a response based on the prompt and image
    response = model.generate_content([prompt, image])
    json_str = response.text.strip().replace("```json","").replace("```","")
    food_data = json.loads(json_str)

    # Process the response to obtain the predicted food class name
    # predicted_food = response.text.strip().lower().replace(" ", "_")

    # Retrieve nutritional information for the predicted food class
    # nutrition_info = nutrition_map.get(predicted_food, None)

    # Handle case where nutritional information is not available
    # if nutrition_info is None:
    #     nutrition_result = {
    #         "message": "Nutritional information not available for this food"
    #     }
    # else:
        # Uncomment and complete this section if handling multiple portion sizes is needed

    # return predicted_food, nutrition_info
    return food_data

def process_image(image: Image, model):
    """Process the input image and return the predicted food and its nutritional information."""
    # Make a copy of the original image for display
    # display_image = image.copy()
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Create a copy for model processing
    model_image = image.copy()
    target_size = (224, 224)

    # Resize the model image copy
    model_image.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Create a new blank image for the model
    new_image = Image.new("RGB", target_size, (255, 255, 255))

    # Calculate the offset to center the thumbnail
    offset = ((target_size[0] - model_image.size[0]) // 2,
              (target_size[1] - model_image.size[1]) // 2)

    # Paste the thumbnail onto the new image
    new_image.paste(model_image, offset)

    # Get predictions using the model-sized image
    food_data = predict_food_and_nutrition(new_image, model)

    # Return both the predictions and the unmodified display image
    return food_data

def display_results(food_data, image=None, nutrition_map=None):
    """
    Display the predicted food name, list of ingredients, and its corresponding nutrition information
    in a two-column layout with improved formatting.
    """
    col1, col2 = st.columns(2)

    # Column 1: Show the food image
    with col1:
        if image is not None:
            # Dynamically set max width: 350px if enough space, else 80% of container
            st.image(
                image,
                caption=food_data.get('dishName', 'Food'),
                use_container_width=True  # This makes the image responsive
            )

    # Column 2: Show the food name, ingredients, and nutrition information
    with col2:
        st.subheader(f"{food_data.get('dishName', 'N/A')}")
        
        # Display ingredients
        st.markdown("---")
        st.markdown("**Ingredients:**")
        ingredients = food_data.get('Ingredients', [])
        if ingredients:
            st.markdown("• " + " \n• ".join(ingredients))
        else:
            st.markdown("_No ingredients detected._")

        # Display nutrition information
        st.markdown("---")
        st.markdown("**Nutrition Information:**")
        nutrients = food_data.get('Nutrients', {})
        
        # Try to get nutrition info from nutrition_map if available
        dish_key = food_data.get('dishName', '').lower().replace(" ", "_")
        
        if nutrition_map and dish_key in nutrition_map:
            nutrition_info = nutrition_map[dish_key]
            # Use correct key names from nutrition map
            nutrients_display = {
                "Total Fat": nutrition_info.get("Total fat (g)", "N/A"),
                "Protein": nutrition_info.get("Protein (g)", "N/A"),
                "Carbohydrates": nutrition_info.get("Carbohydrate (g)", "N/A"),
                "Sugars": nutrition_info.get("Sugar (g)", "N/A"),
                "Sodium": nutrition_info.get("Sodium (mg)", "N/A"),
                "Calories": nutrition_info.get("Energy (kcal)", "N/A"),
                "Per Serving Household Measure": nutrition_info.get("Per Serving Household Measure", "N/A")
            }
        else:
            # Fallback to model predictions if nutrition map data isn't available
            nutrients_display = {
                "Total Fat": nutrients.get("Total Fat", "N/A"),
                "Protein": nutrients.get("Protein", "N/A"),
                "Carbohydrates": nutrients.get("Carbohydrates", "N/A"),
                "Sugars": nutrients.get("Sugars", "N/A"),
                "Sodium": nutrients.get("Sodium", "N/A"),
                "Calories": nutrients.get("Calories", "N/A"),
                "Per Serving Household Measure": nutrients.get("Per Serving Household Measure", "N/A")
            }

        if any(value != "N/A" for value in nutrients_display.values()):
            col_left, col_right = st.columns(2)
            
            st.markdown(f"Per Serving Household Measure : {nutrients_display.get('Per Serving Household Measure', 'N/A')}")
            with col_left:
                st.markdown(f"Total Fat: {nutrients_display['Total Fat']} g")
                st.markdown(f"Protein: {nutrients_display['Protein']} g")
                st.markdown(f"Carbohydrates: {nutrients_display['Carbohydrates']} g")

            with col_right:
                st.markdown(f"Sugars: {nutrients_display['Sugars']} g")
                st.markdown(f"Sodium: {nutrients_display['Sodium']} mg")
                st.markdown(f"Calories: {nutrients_display['Calories']} kcal")
        else:
            st.info("Nutritional information not available.")
        
        st.markdown("---")

# Main application flow
def main():
    """Main function to run the Streamlit app."""

    # Load model and nutrition data
    model = load_model()
    nutrition_map = load_nutrition_data("Final_key_value_pair.json")
    
    # Check if model loaded correctly
    if model is None:
        st.error("Failed to load the model. Please check your API key and refresh.")
        return

    # Create tabs for camera or upload options
    tabs = st.tabs(["Take Photo", "Upload Image"])

    # Camera input tab
    with tabs[0]:
        st.subheader("Use Camera")

        st.write("Click the button below to access your device camera")

        img_file = st.camera_input("Take a picture of food")

        if img_file is not None:
            with st.spinner("Analyzing food..."):
                try:
                    captured_image = Image.open(img_file)
                    food_data = process_image(captured_image, model)
                    display_results(food_data,captured_image,nutrition_map)
                except Exception as e:
                    st.error(f"Error processing image (click): {e}")

    # File upload tab
    with tabs[1]:
        st.subheader("Upload Food Image")

        st.write("Upload an image of food from your computer or mobile device")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            with st.spinner("Analyzing food..."):
                try:
                    uploaded_image = Image.open(uploaded_file)
                    food_data = process_image(uploaded_image, model)
                    # Fix: Changed parameter order to match the function definition
                    display_results(food_data, uploaded_image,nutrition_map)  # Changed from display_results(uploaded_image, food_data)
                except Exception as e:
                    st.error(f"Error processing image (upload): {e}")

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
