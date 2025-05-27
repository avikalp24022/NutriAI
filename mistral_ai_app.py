import streamlit as st
from PIL import Image
import pandas as pd
import io
import json
import base64
import requests

# Configuration for the Streamlit page
PAGE_TITLE = "Food Nutrition Analyzer"
PAGE_ICON = "🍔"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "collapsed"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

st.title(PAGE_TITLE)
st.write("Take a photo or upload a food image to get nutritional information")

@st.cache_data
def load_nutrition_data(file_path: str) -> dict:
    """
    Load nutrition data from a JSON file, and convert numeric fields to floats
    where possible.

    Args:
        file_path (str): The path to the JSON file containing the nutrition data.

    Returns:
        dict: A dictionary where the keys are the dish names (lowercase and
        underscore-separated) and the values are dictionaries with the same fields
        as the JSON file, but with numeric fields converted to float where possible.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Load the JSON file as a dictionary
            nutrition_data = json.load(file)

        # Define the list of numeric fields
        numeric_fields = [
            'B-Carotene', 'Calcium', 'Carbohydrate', 'Cholesterol', 'Dietary fibre',
            'Energy', 'Iron', 'Monounsaturated fat', 'Phosphorus', 'Polyunsaturated fat',
            'Potassium', 'Protein', 'Retinol', 'Riboflavin', 'Saturated fat', 'Selenium',
            'Sodium', 'Starch', 'Sugar', 'Thiamin', 'Total fat', 'Vitamin A',
            'Vitamin C', 'Vitamin D', 'Water', 'Whole-grains', 'Zinc'
        ]

        # Create an empty dictionary to store the converted data
        nutrition_map = {}

        # Iterate over each dish in the nutrition data
        for dish, entry in nutrition_data.items():
            # Create a copy of the entry to avoid modifying the original data
            dish_data = entry.copy()

            # Iterate over each numeric field and convert it to a float if possible
            for field in numeric_fields:
                val = dish_data.get(field)
                try:
                    # Try to convert the value to a float
                    dish_data[field] = float(val) if val not in [None, "", " "] else None
                except Exception:
                    # If the conversion fails, set the value to None
                    dish_data[field] = None

            # Add the converted data to the nutrition map
            nutrition_map[dish.lower().replace(" ", "_")] = dish_data

        # Return the nutrition map
        return nutrition_map

    except Exception as e:
        # If an error occurs, display an error message and return an empty dictionary
        st.error(f"Error loading nutrition data: {e}")
        return {}

def predict_food_and_nutrition(image_b64: str, nutrition_map: dict):
    """
    Predict the food class and retrieve its nutritional information.

    This function sends a request to the Mistral AI API to predict the food class
    from an image, and then retrieves the corresponding nutritional information
    from the nutrition map.

    Args:
        image_b64 (str): A base64 encoded string representing the image to be
            classified.
        nutrition_map (dict): A dictionary mapping food class names to their
            corresponding nutritional information.

    Returns:
        tuple: A tuple containing the predicted food class and its nutritional
            information. If the prediction fails or the nutrition information is
            not found, the function returns (None, None).
    """

    # Set the API URL and headers
    api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer API-key",
        "Content-Type": "application/json"
    }

    # Define the list of class names
    class_names = [
        "Ayam penyet with rice",
        "BBQ Turkey Bacon Double Cheeseburger, Burger King",
        "Bak kut teh",
        "Bak kut teh, soup only",
        "Baked durian mooncake",
        "Ban mian soup",
        "Banana fritter",
        "Beans broad, coated with satay powder, deep fried",
        "Beef ball kway teow soup",
        "Beef burger with cheese",
        "Beef noodles soup",
        "Beef rendang",
        "Beef satay, without satay sauce",
        "Beef, rendang with soya sauce, simmered (Malay)",
        "Biscuit, savoury, salada style",
        "Braised duck rice",
        "Braised duck with yam rice",
        "Braised pork ribs with black bean sauce",
        "Brown rice laksa noodles, cooked",
        "Brown rice porridge, plain",
        "Bulgogi Gimbap (Korean Rice Roll with Spicy Beef)",
        "Caesar salad",
        "Cafe 26 oriental salad dressing",
        "Cafe 26 tangy salad dressing",
        "Carrot cake with egg, plain, mashed & fried",
        "Cereal prawn",
        "Char kway teow",
        "Char siew chee cheong fun",
        "Chee cheong fun",
        "Chee pah",
        "Cheesy BBQ Meltz, KFC",
        "Chendol",
        "Chendol, durian",
        "Chendol, mango",
        "Chicken briyani",
        "Chicken curry noodles",
        "Chicken satay, without peanut sauce",
        "Chilli crab",
        "Chwee kueh",
        "Claypot rice with salted fish,chicken and chinese sausages",
        "Claypot rice, with mixed vegetable",
        "Claypot rice, with prawn",
        "Claypot rice, with stewed beef",
        "Coconut kueh tutu",
        "Curry fish head",
        "Curry puff, beef",
        "Curry puff, chicken",
        "Curry puff, frozen, deep fried",
        "Curry puff, potato and mutton filling, deep fried",
        "Curry puff, potato, and spices, deep fried",
        "Curry puff, twisted",
        "Deep fried carrot cake",
        "Deep fried fish bee hoon soup with milk",
        "Dim sum, beancurd roll",
        "Dim sum, chicken feet with dark sauce, stewed",
        "Dim sum, dumpling, chives with minced prawn, steamed",
        "Dim sum, dumpling, yam, deep fried",
        "Dim sum, pork ribs",
        "Dim sum, pork tart, BBQ",
        "Dim sum, sharkfin dumpling",
        "Dim sum, siew mai, steamed",
        "Dim sum, turnip cake, steamed",
        "Dim sum, you tiao",
        "Dodol berdurian",
        "Double Cheeseburger, McDonalds'",
        "Drunken prawn",
        "Dry prawn noodles",
        "Duck rice, with skin removed",
        "Durian",
        "Durian Pancake",
        "Durian cake",
        "Durian fermented",
        "Durian ice kacang",
        "Durian pudding",
        "Durian puff",
        "Durian wafer",
        "Durian, Malaysian, mid-range",
        "Durian, raw",
        "Fast foods, salad, vegetable, tossed, without dressing, with chicken",
        "Fish ball mee pok, dry",
        "Fish ball noodles dry",
        "Fish finger, grilled or baked",
        "Fish head ban mian soup",
        "Fish ngoh hiang",
        "Fish satay snack",
        "Fried mee siam",
        "Fried plain carrot cake",
        "Fried vegetarian bee hoon, plain",
        "Frog leg claypot rice",
        "Fruit salad, canned in heavy syrup",
        "Fruit salad, canned in heavy syrup, drained",
        "Fruit salad, canned in pear juice",
        "Fruit salad, canned in pear juice, drained",
        "Fruit salad, canned in pineapple juice",
        "Fruit salad, canned in pineapple juice, drained",
        "Fruit salad, canned in syrup",
        "Fruit salad, canned in syrup, drained",
        "Grains based salad with 3 vegetable toppings, no dressing",
        "Grains based salad with chicken and 3 vegetable toppings, no dressing",
        "Grains based salad with fish and 3 vegetable toppings, no dressing",
        "Gravy, assam pedas",
        "Gravy, for Indian rojak",
        "Gravy, laksa",
        "Gravy, mee rebus",
        "Gravy, mee siam",
        "Grilled stingray with sambal",
        "Ham Salad, Subway",
        "Hokkien mee",
        "Hot and spicy beef noodles soup",
        "Ice kachang",
        "Indian rojak, tempeh, battered, fried",
        "Japanese Pork Ramen",
        "Japanese shio ramen",
        "Japanese shoyu ramen",
        "Kaya Toast with Butter",
        "Kopi",
        "Kopi C",
        "Kopi C siu dai",
        "Kopi O",
        "Kopi O siu dai",
        "Kopi siu dai",
        "Korean bulgogi beef with rice",
        "Kuih apam balik",
        "Kuih bangkit sagu",
        "Kuih koci pulut hitam",
        "Kuih sagu",
        "Kway chap",
        "Kway chap, noodles only",
        "Kway teow soup, with beef balls",
        "Laksa",
        "Laksa lemak, without gravy",
        "Laksa noodles, cooked",
        "Laksa yong tauhu",
        "Laksa, leaf, fresh",
        "Liver roll ngoh hiang",
        "Lontong goreng",
        "Lor mee",
        "Lor mee (NEW)",
        "Ma La Xiang Guo",
        "Mee goreng",
        "Mee goreng, mamak style",
        "Mee rebus",
        "Mee rebus, without gravy",
        "Mee siam",
        "Mee siam, without gravy",
        "Mee soto",
        "Mushrooom Fritter (Fried Mushroom)",
        "Mutton briyani",
        "Mutton curry puff",
        "Mutton satay, without satay sauce",
        "Nasi Lemak with chicken wing",
        "Nasi briyani, rice only",
        "Nasi lemak with fried egg only",
        "Nasi lemak, rice only",
        "Ngoh hiang, meat roll",
        "Ngoh hiang, meat roll, bung bung",
        "Ngoh hiang, mixed items",
        "Ngoh hiang, prawn fritter, crispy",
        "Ngoh hiang, prawn fritter, dough",
        "Ngoh hiang, sausage",
        "Ngoh hiang, yam meat roll",
        "Noodles, instant, chicken curry, with seasoning, uncooked",
        "Noodles, laksa, thick, dried",
        "Noodles, laksa, thick, wet",
        "Noodles, with prawn, tofu and vegetables, soup",
        "Omelette, oyster",
        "Otak",
        "Otak, shrimp",
        "Otak, sotong",
        "Oven Roasted Chicken Breast Salad, Subway",
        "Pandan chiffon cake",
        "Paru goreng",
        "Paste, hainanese chicken rice",
        "Paste, laksa, commercial",
        "Paste, mee rebus",
        "Paste, mee siam, commercial",
        "Peanut kueh tutu",
        "Penang laksa",
        "Penang prawn noodle",
        "Pig's liver soup",
        "Plain roti prata",
        "Pop corn, durian flavoured",
        "Popiah",
        "Popiah circular shape, skin only",
        "Popiah skin",
        "Pork satay, with satay sauce",
        "Potato curry puff",
        "Prawn cocktail",
        "Prawn noodles soup",
        "Pulut hitam with coconut milk",
        "Pulut hitam, served with coconut milk",
        "Red rice porridge, plain",
        "Rendang hati ayam",
        "Rice porridge, fish, dry",
        "Roast Beef Salad, Subway",
        "Roasted duck rice",
        "Roti john",
        "Salad with 3 vegetable toppings, no dressing",
        "Salad with chicken and 3 vegetable toppings, no dressing",
        "Salad with chicken and 4 vegetable toppings, no dressing",
        "Salad with fish and 3 vegetable toppings, no dressing",
        "Salad, ocean chef, Long John Silver's",
        "Salad, seafood, Long John Silver's",
        "Salad, vegetable, tossed, without dressing",
        "Sandwiches and burgers, roast beef sandwich with cheese",
        "Sardine curry puff",
        "Satay bee hoon",
        "Satay sauce",
        "Satay, beef, frozen",
        "Satay, chicken, canned",
        "Satay, chicken, frozen",
        "Satay, mutton, frozen",
        "Sauce, BBQ, McDonalds'",
        "Sauce, chee cheong fun",
        "Sausage, cocktail, chicken, boiled",
        "Sayur lodeh",
        "Shrimp chee cheong fun",
        "Soto ayam",
        "Soup, pig's liver , chinese spinach",
        "Spicy cucumber salad with coconut milk",
        "Subway Club Salad, Subway",
        "Sup tulang",
        "Sushi roll",
        "Sushi, california roll",
        "Sushi, raw tuna, roll",
        "Sushi, roll, cucumber",
        "Sushi, roll, futomaki",
        "Sushi, tuna salad",
        "Sweet Onion Chicken Teriyaki Salad, Subway",
        "Sweet potato ondeh ondeh",
        "Thai chicken feet salad",
        "Thai mango salad",
        "Thunder Tea Rice with Soup",
        "Traditional ondeh ondeh",
        "Tuna, salad, with thousand island dressing, canned",
        "Turkey Breast Salad, Subway",
        "Turtle soup",
        "Vadai with kacang hitam",
        "Vadai, kacang dal kuning",
        "Vegetable briyani",
        "Vegetarian brown rice porridge",
        "Vegetarian fried bee hoon",
        "Veggie Delite Salad, Subway",
        "Yong tau foo, beancurd skin, deep fried",
        "Yong tau foo, bittergourd with fish paste, boiled",
        "Yong tau foo, chilli sauce",
        "Yong tau foo, eggplant with fish paste",
        "Yong tau foo, fishmeat wrapped with taukee",
        "Yong tau foo, mixed items, noodles not included",
        "Yong tau foo, okra with fish paste",
        "Yong tau foo, pork skin, deep fried",
        "Yong tau foo, red chilli with fish",
        "Yong tau foo, red sauce",
        "Yong tau foo, squid roll",
        "Yong tau foo, taupok with fish paste",
        "Yong tau foo, tofu with fish paste",
        "You tiao"
    ]


    # Construct the prompt for the API
    prompt = f"""
    Given the image of a food item, identify which of the following Singaporean food classes it belongs to.
    Only return the single best match from this list. Do not add any extra text or explanation.
    Food classes: {', '.join(class_names)}
    """

    # Construct the payload for the API
    payload = {
        "model": "mistralai/mistral-small-3.1-24b-instruct-2503",
        "messages": [{"role": "user", "content": prompt + f"\n<img src=\"data:image/png;base64,{image_b64}\" />"}],
        "max_tokens": 4000,
        "temperature": 0.20,
        "top_p": 0.70,
        "frequency_penalty": 0.00,
        "presence_penalty": 0.00,
        "stream": False
    }

    # Send the request and get the response
    response = requests.post(api_url, headers=headers, json=payload).json()

    # Check for errors
    if response.get("error"):
        st.error(f"Model error: {response['error']['message']}")
        return None, None

    # Extract the predicted food class and its nutritional information
    choice = response.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "").strip()
    predicted_food = content.lower().replace(" ", "_")
    nutrition_info = nutrition_map.get(predicted_food)

    # If the nutrition information is not found, display a warning
    if not nutrition_info:
        st.warning(f"Nutrition information for '{predicted_food}' not found.")
        nutrition_info = {"message": "Nutrition information not available."}

    # Return the predicted food class and its nutritional information
    return predicted_food, nutrition_info

def process_image(image: Image.Image, nutrition_map: dict):
    """
    Process the image by resizing it and encoding it as a base64 string.

    This function takes an image and a nutrition map as input, and returns
    a tuple containing the predicted food class and its nutritional information
    as a dictionary.

    The image is resized to 224x224 pixels using the Lanczos filter, and then
    pasted onto a new 224x224 canvas with a white background. The image is then
    saved as a PNG image to a BytesIO buffer, which is then encoded as a
    base64 string.

    The base64 string is then passed to the predict_food_and_nutrition function,
    which sends a request to the Mistral AI API to predict the food class from
    the image. The predicted food class and its nutritional information are then
    retrieved from the nutrition map and returned as a tuple.
    """

    # Convert the image to RGB mode
    image = image.convert("RGB")

    # Resize the image to 224x224 pixels using the Lanczos filter
    image.thumbnail((224, 224), Image.Resampling.LANCZOS)

    # Create a new 224x224 canvas with a white background
    canvas = Image.new("RGB", (224, 224), "white")

    # Calculate the offset to center the image on the canvas
    offset = ((224 - image.width) // 2, (224 - image.height) // 2)

    # Paste the image onto the canvas
    canvas.paste(image, offset)

    # Save the image as a PNG to a BytesIO buffer
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")

    # Encode the image as a base64 string
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Predict the food class and retrieve its nutritional information
    return predict_food_and_nutrition(image_b64, nutrition_map)

def display_results(predicted_food: str, nutrition_info: dict, image: Image):
    """
    Display the predicted food and its corresponding nutrition information
    in a two-column layout. The first column contains the food image and the
    second column contains a card with the predicted food name and nutrition
    information.

    Args:
        predicted_food (str): The predicted food name.
        nutrition_info (dict): The nutrition information for the predicted food.
        image (Image): The input food image.
    """
    # Create two columns for layout in the Streamlit app
    col1, col2 = st.columns(2)

    # Column 1: Show the food image with a caption
    with col1:
        st.image(image, caption=predicted_food, use_container_width=True)

    # Column 2: Show the nutrition information card
    with col2:
        # Display the predicted food name as a subheader
        st.subheader(f"Food: {predicted_food}")
        
        # Display "Nutrition Information" as another subheader
        st.subheader("Nutrition Information")

        # Check if nutrition information is unavailable or contains a message
        if nutrition_info is None or "message" in nutrition_info:
            # Get the message from nutrition_info or use a default message
            message = nutrition_info.get("message", "Nutritional information not available.")
            # Display the message in an info box
            st.info(message)
        else:
            # Create a styled HTML card for displaying nutrition facts
            nutrition_card = (
                "<div style='background-color:#f0f2f6;padding:20px;border-radius:10px;color:black;'>"
                "<h3>Nutrition Facts</h3><hr>"
                "<p><strong>Calories:</strong> {} kcal</p>"
                "<p><strong>Protein:</strong> {} g</p>"
                "<p><strong>Carbohydrates:</strong> {} g</p>"
                "<p><strong>Total fat:</strong> {} g</p>"
                "<p><strong>Dietary fibre:</strong> {} g</p>"
                "<p><strong>Sugars:</strong> {} g</p>"
                "<p><strong>Sodium:</strong> {} mg</p>"
                "</div>"
            ).format(
                # Format the nutrition facts using values from the nutrition_info dictionary
                nutrition_info.get('Energy (kcal)', 'N/A'),
                nutrition_info.get('Protein (g)', 'N/A'),
                nutrition_info.get('Carbohydrate (g)', 'N/A'),
                nutrition_info.get('Total fat (g)', 'N/A'),
                nutrition_info.get('Dietary fibre (g)', 'N/A'),
                nutrition_info.get('Sugar (g)', 'N/A'),
                nutrition_info.get('Sodium (mg)', 'N/A')
            )

            # Display the nutrition card using Markdown with HTML allowed
            st.markdown(nutrition_card, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app."""

    # Load nutrition data from a JSON file
    nutrition_map = load_nutrition_data("Final_key_value_pair.json")

    # Create two tabs in the Streamlit interface: one for taking a photo and another for uploading an image
    tabs = st.tabs(["Take Photo", "Upload Image"])

    # Camera input tab
    with tabs[0]:
        st.subheader("Use Camera")  # Subheader for camera use
        st.write("Click the button below to access your device camera")  # Instructions for the user

        # Input from the camera, allowing the user to take a picture
        image_file = st.camera_input("Take a picture of food")

        # If an image has been captured
        if image_file is not None:
            with st.spinner("Analyzing food..."):  # Show a spinner while processing the image
                try:
                    # Open the image file using PIL
                    image = Image.open(image_file)
                    # Process the image and predict the food type and nutrition information
                    predicted_food, nutrition_info = process_image(image, nutrition_map)
                    # Display the prediction results
                    display_results(predicted_food, nutrition_info, image)
                except Exception as e:
                    # If an error occurs, display an error message
                    st.error(f"Error processing image: {e}")

    # File upload tab
    with tabs[1]:
        st.subheader("Upload Food Image")  # Subheader for image upload
        st.write("Upload an image of food from your computer or mobile device")  # Instructions for the user

        # File uploader widget for selecting an image file
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        # If a file has been uploaded
        if uploaded_file is not None:
            with st.spinner("Analyzing food..."):  # Show a spinner while processing the image
                try:
                    # Open the uploaded image file using PIL
                    image = Image.open(uploaded_file)
                    # Process the image and predict the food type and nutrition information
                    predicted_food, nutrition_info = process_image(image, nutrition_map)
                    # Display the prediction results
                    display_results(predicted_food, nutrition_info, image)
                except Exception as e:
                    # If an error occurs, display an error message
                    st.error(f"Error processing image: {e}")

    # Expander for troubleshooting tips
    with st.expander("Having trouble with the camera or upload?"):
        # Display troubleshooting information for users facing issues
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



