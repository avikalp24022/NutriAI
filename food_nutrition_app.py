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

# # Create a sidebar for options
# st.sidebar.header("Settings")
# portion_size = st.sidebar.selectbox("Portion Size", ["small", "medium", "large"], index=1)


# Then modify your load_model function
@st.cache_resource
def load_model():
    """Load and configure the Gemini Pro Vision model"""
    try:
        genai.configure(api_key="AIzaSyB7J2toPbIOC0dZho2mJNIy1cReogQnmDw")
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
        for label, group in nutrition_df.groupby('Dish'):
            # Convert numeric columns to appropriate types
            for col in ['Per Serving Household Measure','B-Carotene','Calcium','Carbohydrate','Cholesterol','Dietary fibre','Energy','Iron','Monounsaturated fat','Phosphorus',"Polyunsaturated fat","Potassium","Protein","Retinol","Riboflavin","Saturated fat","Selenium","Sodium","Starch","Sugar","Thiamin","'otal fat','Vitamin A','Vitamin C','Vitamin D','Water','Whole-grains','Zinc']:
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
    # else:
        # If multiple portion sizes are available, select based on preference
        # if isinstance(nutrition_info, list) and len(nutrition_info) > 0:
        #     if portion_size == 'small':
        #         nutrition_result = nutrition_info[0]  # smallest portion
        #     elif portion_size == 'large':
        #         nutrition_result = nutrition_info[-1]  # largest portion
        #     else:  # medium (default)
        #         mid_idx = len(nutrition_info) // 2
        #         nutrition_result = nutrition_info[mid_idx]

            # Add information about available portion sizes
        # available_portions = [item['weight'] for item in nutrition_info]
        # nutrition_result['available_portions'] = available_portions
        # else:
            # nutrition_result = nutrition_info

    return nutrition_info

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
        nutrition_map = load_nutrition_data("Final_all_nutrition_per_serving.csv")

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
