
import requests, base64

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False


with open(r"C:\Users\sai krishna\OneDrive\Desktop\download.jpeg", "rb") as f:
  image_b64 = base64.b64encode(f.read()).decode()

headers = {
  "Authorization": "Bearer API-key",
  "Accept": "text/event-stream" if stream else "application/json"
}

# List of food class names in the dataset
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

# Create a prompt for the model to classify the food image
prompt = f"""
Given the image of a food item, identify which of the following singaporean food classes it belongs to.
Only return the single best match from this list. Do not add any extra text or explanation.

Food classes: {', '.join(class_names)}
"""


payload = {
  "model": "mistralai/mistral-small-3.1-24b-instruct-2503",
  "messages": [
      {
        "role": "user",
        "content": prompt + f"\n <img src=\"data:image/png;base64,{image_b64}\" />"
      }
    ],
  "max_tokens": 4000,
  "temperature": 0.20,
  "top_p": 0.70,
  "frequency_penalty": 0.00,
  "presence_penalty": 0.00,
  "stream": stream
}

response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    res = response.json()
    if "choices" in res and len(res["choices"]) > 0:
        print("dish is :",res["choices"][0]["message"]["content"])
