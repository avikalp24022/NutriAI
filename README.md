
# NutriAI
================

## Table of Contents
-----------------

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Training](#model-training)
7. [Nutrition Data](#nutrition-data)
8. [API Documentation](#api-documentation)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction
------------

NutriAI is a food nutrition analysis application that uses machine learning to predict the nutritional information of a given food image. The application is built using Python and utilizes the Streamlit framework for the user interface.

## Features
--------

*   Image classification using a pre-trained machine learning model
*   Nutrition information prediction based on the classified food image
*   User-friendly interface for uploading food images
*   Display of nutrition information in a easy-to-read format

## Requirements
------------

*   Python 3.8+
*   Streamlit 1.45.0+
*   Torch 2.7.0+
*   Transformers 4.51.3+
*   Pillow 11.2.1+
*   Pandas 2.2.3+
*   Matplotlib 3.10.1+
*   Huggingface-hub 0.30.2+
*   Google-generativeai

## Installation
------------

To install the required dependencies, run the following command:

```bash
# 1. Install Python venv module if missing
sudo apt install python3-venv -y

# 2. Create a virtual environment in your project directory
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Now install your Python packages using pip
pip install streamlit==1.45.0 \
            google-generativeai \
            Pillow==11.2.1 \
            pandas==2.2.3 \
            python-dotenv
```

## Usage
-----

To run the application, execute the following command:

```bash
streamlit run food_nutrition_app.py
```

This will start the Streamlit server and make the application available at `http://localhost:8501`.

## Model Training
--------------

The machine learning model used in this application is pre-trained and can be found in the `food_nutrition_app.py` file. The model is trained on a dataset of food images and their corresponding nutrition information.

## Nutrition Data
-------------

The nutrition data used in this application is stored in a JSON file called `Final_key_value_pair.json`. This file contains a dictionary of food items and their corresponding nutrition information.

## API Documentation
-----------------

The API documentation for this application can be found in the `food_nutrition_app.py` file. The API is built using the Streamlit framework and provides endpoints for uploading food images and retrieving nutrition information.

## Contributing
------------

Contributions to this project are welcome. To contribute, please fork the repository and submit a pull request with your changes.
