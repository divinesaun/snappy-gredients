# app.py
import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv
import requests
import io

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OCR_SPACE_API_KEY = os.getenv('OCR_SPACE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

def ocr_space_file(image_data, api_key=OCR_SPACE_API_KEY, language='eng'):
    if not api_key:
        return None, "OCR Space API key is not configured. Please check your environment variables."
        
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': api_key,
        'language': language,
        'isOverlayRequired': False,
        'OCREngine': 2  # Using OCR Engine 2 for better accuracy
    }
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image_data.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    files = {
        'file': ('image.png', img_byte_arr, 'image/png')
    }
    
    try:
        response = requests.post(url, files=files, data=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        
        if result['IsErroredOnProcessing']:
            return None, f"Error: {result['ErrorMessage']}"
            
        if not result['ParsedResults']:
            return None, "No text was detected in the image."
            
        return result['ParsedResults'][0]['ParsedText'], None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            return None, "Authentication failed. Please check if your OCR Space API key is valid and properly configured."
        return None, f"HTTP Error: {str(e)}"
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Set page config
st.set_page_config(
    page_title="Snappy Gredients - Ingredient Analyzer",
    page_icon="🥣",
    layout="wide"
)

st.title("🥣 Snappy Gredients - Ingredient Analyzer")
st.markdown("""
This tool helps you understand the ingredients in your food products. 
Upload an image or take a photo of the ingredients list, and we'll analyze it for you!
""")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📸 Take Photo", "📤 Upload Image"])

def analyze_ingredients(text):
    if not text.strip():
        return "No text was detected. Please try again with a clearer image."
    
    prompt = f"""You are an expert food ingredient analyzer. Your task is to analyze the following list of ingredients and provide health-related insights. 
    Focus ONLY on the ingredients and their potential health implications. If the text is unclear or seems unrelated to food ingredients, 
    inform the user to try again with a better image.

    Ingredients list:
    {text}

    Please provide:
    1. A brief overview of the main ingredients
    2. Potential allergens
    3. Additives or preservatives to be cautious about
    4. Any health concerns or benefits
    5. Recommendations for people with specific dietary restrictions

    If the text doesn't appear to be a list of food ingredients, please inform the user to try again with a clearer image of the ingredients list.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing ingredients: {str(e)}"

def process_image(image):
    with st.spinner("Extracting text..."):
        extracted_text, error = ocr_space_file(image)
    
    if error:
        st.error(error)
        return
        
    st.subheader("📝 Extracted Text")
    st.text_area("Text Output", extracted_text, height=150)
    
    if extracted_text and extracted_text.strip():
        st.subheader("🔬 Analysis")
        analysis = analyze_ingredients(extracted_text)
        st.markdown(analysis)
    else:
        st.warning("No text was detected in the image. Please try again with a clearer image.")

# Camera tab
with tab1:
    st.subheader("Take a Photo")
    img_file_buffer = st.camera_input("Take a picture of the ingredients list")
    
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        process_image(image)

# Upload tab
with tab2:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        process_image(image)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔍 Snappy Gredients - Making ingredient lists easier to understand</p>
    <p>Note: This tool is for informational purposes only. Always consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)
