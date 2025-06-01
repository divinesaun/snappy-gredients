# app.py
import streamlit as st
from PIL import Image
import pytesseract
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Optional: specify tesseract path manually (mainly for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\divin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Set page config
st.set_page_config(
    page_title="Snappy Gredients - Ingredient Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Snappy Gredients - Ingredient Analyzer")
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
        extracted_text = pytesseract.image_to_string(image)
    
    st.subheader("📝 Extracted Text")
    st.text_area("Text Output", extracted_text, height=150)
    
    if extracted_text.strip():
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
