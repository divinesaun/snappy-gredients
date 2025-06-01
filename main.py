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

def compress_image(image, max_size_mb=1):
    """Compress image to be under max_size_mb"""
    # Convert to RGB if image is in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Start with high quality
    quality = 95
    output = io.BytesIO()
    
    # Save with initial quality
    image.save(output, format='JPEG', quality=quality)
    size = output.tell() / (1024 * 1024)  # Size in MB
    
    # If image is already under max_size_mb, return as is
    if size <= max_size_mb:
        output.seek(0)
        return Image.open(output)
    
    # Binary search for the right quality
    min_quality = 5
    max_quality = 95
    
    while min_quality <= max_quality:
        quality = (min_quality + max_quality) // 2
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality)
        size = output.tell() / (1024 * 1024)
        
        if size <= max_size_mb:
            min_quality = quality + 1
        else:
            max_quality = quality - 1
    
    # Use the last quality that worked
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=max_quality)
    output.seek(0)
    return Image.open(output)

def ocr_space_file(image_data, api_key=OCR_SPACE_API_KEY, language='eng'):
    if not api_key:
        return None, "OCR Space API key is not configured. Please check your environment variables."
        
    url = 'https://api.ocr.space/parse/image'
    
    def compress_image_to_size(image, max_size_mb=1):
        """Compress image to be under max_size_mb"""
        # Convert to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Start with high quality
        quality = 95
        min_quality = 5
        max_quality = 95
        
        while min_quality <= max_quality:
            # Calculate current quality
            quality = (min_quality + max_quality) // 2
            
            # Save with current quality
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
            size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
            
            if size_mb <= max_size_mb:
                # If we're under the size limit, try to increase quality
                min_quality = quality + 1
            else:
                # If we're over the size limit, decrease quality
                max_quality = quality - 1
                
            # If we've found a working quality, return the compressed image
            if size_mb <= max_size_mb:
                img_byte_arr.seek(0)
                return img_byte_arr.getvalue()
        
        # If we get here, try one last time with minimum quality
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=5, optimize=True)
        return img_byte_arr.getvalue()
    
    # Compress the image
    compressed_image = compress_image_to_size(image_data)
    
    # Verify the size
    size_mb = len(compressed_image) / (1024 * 1024)
    if size_mb > 1:
        return None, f"Unable to compress image below 1MB. Current size: {size_mb:.2f}MB"
    
    # Prepare the request
    payload = {
        'apikey': api_key,
        'language': language,
        'isOverlayRequired': False,
        'detectOrientation': True,
        'OCREngine': 2,
        'scale': True,
        'isTable': False,
        'filetype': 'jpg'
    }
    
    files = {
        'file': ('image.jpg', compressed_image, 'image/jpeg')
    }
    
    try:
        # Add headers to specify content type
        headers = {
            'apikey': api_key
        }
        
        response = requests.post(
            url,
            files=files,
            data=payload,
            headers=headers,
            timeout=30
        )
        
        # Print response for debugging
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text[:200]}")
        
        response.raise_for_status()
        result = response.json()
        
        if result.get('IsErroredOnProcessing'):
            error_message = result.get('ErrorMessage', 'Unknown error occurred')
            return None, f"Error: {error_message}"
            
        if not result.get('ParsedResults'):
            return None, "No text was detected in the image."
            
        return result['ParsedResults'][0]['ParsedText'], None
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error: {str(e)}"
        if e.response is not None:
            try:
                error_data = e.response.json()
                error_msg = f"API Error: {error_data.get('ErrorMessage', str(e))}"
            except:
                error_msg = f"HTTP Error {e.response.status_code}: {str(e)}"
        return None, error_msg
    except requests.exceptions.RequestException as e:
        return None, f"Request Error: {str(e)}"
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
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Take Photo"])

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
    # Compress image if needed
    compressed_image = compress_image(image)
    
    with st.spinner("Extracting text..."):
        extracted_text, error = ocr_space_file(compressed_image)
    
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
with tab2:
    st.subheader("Take a Photo")
    img_file_buffer = st.camera_input("Take a picture of the ingredients list")
    
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        process_image(image)

# Upload tab
with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        process_image(image)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔍 Snappy Gredients - Making ingredient lists easier to understand</p>
    <p>Note: This tool is for informational purposes only. Always consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)
