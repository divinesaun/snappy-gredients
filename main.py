import streamlit as st
import sqlite3
import hashlib
import os
from datetime import datetime, timedelta
from PIL import Image
import io
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import time

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# API Functions (replaced with LLM-based structured output)
def find_food_llm(query):
    """Search for food using LLM with structured output"""
    try:
        prompt = f"""
        You are a nutrition expert. Given the food query: "{query}", provide nutrition information in the following JSON format:
        {{
            "foods": [
                {{
                    "food_name": "exact food name",
                    "food_description": "Per 100g - Calories: X kcal | Fat: X g | Carbs: X g | Protein: X g",
                    "calories": X,
                    "protein_g": X.X,
                    "carbohydrates_total_g": X.X,
                    "fat_total_g": X.X,
                    "fiber_g": X.X,
                    "sugar_g": X.X
                }}
            ]
        }}
        
        Provide 3-5 common variations or similar foods. Use realistic nutrition values based on standard food databases.
        Only return valid JSON, no other text.
        """
        
        response = model.generate_content(prompt)
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"foods": []}
    except Exception as e:
        st.error(f"Error searching for food: {e}")
        return {"foods": []}

def calories_burned_llm(activity):
    """Calculate calories burned using LLM with structured output"""
    try:
        prompt = f"""
        You are a fitness expert. Given the exercise activity: "{activity}", provide calorie burn information in the following JSON format:
        {{
            "exercises": [
                {{
                    "name": "Exercise name with intensity",
                    "calories_per_hour": X,
                    "duration_minutes": X,
                    "total_calories": X,
                    "intensity": "low/moderate/high"
                }}
            ]
        }}
        Use realistic calorie burn rates based on standard fitness databases.
        Only return valid JSON, no other text.
        """
        
        response = model.generate_content(prompt)
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"exercises": []}
    except Exception as e:
        st.error(f"Error calculating calories: {e}")
        return {"exercises": []}

def extract_nutrition_from_image_llm(image, image_type):
    """Extract nutrition facts from image using LLM with structured output"""
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        if image_type == "nutrition facts table":
            prompt = """
            Analyze this nutrition facts table image and extract the nutrition information in the following JSON format:
            {
                "nutrition_facts": {
                    "food_name": "product name",
                    "serving_size": "serving size",
                    "calories": X,
                    "protein_g": X.X,
                    "carbohydrates_total_g": X.X,
                    "fat_total_g": X.X,
                    "fiber_g": X.X,
                    "sugar_g": X.X,
                    "sodium_mg": X.X,
                    "additional_info": "any other relevant nutrition info"
                }
            }
            
            Extract all visible nutrition information. If a value is not visible, use 0.
            Only return valid JSON, no other text.
            """
        else:  # ingredients list
            prompt = """
            Analyze this ingredients list image and extract nutrition information in the following JSON format:
            {
                "nutrition_facts": {
                    "food_name": "product name",
                    "ingredients": ["ingredient1", "ingredient2", ...],
                    "estimated_calories": X,
                    "estimated_protein_g": X.X,
                    "estimated_carbohydrates_total_g": X.X,
                    "estimated_fat_total_g": X.X,
                    "allergens": ["allergen1", "allergen2", ...],
                    "health_notes": "health implications and recommendations"
                }
            }
            
            Estimate nutrition based on ingredients. If exact values aren't visible, provide reasonable estimates.
            Only return valid JSON, no other text.
            """
        
        # Generate response using Gemini
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_byte_arr}])
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"nutrition_facts": {}}
    except Exception as e:
        return {"nutrition_facts": {}, "error": str(e)}

# Keep the old functions for backward compatibility but mark as deprecated
def find_food(query):
    """Deprecated: Use find_food_llm instead"""
    return find_food_llm(query)

def calories_burned(activity):
    """Deprecated: Use calories_burned_llm instead"""
    return calories_burned_llm(activity)

def scan_barcode(image_data):
    """Scan barcode using Nutritionix API"""
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image_data.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        url = "https://trackapi.nutritionix.com/v2/search/item"
        headers = {
            "x-app-id": os.getenv("NUTRITIONIX_APP_ID"),
            "x-app-key": os.getenv("NUTRITIONIX_API_KEY"),
            "Content-Type": "application/json"
        }
        data = {
            "upc": img_str,
            "image_type": "base64"
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error scanning barcode: {e}")
        return None

# Database setup
def init_database():
    """Initialize SQLite database with tables"""
    conn = sqlite3.connect('health_tracker.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            created_at TEXT
        )
    ''')
    
    # User health details table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            age INTEGER,
            weight REAL,
            height REAL,
            gender TEXT,
            activity_level TEXT,
            goal TEXT,
            updated_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Nutrition logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nutrition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            food_name TEXT,
            calories REAL,
            protein REAL,
            carbs REAL,
            fat REAL,
            fiber REAL,
            sugar REAL,
            serving_size TEXT,
            meal_type TEXT,
            logged_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Exercise logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exercise_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            activity_name TEXT,
            calories_burned REAL,
            duration_minutes INTEGER,
            intensity TEXT,
            logged_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Image analysis logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_analysis_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_type TEXT,
            analysis_text TEXT,
            nutrition_data TEXT,
            logged_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_current_timestamp():
    """Get current timestamp in consistent format"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email=None):
    """Register a new user (email is now optional and ignored)"""
    conn = sqlite3.connect('health_tracker.db')
    cursor = conn.cursor()
    try:
        password_hash = hash_password(password)
        current_time = get_current_timestamp()
        cursor.execute('INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
                      (username, password_hash, current_time))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticate user login"""
    conn = sqlite3.connect('health_tracker.db')
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    cursor.execute('SELECT id, username FROM users WHERE username = ? AND password_hash = ?',
                  (username, password_hash))
    user = cursor.fetchone()
    conn.close()
    
    return user

def save_user_health(user_id, age, weight, height, gender, activity_level, goal):
    """Save or update user health details"""
    conn = sqlite3.connect('health_tracker.db')
    cursor = conn.cursor()
    
    current_time = get_current_timestamp()
    
    # Check if health details already exist
    cursor.execute('SELECT id FROM user_health WHERE user_id = ?', (user_id,))
    existing = cursor.fetchone()
    
    if existing:
        cursor.execute('''
            UPDATE user_health 
            SET age = ?, weight = ?, height = ?, gender = ?, activity_level = ?, goal = ?, updated_at = ?
            WHERE user_id = ?
        ''', (age, weight, height, gender, activity_level, goal, current_time, user_id))
    else:
        cursor.execute('''
            INSERT INTO user_health (user_id, age, weight, height, gender, activity_level, goal, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, age, weight, height, gender, activity_level, goal, current_time))
    
    conn.commit()
    conn.close()

def get_user_health(user_id):
    """Get user health details"""
    conn = sqlite3.connect('health_tracker.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT age, weight, height, gender, activity_level, goal FROM user_health WHERE user_id = ?', (user_id,))
    health = cursor.fetchone()
    conn.close()
    
    return health

def log_nutrition(user_id, food_data, meal_type):
    """Log nutrition data"""
    conn = sqlite3.connect('health_tracker.db')
    cursor = conn.cursor()
    
    current_time = get_current_timestamp()
    
    cursor.execute('''
        INSERT INTO nutrition_logs (user_id, food_name, calories, protein, carbs, fat, fiber, sugar, serving_size, meal_type, logged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, food_data.get('food_name', ''), food_data.get('nf_calories', 0),
          food_data.get('nf_protein', 0), food_data.get('nf_total_carbohydrate', 0),
          food_data.get('nf_total_fat', 0), food_data.get('nf_dietary_fiber', 0),
          food_data.get('nf_sugars', 0), food_data.get('serving_size', ''), meal_type, current_time))
    
    conn.commit()
    conn.close()

def log_exercise(user_id, exercise_data):
    """Log exercise data"""
    conn = sqlite3.connect('health_tracker.db')
    cursor = conn.cursor()
    
    current_time = get_current_timestamp()
    
    cursor.execute('''
        INSERT INTO exercise_logs (user_id, activity_name, calories_burned, duration_minutes, intensity, logged_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, exercise_data.get('name', ''), exercise_data.get('nf_calories', 0),
          exercise_data.get('duration_min', 0), exercise_data.get('intensity', 'moderate'), current_time))
    
    conn.commit()
    conn.close()

def log_image_analysis(user_id, image_type, analysis_text, nutrition_data):
    """Log image analysis results"""
    conn = sqlite3.connect('health_tracker.db')
    cursor = conn.cursor()
    
    current_time = get_current_timestamp()
    
    cursor.execute('''
        INSERT INTO image_analysis_logs (user_id, image_type, analysis_text, nutrition_data, logged_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, image_type, analysis_text, json.dumps(nutrition_data) if nutrition_data else None, current_time))
    
    conn.commit()
    conn.close()

def get_daily_summary(user_id, date=None):
    """Get daily nutrition and exercise summary"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    conn = sqlite3.connect('health_tracker.db')
    
    # Get nutrition summary - use proper date format for comparison with TEXT timestamps
    nutrition_summary = pd.read_sql_query('''
        SELECT 
            COALESCE(SUM(calories), 0) as total_calories,
            COALESCE(SUM(protein), 0) as total_protein,
            COALESCE(SUM(carbs), 0) as total_carbs,
            COALESCE(SUM(fat), 0) as total_fat,
            COALESCE(SUM(fiber), 0) as total_fiber,
            COALESCE(SUM(sugar), 0) as total_sugar
        FROM nutrition_logs 
        WHERE user_id = ? AND substr(logged_at, 1, 10) = ?
    ''', conn, params=(user_id, date))
    
    # Get exercise summary - use proper date format for comparison with TEXT timestamps
    exercise_summary = pd.read_sql_query('''
        SELECT 
            COALESCE(SUM(calories_burned), 0) as total_calories_burned,
            COALESCE(SUM(duration_minutes), 0) as total_duration
        FROM exercise_logs 
        WHERE user_id = ? AND substr(logged_at, 1, 10) = ?
    ''', conn, params=(user_id, date))
    
    conn.close()
    
    return nutrition_summary, exercise_summary

def get_user_timeline(user_id, days=7):
    """Get user's nutrition and exercise timeline"""
    conn = sqlite3.connect('health_tracker.db')
    
    # Calculate the date threshold for the specified number of days
    threshold_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    
    # Get nutrition logs - use proper date format for TEXT timestamps
    nutrition_df = pd.read_sql_query('''
        SELECT 'Nutrition' as type, food_name as name, calories, logged_at, meal_type
        FROM nutrition_logs 
        WHERE user_id = ? AND logged_at >= ?
        ORDER BY logged_at DESC
    ''', conn, params=(user_id, threshold_date))
    
    # Get exercise logs - use proper date format for TEXT timestamps
    exercise_df = pd.read_sql_query('''
        SELECT 'Exercise' as type, activity_name as name, calories_burned as calories, logged_at, intensity as meal_type
        FROM exercise_logs 
        WHERE user_id = ? AND logged_at >= ?
        ORDER BY logged_at DESC
    ''', conn, params=(user_id, threshold_date))
    
    conn.close()
    
    # Combine and format
    if not nutrition_df.empty:
        nutrition_df['logged_at'] = pd.to_datetime(nutrition_df['logged_at'])
    if not exercise_df.empty:
        exercise_df['logged_at'] = pd.to_datetime(exercise_df['logged_at'])
    
    return nutrition_df, exercise_df

def analyze_text_with_gemini(text, text_type):
    """Analyze extracted text using Gemini LLM"""
    try:
        if text_type == "ingredients":
            prompt = f"""
            You are an expert food ingredient analyzer. Analyze the following ingredients list and provide health-related insights.
            
            Ingredients list:
            {text}
            
            Please provide:
            1. A brief overview of the main ingredients
            2. Potential allergens (common ones like nuts, dairy, soy, etc.)
            3. Additives or preservatives to be cautious about
            4. Any health, religious concerns or benefits
            5. Recommendations for people with specific dietary restrictions
            6. Estimated nutrition information (calories, protein, carbs, fat) based on ingredients
            
            If the text doesn't appear to be a list of food ingredients, please inform the user to try again with a clearer image.
            """
        else:  # nutrition facts
            prompt = f"""
            You are a nutrition expert. Analyze the following nutrition facts and provide health insights.
            
            Nutrition facts:
            {text}
            
            Please provide:
            1. Key nutritional highlights
            2. Calorie content and macronutrient breakdown analysis
            3. Notable vitamins/minerals if mentioned
            4. Health assessment and recommendations
            5. Portion control advice
            6. Any red flags or positive aspects
            
            If the text doesn't appear to be nutrition facts, please inform the user to try again with a clearer image.
            """
        
        response = model.generate_content(prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)

def ocr_space_file(image_data, api_key=None, language='eng'):
    if api_key is None:
        api_key = os.getenv('OCR_SPACE_API_KEY')
    if not api_key:
        return None, "OCR Space API key is not configured. Please check your environment variables."
    url = 'https://api.ocr.space/parse/image'
    def compress_image_to_size(image, max_size_mb=1):
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        quality = 95
        min_quality = 5
        max_quality = 95
        while min_quality <= max_quality:
            quality = (min_quality + max_quality) // 2
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
            size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
            if size_mb <= max_size_mb:
                min_quality = quality + 1
            else:
                max_quality = quality - 1
            if size_mb <= max_size_mb:
                img_byte_arr.seek(0)
                return img_byte_arr.getvalue()
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=5, optimize=True)
        return img_byte_arr.getvalue()
    compressed_image = compress_image_to_size(image_data)
    size_mb = len(compressed_image) / (1024 * 1024)
    if size_mb > 1:
        return None, f"Unable to compress image below 1MB. Current size: {size_mb:.2f}MB"
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
        headers = {'apikey': api_key}
        response = requests.post(
            url,
            files=files,
            data=payload,
            headers=headers,
            timeout=30
        )
        result = response.json()
        if result.get('IsErroredOnProcessing'):
            error_message = result.get('ErrorMessage', 'Unknown error occurred')
            return None, f"Error: {error_message}"
        if not result.get('ParsedResults'):
            return None, "No text was detected in the image."
        return result['ParsedResults'][0]['ParsedText'], None
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def parse_nutrition_facts_from_text(text):
    """Parse nutrition facts from OCR text. Returns a dict with keys: calories, protein_g, carbohydrates_total_g, fat_total_g, fiber_g, sugar_g, sodium_mg, serving_size."""
    import re
    facts = {}
    # Try to find serving size
    serving_match = re.search(r'Serving Size:?\s*([\w\d .]+)', text, re.IGNORECASE)
    if serving_match:
        facts['serving_size'] = serving_match.group(1).strip()
    # Calories
    cal_match = re.search(r'Calories:?\s*(\d+)', text, re.IGNORECASE)
    if cal_match:
        facts['calories'] = int(cal_match.group(1))
    # Protein
    protein_match = re.search(r'Protein:?\s*(\d+(?:\.\d+)?)\s*g', text, re.IGNORECASE)
    if protein_match:
        facts['protein_g'] = float(protein_match.group(1))
    # Carbs
    carbs_match = re.search(r'Carbohydrate[s]?:?\s*(\d+(?:\.\d+)?)\s*g', text, re.IGNORECASE)
    if carbs_match:
        facts['carbohydrates_total_g'] = float(carbs_match.group(1))
    # Fat
    fat_match = re.search(r'Fat:?\s*(\d+(?:\.\d+)?)\s*g', text, re.IGNORECASE)
    if fat_match:
        facts['fat_total_g'] = float(fat_match.group(1))
    # Fiber
    fiber_match = re.search(r'Fiber:?\s*(\d+(?:\.\d+)?)\s*g', text, re.IGNORECASE)
    if fiber_match:
        facts['fiber_g'] = float(fiber_match.group(1))
    # Sugar
    sugar_match = re.search(r'Sugar[s]?:?\s*(\d+(?:\.\d+)?)\s*g', text, re.IGNORECASE)
    if sugar_match:
        facts['sugar_g'] = float(sugar_match.group(1))
    # Sodium
    sodium_match = re.search(r'Sodium:?\s*(\d+(?:\.\d+)?)\s*mg', text, re.IGNORECASE)
    if sodium_match:
        facts['sodium_mg'] = float(sodium_match.group(1))
    return facts

def parse_ingredients_from_text(text):
    """Parse ingredients from OCR text. Returns a list of ingredients if found."""
    import re
    # Look for a line starting with 'Ingredients:'
    match = re.search(r'Ingredients?:?\s*(.*)', text, re.IGNORECASE)
    if match:
        # Split by comma and clean up
        ingredients = [i.strip() for i in match.group(1).split(',') if i.strip()]
        return ingredients
    return []

def compress_image(image_file, max_size=(800, 800), quality=70):
    image = Image.open(image_file)
    image.thumbnail(max_size)
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=quality, optimize=True)
    buf.seek(0)
    return buf

# Initialize database
init_database()

# Page configuration
st.set_page_config(
    page_title="Snappy Gredients",
    page_icon="🥗",
    layout="wide"
)

# Session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

# Main app
def main():
    st.title("🥗 Snappy Gredients")
    
    # Authentication
    if st.session_state.user_id is None:
        show_authentication()
    else:
        show_main_interface()

def show_authentication():
    """Show login/register interface"""
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        st.subheader("Sign In to Snappy Gredients")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In"):
            if username and password:
                user = authenticate_user(username, password)
                if user:
                    st.session_state.user_id = user[0]
                    st.session_state.username = user[1]
                    st.success("Sign in successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Please enter both username and password")

    with tab2:
        st.subheader("Sign Up for Snappy Gredients")
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_password2 = st.text_input("Confirm Password", type="password", key="reg_password2")
        if st.button("Sign Up"):
            if reg_username and reg_password and reg_password2:
                if reg_password != reg_password2:
                    st.error("Passwords do not match!")
                elif register_user(reg_username, reg_password):
                    st.success("Registration successful! Please sign in.")
                else:
                    st.error("Username already exists")
            else:
                st.error("Please fill in all fields")

def show_main_interface():
    """Show main application interface"""
    st.sidebar.title("🥗 Snappy Gredients")
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()
    # --- Track last selected tab to force dashboard refresh ---
    tab_labels = [
        "🏠 SnapShot (Dashboard)",
        "🧑‍🍳 My Profile",
        "🍔 Quick Bite Log",
        "💪 Move & Burn",
        "📸 Snap and Analyze",
        "🕒 My Log Timeline",
        "🤖 Chat with Snappy"
    ]
    if 'last_selected_tab' not in st.session_state:
        st.session_state.last_selected_tab = 0
    selected_tab = st.selectbox("Navigation", tab_labels, index=st.session_state.last_selected_tab, key="main_tabs")
    if selected_tab != tab_labels[st.session_state.last_selected_tab]:
        st.session_state.last_selected_tab = tab_labels.index(selected_tab)
        st.rerun()
    # Render the selected tab
    if selected_tab == tab_labels[0]:
        show_dashboard()
    elif selected_tab == tab_labels[1]:
        show_health_profile()
    elif selected_tab == tab_labels[2]:
        show_nutrition_tracker()
    elif selected_tab == tab_labels[3]:
        show_exercise_tracker()
    elif selected_tab == tab_labels[4]:
        show_image_analysis()
    elif selected_tab == tab_labels[5]:
        show_timeline()
    elif selected_tab == tab_labels[6]:
        show_gemini_chat()

def show_dashboard():
    st.subheader("🏠 SnapShot: Your Day at a Glance")
    selected_date = st.date_input(
        "Select Date",
        value=datetime.now().date(),
        format="YYYY-MM-DD"
    )
    
    # Get summary for selected date
    nutrition_summary, exercise_summary = get_daily_summary(st.session_state.user_id, selected_date.strftime('%Y-%m-%d'))
    
    # Create styled tables for nutrition and exercise data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🍽️ Nutrition Summary")
        if not nutrition_summary.empty:
            nutrition_data = nutrition_summary.iloc[0]
            
            # Create nutrition table
            nutrition_table_data = {
                "Metric": ["Total Calories", "Protein", "Carbohydrates", "Fat", "Fiber", "Sugar"],
                "Value": [
                    f"{nutrition_data['total_calories']:.0f} kcal",
                    f"{nutrition_data['total_protein']:.1f} g",
                    f"{nutrition_data['total_carbs']:.1f} g",
                    f"{nutrition_data['total_fat']:.1f} g",
                    f"{nutrition_data['total_fiber']:.1f} g",
                    f"{nutrition_data['total_sugar']:.1f} g"
                ]
            }
            
            nutrition_df = pd.DataFrame(nutrition_table_data)
            st.dataframe(
                nutrition_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="medium")
                }
            )
        else:
            st.info("No nutrition data logged for this date")
    
    with col2:
        st.subheader("🏃 Exercise Summary")
        if not exercise_summary.empty:
            exercise_data = exercise_summary.iloc[0]
            
            # Create exercise table
            exercise_table_data = {
                "Metric": ["Calories Burned", "Total Duration", "Net Calories"],
                "Value": [
                    f"{exercise_data['total_calories_burned']:.0f} kcal",
                    f"{exercise_data['total_duration']:.0f} min",
                    f"{nutrition_summary['total_calories'].iloc[0] - exercise_data['total_calories_burned']:.0f} kcal"
                ]
            }
            
            exercise_df = pd.DataFrame(exercise_table_data)
            st.dataframe(
                exercise_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="medium")
                }
            )
        else:
            st.info("No exercise data logged for this date")

    # Weekly charts
    st.subheader("📈 Weekly Overview")
    nutrition_df, exercise_df = get_user_timeline(st.session_state.user_id, 7)
    
    if not nutrition_df.empty or not exercise_df.empty:
        # Create combined timeline
        all_activities = []
        if not nutrition_df.empty:
            all_activities.extend(nutrition_df.to_dict('records'))
        if not exercise_df.empty:
            all_activities.extend(exercise_df.to_dict('records'))
        
        if all_activities:
            timeline_df = pd.DataFrame(all_activities)
            timeline_df['logged_at'] = pd.to_datetime(timeline_df['logged_at'])
            timeline_df['date'] = timeline_df['logged_at'].dt.date
            
            # Daily calories chart (improved)
            daily_calories = timeline_df.groupby(['date', 'type'])['calories'].sum().reset_index()
            # Pivot for grouped bar chart
            daily_pivot = daily_calories.pivot(index='date', columns='type', values='calories').fillna(0)
            daily_pivot = daily_pivot.sort_index()
            # Calculate net calories (Nutrition - Exercise)
            daily_pivot['Net Calories'] = daily_pivot.get('Nutrition', 0) - daily_pivot.get('Exercise', 0)
            # Reset index for plotting
            daily_pivot = daily_pivot.reset_index()
            # Bar chart for Nutrition and Exercise
            fig = go.Figure()
            if 'Nutrition' in daily_pivot:
                fig.add_trace(go.Bar(
                    x=daily_pivot['date'],
                    y=daily_pivot['Nutrition'],
                    name='Nutrition (Calories Consumed)',
                    marker_color='green',
                    hovertemplate='Date: %{x}<br>Nutrition: %{y} kcal'
                ))
            if 'Exercise' in daily_pivot:
                fig.add_trace(go.Bar(
                    x=daily_pivot['date'],
                    y=daily_pivot['Exercise'],
                    name='Exercise (Calories Burned)',
                    marker_color='blue',
                    hovertemplate='Date: %{x}<br>Exercise: %{y} kcal'
                ))
            # Line for Net Calories
            fig.add_trace(go.Scatter(
                x=daily_pivot['date'],
                y=daily_pivot['Net Calories'],
                name='Net Calories',
                mode='lines+markers',
                marker_color='orange',
                hovertemplate='Date: %{x}<br>Net Calories: %{y} kcal'
            ))
            fig.update_layout(
                barmode='group',
                title='Daily Nutrition, Exercise, and Net Calories',
                xaxis_title='Date',
                yaxis_title='Calories',
                legend_title='Legend',
                xaxis=dict(type='category', tickformat='%Y-%m-%d'),
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent activities table
            st.subheader("📋 Recent Activities")
            recent_activities = timeline_df.head(10)[['date', 'type', 'name', 'calories', 'meal_type']].copy()
            
            # Ensure date column is datetime and format it properly
            recent_activities['date'] = pd.to_datetime(recent_activities['date'])
            recent_activities['date'] = recent_activities['date'].dt.strftime('%Y-%m-%d %H:%M')
            recent_activities['calories'] = recent_activities['calories'].round(0).astype(int)
            
            st.dataframe(
                recent_activities,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "date": st.column_config.TextColumn("Date & Time", width="medium"),
                    "type": st.column_config.TextColumn("Type", width="small"),
                    "name": st.column_config.TextColumn("Activity", width="large"),
                    "calories": st.column_config.NumberColumn("Calories", width="small"),
                    "meal_type": st.column_config.TextColumn("Category", width="small")
                }
            )
    else:
        st.info("No activities logged yet. Start tracking your nutrition and exercise!")

def show_health_profile():
    st.subheader("🧑 My Profile")
    # Get current health data
    health_data = get_user_health(st.session_state.user_id)
    
    with st.form("health_profile"):
        age = st.number_input("Age", min_value=1, max_value=120, value=health_data[0] if health_data else 25)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=float(health_data[1]) if health_data and health_data[1] else 70.0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=float(health_data[2]) if health_data and health_data[2] else 170.0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if not health_data else ["Male", "Female", "Other"].index(health_data[3]))
        activity_level = st.selectbox("Activity Level", 
                                    ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
                                    index=0 if not health_data else ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"].index(health_data[4]))
        goal = st.selectbox("Goal", ["Lose Weight", "Maintain Weight", "Gain Weight"], 
                           index=0 if not health_data else ["Lose Weight", "Maintain Weight", "Gain Weight"].index(health_data[5]))
        
        if st.form_submit_button("Save Profile"):
            save_user_health(st.session_state.user_id, age, weight, height, gender, activity_level, goal)
            st.success("Health profile updated!")

def show_nutrition_tracker():
    st.subheader("🍔 Quick Bite Log")
    st.subheader("Add a Bite!")

    # --- LLM Food Search State ---
    if 'last_food_result' not in st.session_state:
        st.session_state.last_food_result = None
        st.session_state.last_food_meal_type = 'Breakfast'
        st.session_state.last_food_serving_size = '100g'

    # Search and add food
    with st.form("food_search_form"):
        search_query = st.text_input("Enter your meal:")
        search_submitted = st.form_submit_button("Submit")

    if search_query and search_submitted:
        try:
            food_results = find_food_llm(search_query)
            if food_results and 'foods' in food_results and food_results['foods']:
                selected_food = food_results['foods'][0]
                st.session_state.last_food_result = selected_food
                st.session_state.last_food_meal_type = 'Breakfast'
                st.session_state.last_food_serving_size = selected_food.get('serving_size', '100g')
            else:
                st.session_state.last_food_result = None
                st.info("No food results found. Try a different search term.")
        except Exception as e:
            st.session_state.last_food_result = None
            st.error(f"Error searching for food: {e}")

    # Show last food result if available
    if st.session_state.last_food_result:
        selected_food = st.session_state.last_food_result
        st.subheader("Food Search Result")
        food_details = {
            "Metric": ["Food Name", "Calories", "Protein", "Carbohydrates", "Fat", "Fiber", "Sugar", "Serving Size"],
            "Value": [
                selected_food.get('food_name', 'Unknown Food'),
                f"{selected_food.get('calories', 0):.0f} kcal",
                f"{selected_food.get('protein_g', 0):.1f} g",
                f"{selected_food.get('carbohydrates_total_g', 0):.1f} g",
                f"{selected_food.get('fat_total_g', 0):.1f} g",
                f"{selected_food.get('fiber_g', 0):.1f} g",
                f"{selected_food.get('sugar_g', 0):.1f} g",
                st.session_state.last_food_serving_size
            ]
        }
        food_details_df = pd.DataFrame(food_details)
        st.dataframe(
            food_details_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="medium")
            }
        )
        if selected_food.get('food_description'):
            st.write(f"**Description:** {selected_food['food_description']}")
        st.session_state.last_food_meal_type = st.selectbox(
            "Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"],
            index=["Breakfast", "Lunch", "Dinner", "Snack"].index(st.session_state.last_food_meal_type),
            key="llm_meal_type"
        )
        st.session_state.last_food_serving_size = st.text_input(
            "Serving Size", value=st.session_state.last_food_serving_size, key="llm_serving_size"
        )
        if st.button("Add to Daily Log", key="add_llm_food"):
            food_data = {
                'food_name': selected_food.get('food_name', 'Unknown Food'),
                'nf_calories': selected_food.get('calories', 0),
                'nf_protein': selected_food.get('protein_g', 0),
                'nf_total_carbohydrate': selected_food.get('carbohydrates_total_g', 0),
                'nf_total_fat': selected_food.get('fat_total_g', 0),
                'nf_dietary_fiber': selected_food.get('fiber_g', 0),
                'nf_sugars': selected_food.get('sugar_g', 0),
                'serving_size': st.session_state.last_food_serving_size
            }
            log_nutrition(st.session_state.user_id, food_data, st.session_state.last_food_meal_type)
            st.success(f"Added {food_data['food_name']} to your daily log!")
            st.session_state.last_food_result = None

    # Manual entry
    st.subheader("Manual Entry")
    with st.form("manual_nutrition"):
        food_name = st.text_input("Food Name")
        calories = st.number_input("Calories", min_value=0, value=0)
        protein = st.number_input("Protein (g)", min_value=0.0, value=0.0)
        carbs = st.number_input("Carbohydrates (g)", min_value=0.0, value=0.0)
        fat = st.number_input("Fat (g)", min_value=0.0, value=0.0)
        serving_size = st.text_input("Serving Size", value="100g")
        meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])
        
        if st.form_submit_button("Add Manual Entry"):
            food_data = {
                'food_name': food_name,
                'nf_calories': calories or 0,
                'nf_protein': protein or 0,
                'nf_total_carbohydrate': carbs or 0,
                'nf_total_fat': fat or 0,
                'serving_size': serving_size
            }
            log_nutrition(st.session_state.user_id, food_data, meal_type)
            st.success("Manual entry added!")

def show_exercise_tracker():
    st.subheader("💪 Move & Burn")
    st.subheader("Track Your Moves!")
    # --- LLM Exercise Search State ---
    if 'last_exercise_result' not in st.session_state:
        st.session_state.last_exercise_result = None
        st.session_state.last_exercise_meal_type = 'Moderate'

    # Exercise calculator
    with st.form("exercise_search_form"):
        exercise_query = st.text_input("Enter exercise (e.g., 'Went for a 30 minute jog'):")
        exercise_submitted = st.form_submit_button("Submit")

    if exercise_query and exercise_submitted:
        exercise_data = calories_burned_llm(exercise_query)
        if exercise_data and 'exercises' in exercise_data and exercise_data['exercises']:
            selected_exercise = exercise_data['exercises'][0]
            st.session_state.last_exercise_result = selected_exercise
            st.session_state.last_exercise_meal_type = selected_exercise.get('intensity', 'Moderate').title()
        else:
            st.session_state.last_exercise_result = None
            st.info("Could not calculate calories for this exercise. Try a different format.")

    # Show last exercise result if available
    if st.session_state.last_exercise_result:
        selected_exercise = st.session_state.last_exercise_result
        st.subheader("Exercise Search Result")
        exercise_details = {
            "Metric": ["Exercise Name", "Calories/Hour", "Duration", "Intensity"],
            "Value": [
                selected_exercise.get('name', 'Unknown Exercise'),
                f"{selected_exercise.get('calories_per_hour', 0):.0f} kcal/hr",
                f"{selected_exercise.get('duration_minutes', 60)} minutes",
                selected_exercise.get('intensity', 'moderate').title()
            ]
        }
        exercise_details_df = pd.DataFrame(exercise_details)
        st.dataframe(
            exercise_details_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="medium")
            }
        )
        # Intensity selectbox for possible override
        st.session_state.last_exercise_meal_type = st.selectbox(
            "Intensity", ["Low", "Moderate", "High"],
            index=["Low", "Moderate", "High"].index(st.session_state.last_exercise_meal_type),
            key="llm_ex_intensity"
        )
        if st.button("Add to Exercise Log", key="add_llm_exercise"):
            exercise_log_data = {
                'name': selected_exercise.get('name', 'Unknown Exercise'),
                'nf_calories': selected_exercise.get('calories_per_hour', 0),
                'duration_min': selected_exercise.get('duration_minutes', 60),
                'intensity': st.session_state.last_exercise_meal_type
            }
            log_exercise(st.session_state.user_id, exercise_log_data)
            st.success("Exercise logged!")
            st.session_state.last_exercise_result = None

    # Manual exercise entry
    st.subheader("Manual Exercise Entry")
    with st.form("manual_exercise"):
        activity_name = st.text_input("Activity Name")
        manual_calories_burned = st.number_input("Calories Burned", min_value=0, value=0)
        duration = st.number_input("Duration (minutes)", min_value=1, value=30)
        intensity = st.selectbox("Intensity", ["Low", "Moderate", "High"])
        
        if st.form_submit_button("Add Exercise"):
            exercise_data = {
                'name': activity_name,
                'nf_calories': manual_calories_burned or 0,
                'duration_min': duration or 0,
                'intensity': intensity
            }
            log_exercise(st.session_state.user_id, exercise_data)
            st.success("Exercise logged!")

def show_image_analysis():
    st.subheader("📸 Snap and Analyze")
    image_type = st.selectbox("Select Image Type", ["Ingredients List", "Nutrition Facts Table"])
    uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            compressed = compress_image(uploaded_file)
            image = Image.open(compressed)
            st.image(image, caption="Uploaded Image")
            
            if st.button("Analyze Image"):
                with st.spinner("Extracting text from image..."):
                    ocr_text, error = ocr_space_file(compressed)
                    if error:
                        st.error(f"OCR Error: {error}")
                        return
                    
                    # Use Gemini to analyze the extracted text
                    with st.spinner("Analyzing with AI..."):
                        analysis_text, analysis_error = analyze_text_with_gemini(ocr_text, image_type.lower())
                        
                        if analysis_error:
                            st.error(f"Analysis Error: {analysis_error}")
                        else:
                            st.subheader("🔬 AI Analysis Results")
                            st.markdown(analysis_text)
                    
                    # Parse nutrition facts for logging (if nutrition facts table)
                    if image_type == "Nutrition Facts Table":
                        facts = parse_nutrition_facts_from_text(ocr_text)
                        if facts:
                            st.subheader("📊 Parsed Nutrition Information")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Calories", f"{facts.get('calories', 0):.0f} kcal")
                            with col2:
                                st.metric("Protein", f"{facts.get('protein_g', 0):.1f}g")
                            with col3:
                                st.metric("Carbs", f"{facts.get('carbohydrates_total_g', 0):.1f}g")
                            with col4:
                                st.metric("Fat", f"{facts.get('fat_total_g', 0):.1f}g")
                            
                            if facts.get('serving_size'):
                                st.write(f"**Serving Size:** {facts['serving_size']}")
                            if facts.get('fiber_g'):
                                st.write(f"**Fiber:** {facts['fiber_g']:.1f}g")
                            if facts.get('sugar_g'):
                                st.write(f"**Sugar:** {facts['sugar_g']:.1f}g")
                            if facts.get('sodium_mg'):
                                st.write(f"**Sodium:** {facts['sodium_mg']:.0f}mg")
                            
                            # Option to log nutrition facts
                            if st.button("Log Nutrition Facts to Daily Tracker"):
                                food_data = {
                                    'food_name': 'Extracted Food',
                                    'nf_calories': facts.get('calories', 0),
                                    'nf_protein': facts.get('protein_g', 0),
                                    'nf_total_carbohydrate': facts.get('carbohydrates_total_g', 0),
                                    'nf_total_fat': facts.get('fat_total_g', 0),
                                    'nf_dietary_fiber': facts.get('fiber_g', 0),
                                    'nf_sugars': facts.get('sugar_g', 0),
                                    'serving_size': facts.get('serving_size', '100g')
                                }
                                meal_type = st.selectbox("Select meal type:", ["Breakfast", "Lunch", "Dinner", "Snack"], key="image_meal_type")
                                if st.button("Add to Daily Log", key="add_image_nutrition"):
                                    log_nutrition(st.session_state.user_id, food_data, meal_type)
                                    st.success(f"Added {food_data['food_name']} to your daily log!")
                        else:
                            st.info("Could not parse nutrition facts from the extracted text.")
                    
                    else:  # Ingredients List
                        ingredients = parse_ingredients_from_text(ocr_text)
                        if ingredients:
                            st.subheader("🧾 Parsed Ingredients")
                            for ingredient in ingredients:
                                st.write(f"• {ingredient}")
                        else:
                            st.info("Could not parse ingredients from the extracted text.")
                    
                    # Log the analysis (OCR text + AI analysis)
                    log_image_analysis(st.session_state.user_id, image_type, f"OCR Text: {ocr_text}\n\nAI Analysis: {analysis_text if analysis_text else 'Analysis failed'}", None)
                    st.success("Analysis logged to your timeline!")
        except MemoryError:
            st.error("This image is too large to process on your device. Please try a smaller image.")

def show_timeline():
    st.subheader("🕒 My Log Timeline")
    
    # Date range selector
    days = st.selectbox("Show last", [7, 14, 30, 90], index=0)
    
    # Get timeline data
    nutrition_df, exercise_df = get_user_timeline(st.session_state.user_id, days)
    
    # Display timeline
    if not nutrition_df.empty or not exercise_df.empty:
        st.subheader("Recent Activities")
        
        # Combine activities
        all_activities = []
        if not nutrition_df.empty:
            for _, row in nutrition_df.iterrows():
                all_activities.append({
                    'date': row['logged_at'],
                    'type': 'Nutrition',
                    'name': row['name'],
                    'details': f"{row['calories']:.0f} kcal ({row['meal_type']})"
                })
        
        if not exercise_df.empty:
            for _, row in exercise_df.iterrows():
                all_activities.append({
                    'date': row['logged_at'],
                    'type': 'Exercise',
                    'name': row['name'],
                    'details': f"{row['calories']:.0f} kcal burned ({row['meal_type']})"
                })
        
        # Sort by date
        all_activities.sort(key=lambda x: x['date'], reverse=True)
        
        # Display timeline
        for activity in all_activities:
            with st.expander(f"{activity['date'].strftime('%Y-%m-%d %H:%M')} - {activity['type']}: {activity['name']}"):
                st.write(f"**Details:** {activity['details']}")
        
        # Charts
        st.subheader("Activity Charts")
        
        if not nutrition_df.empty:
            # Nutrition chart
            fig = px.bar(nutrition_df, x='logged_at', y='calories', 
                        title="Daily Nutrition Calories", color='meal_type')
            st.plotly_chart(fig)
        
        if not exercise_df.empty:
            # Exercise chart
            fig = px.bar(exercise_df, x='logged_at', y='calories', 
                        title="Daily Exercise Calories Burned", color='meal_type')
            st.plotly_chart(fig)
    else:
        st.info("No activities logged yet. Start tracking your nutrition and exercise!")

def show_gemini_chat():
    """Chat interface with Gemini as a health, exercise, and nutrition coach. Gives quick, immediate, and helpful responses with access to user profile and logs. Supports markdown formatting for clarity."""
    user_id = st.session_state.user_id
    username = st.session_state.username
    # --- Retrieve user profile and logs for context ---
    health = get_user_health(user_id)
    nutrition_summary, exercise_summary = get_daily_summary(user_id)
    # Format health profile string
    if health:
        health_profile_str = (
            f"Age: {health[0]}, Weight: {health[1]} kg, Height: {health[2]} cm, "
            f"Gender: {health[3]}, Activity Level: {health[4]}, Goal: {health[5]}"
        )
    else:
        health_profile_str = "No health profile information available."
    # Format nutrition and exercise summary
    nutrition_str = nutrition_summary.to_dict() if hasattr(nutrition_summary, 'to_dict') else nutrition_summary
    exercise_str = exercise_summary.to_dict() if hasattr(exercise_summary, 'to_dict') else exercise_summary
    # --- System prompt for Gemini ---
    system_prompt = (
        f"You are Snappy, a friendly, expert exercise and nutrition coach. "
        f"Give short, organized responses. You can use markdown formatting (including tables, lists, bold, etc.) to make your responses more human readable and intuitive. "
        f"Here is the user's health profile: {health_profile_str}\n"
        f"Today's nutrition summary: {nutrition_str}\n"
        f"Today's exercise summary: {exercise_str}\n"
        f"Encourage healthy habits, answer questions, and help users make good choices. "
        f"Stay positive and focus on actionable tips."
    )
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])  # Always render markdown
    # Accept user input
    if prompt := st.chat_input("Message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Compose conversation for Gemini, always prepending the system prompt
            conversation = system_prompt + "\n"
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    conversation += f"User: {msg['content']}\n"
                else:
                    conversation += f"Assistant: {msg['content']}\n"
            try:
                response = model.generate_content(conversation)
                reply = response.text.strip()
            except Exception as e:
                reply = f"[Error from Gemini: {e}]"
            # Streaming simulation: display reply word by word
            stream_placeholder = st.empty()
            streamed = ""
            for chunk in reply.split():
                streamed += chunk + " "
                stream_placeholder.markdown(streamed)  # Always render markdown
                time.sleep(0.05)  # Adjust speed as desired
            stream_placeholder.markdown(streamed)
        st.session_state.messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    # Initialize database on app start
    init_database()
    main()
