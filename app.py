import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import sqlite3
import openai
import smtplib
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Skin Disease Detection System",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Expert Doctors List
EXPERT_DOCTORS = [
    "Dr. Ravi Kumar - Apollo Hospital, Chennai",
    "Dr. Priya Sharma - Fortis Hospital, Coimbatore",
    "Dr. Karthik Reddy - MIOT Hospital, Chennai",
    "Dr. Anitha Raj - PSG Hospital, Coimbatore",
    "Dr. Suresh Babu - SRM Hospital, Trichy"
]

# Define image classes
CLASS_NAMES = ['acne', 'allergy', 'black_spots', 'clear_skin', 'dermatitis', 'eczema', 'melanoma', 'not_skin_image', 'psoriasis', 'rosacea', 'vitiligo']

# Disease Information Dictionary
DISEASE_INFO = {
    'acne': {
        'description': 'A common skin condition that occurs when hair follicles become clogged with oil and dead skin cells.',
        'symptoms': ['Whiteheads', 'Blackheads', 'Pimples', 'Oily skin', 'Scarring'],
        'treatment': ['Topical medications', 'Oral medications', 'Lifestyle changes', 'Professional treatments'],
        'prevention': ['Regular cleansing', 'Avoid touching face', 'Use non-comedogenic products', 'Manage stress']
    },
    'allergy': {
        'description': 'An immune system reaction to substances that are usually harmless.',
        'symptoms': ['Rash', 'Itching', 'Redness', 'Swelling', 'Hives'],
        'treatment': ['Antihistamines', 'Topical steroids', 'Avoiding triggers', 'Allergy shots'],
        'prevention': ['Identify triggers', 'Use hypoallergenic products', 'Keep skin moisturized']
    },
    'black_spots': {
        'description': 'Dark spots or hyperpigmentation on the skin caused by various factors.',
        'symptoms': ['Dark patches', 'Uneven skin tone', 'Sun damage'],
        'treatment': ['Topical lightening agents', 'Chemical peels', 'Laser therapy'],
        'prevention': ['Sun protection', 'Regular exfoliation', 'Healthy skincare routine']
    },
    'clear_skin': {
        'description': 'Healthy skin without any visible conditions or issues.',
        'symptoms': ['Even tone', 'Smooth texture', 'No visible blemishes'],
        'treatment': ['Maintain current routine', 'Regular check-ups'],
        'prevention': ['Healthy lifestyle', 'Proper skincare', 'Sun protection']
    },
    'dermatitis': {
        'description': 'Inflammation of the skin causing itchy rashes and blisters.',
        'symptoms': ['Itching', 'Redness', 'Rash', 'Dry skin', 'Blisters'],
        'treatment': ['Moisturizers', 'Topical steroids', 'Antihistamines'],
        'prevention': ['Avoid irritants', 'Regular moisturizing', 'Stress management']
    },
    'eczema': {
        'description': 'A condition that makes your skin red and itchy.',
        'symptoms': ['Dry skin', 'Itching', 'Red patches', 'Cracked skin'],
        'treatment': ['Moisturizers', 'Topical steroids', 'Antihistamines'],
        'prevention': ['Regular moisturizing', 'Avoid triggers', 'Gentle skincare']
    },
    'melanoma': {
        'description': 'A serious form of skin cancer that develops in melanocytes.',
        'symptoms': ['New moles', 'Changes in existing moles', 'Irregular borders', 'Color changes'],
        'treatment': ['Surgical removal', 'Chemotherapy', 'Radiation therapy', 'Immunotherapy'],
        'prevention': ['Sun protection', 'Regular skin checks', 'Avoid tanning beds']
    },
    'not_skin_image': {
        'description': 'The uploaded image does not appear to be a skin condition.',
        'symptoms': ['N/A'],
        'treatment': ['Please upload a clear image of the affected skin area'],
        'prevention': ['N/A']
    },
    'psoriasis': {
        'description': 'A skin disorder that causes skin cells to multiply up to 10 times faster than normal.',
        'symptoms': ['Red patches', 'Silvery scales', 'Dry skin', 'Itching'],
        'treatment': ['Topical treatments', 'Light therapy', 'Systemic medications'],
        'prevention': ['Stress management', 'Avoid triggers', 'Regular moisturizing']
    },
    'rosacea': {
        'description': 'A chronic skin condition that causes redness and visible blood vessels.',
        'symptoms': ['Facial redness', 'Visible blood vessels', 'Swollen bumps', 'Eye problems'],
        'treatment': ['Topical medications', 'Oral antibiotics', 'Laser therapy'],
        'prevention': ['Avoid triggers', 'Sun protection', 'Gentle skincare']
    },
    'vitiligo': {
        'description': 'A condition that causes the loss of skin color in patches.',
        'symptoms': ['White patches', 'Premature whitening of hair', 'Loss of color in tissues'],
        'treatment': ['Topical steroids', 'Light therapy', 'Depigmentation'],
        'prevention': ['Sun protection', 'Stress management', 'Healthy lifestyle']
    }
}

# Email Configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Database connection
def get_db_connection():
    try:
        conn = sqlite3.connect('skin_disease.db', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Initialize database tables
def init_db():
    try:
        conn = get_db_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        
        # Create appointments table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            doctor TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create chat history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create prediction history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            predicted_disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"Error initializing database: {e}")

# Function to Send Email (any SMTP server)
def send_email(to_email, doctor, date, time):
    if not (EMAIL_SENDER and EMAIL_PASSWORD):
        st.error("Email credentials are missing. Please set EMAIL_SENDER and EMAIL_PASSWORD in your .env file.")
        return False
    try:
        msg = MIMEMultipart()
        msg['Subject'] = "Doctor Appointment Confirmation"
        msg['From'] = EMAIL_SENDER
        msg['To'] = to_email
        body = f"""
        Dear Patient,\n\nYour appointment has been confirmed with the following details:\nDoctor: {doctor}\nDate: {date}\nTime: {time}\n\nPlease arrive 15 minutes before your scheduled time.\nIf you need to reschedule, please contact us at least 24 hours in advance.\n\nBest regards,\nSkin Disease Detection Team
        """
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

def send_sms_fast2sms(to_number, message):
    api_key = os.getenv('FAST2SMS_API_KEY')
    if not api_key:
        st.error("Fast2SMS API key not set in .env")
        return False
    url = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        "sender_id": "FSTSMS",
        "message": message,
        "language": "english",
        "route": "p",
        "numbers": to_number,
    }
    headers = {
        "authorization": api_key,
        "Content-Type": "application/x-www-form-urlencoded",
        "Cache-Control": "no-cache"
    }
    try:
        response = requests.post(url, data=payload, headers=headers)
        result = response.json()
        if result.get('return'):
            st.success("SMS sent successfully!")
            return True
        else:
            st.error(f"SMS failed: {result.get('message')}")
            return False
    except Exception as e:
        st.error(f"Error sending SMS: {e}")
        return False

# Function to Get ChatGPT Response (OpenAI >= 1.0.0)
def chat_with_gpt(user_message):
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file.")
        return "OpenAI API key missing."
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant specializing in skin diseases."},
                {"role": "user", "content": user_message}
            ]
        )
        bot_response = response.choices[0].message.content
        # Save chat history
        conn = get_db_connection()
        if conn is not None:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO chat_history (user_message, bot_response) 
            VALUES (?, ?)
            """, (user_message, bot_response))
            conn.commit()
            cursor.close()
            conn.close()
        return bot_response
    except Exception as e:
        st.error(f"Error with chatbot: {e}")
        return "I apologize, but I'm having trouble processing your request right now."

# Function to Get Chat History
def get_chat_history():
    try:
        conn = get_db_connection()
        if conn is None:
            return []
        
        cursor = conn.cursor()
        cursor.execute("""
        SELECT user_message, bot_response, timestamp 
        FROM chat_history 
        ORDER BY timestamp DESC 
        LIMIT 10
        """)
        history = cursor.fetchall()
        cursor.close()
        conn.close()
        return history
    except sqlite3.Error as e:
        st.error(f"Error getting chat history: {e}")
        return []

# Load the trained model
@st.cache_resource
def download_model():
    file_id = "104aRXp5pKcSIVIui2KRPkzYnb4HTILlt"  # your actual file ID
    destination = "models/final_model.h5"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    if not os.path.exists(destination):
        print("Downloading model...")
        response = requests.get(url, allow_redirects=True)
        with open(destination, "wb") as f:
            f.write(response.content)
        print("Model downloaded.")
    else:
        print("Model already exists locally.")


def load_model():
    try:
        download_model_from_drive()
        model = tf.keras.models.load_model("models/final_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    # Resize image to match model's expected input size
    image = cv2.resize(image, (192, 192))
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # Normalize
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Function to save prediction to history
def save_prediction(image_path, predicted_disease, confidence):
    try:
        conn = get_db_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO prediction_history (image_path, predicted_disease, confidence)
        VALUES (?, ?, ?)
        """, (image_path, predicted_disease, confidence))
        conn.commit()
        cursor.close()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"Error saving prediction: {e}")

def get_appointment_status(email):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM appointments 
    WHERE email = ? 
    ORDER BY created_at DESC
    """, (email,))
    appointments = cursor.fetchall()
    conn.close()
    return appointments

# Initialize database
init_db()

# --- Navigation State ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# Sidebar Navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Home", "Disease Detection", "Book Appointment", "FAQ", "History"], index=["Home", "Disease Detection", "Book Appointment", "FAQ", "History"].index(st.session_state['page']))
if selected_page != st.session_state['page']:
    st.session_state['page'] = selected_page
    st.rerun()

# Home Page
if st.session_state['page'] == "Home":
    st.header("Welcome to Skin Disease Detection System")
    st.write("""
    This system helps you:
    - Detect skin diseases using AI
    - Book appointments with expert dermatologists
    - Get instant answers to your questions
    - Track your medical history
    """)
    # Quick Actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔍 Start Detection"):
            st.session_state['page'] = "Disease Detection"
            st.rerun()
    with col2:
        if st.button("📅 Book Appointment"):
            st.session_state['page'] = "Book Appointment"
            st.rerun()
    with col3:
        if st.button("❓ FAQ"):
            st.session_state['page'] = "FAQ"
            st.rerun()

# Disease Detection Page
elif st.session_state['page'] == "Disease Detection":
    st.header("🔍 Skin Disease Detection")
    
    uploaded_file = st.file_uploader("Upload an image of the affected area", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                # Preprocess image
                img_array = np.array(image)
                img_array = preprocess_image(img_array)
                
                # Make prediction
                model = load_model()
                if model is not None:
                    prediction = model.predict(img_array)
                    class_index = np.argmax(prediction)
                    confidence = prediction[0][class_index] * 100
                    disease = CLASS_NAMES[class_index]
                    info = DISEASE_INFO[disease]

                    # Confidence threshold logic
                    if confidence < 70:
                        st.warning("The uploaded image does not appear to be a human skin condition. Please upload a clear image of the affected skin area.")
                    else:
                        # Display results
                        st.success(f"Predicted Disease: {disease}")
                        st.info(f"Confidence: {confidence:.2f}%")
                        # Display disease information
                        st.subheader("Disease Information")
                        st.write(f"**Description:** {info['description']}")
                        st.write("**Common Symptoms:**")
                        for symptom in info['symptoms']:
                            st.write(f"- {symptom}")
                        st.write("**Treatment Options:**")
                        for treatment in info['treatment']:
                            st.write(f"- {treatment}")
                        st.write("**Prevention Tips:**")
                        for prevention in info['prevention']:
                            st.write(f"- {prevention}")
                        # Save prediction
                        save_prediction(uploaded_file.name, disease, confidence)

# Appointment Booking Page
elif st.session_state['page'] == "Book Appointment":
    st.header("📅 Book an Appointment")
    
    with st.form("appointment_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")
        doctor = st.selectbox("Select Doctor", EXPERT_DOCTORS)
        date = st.date_input("Preferred Date")
        time = st.time_input("Preferred Time")
        
        submitted = st.form_submit_button("Book Appointment")
        
        if submitted:
            if name and email and phone and doctor and date and time:
                # Save appointment
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO appointments (name, email, phone, doctor, date, time)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (name, email, phone, doctor, date.strftime('%Y-%m-%d'), time.strftime('%H:%M')))
                conn.commit()
                
                # Send notifications
                email_sent = send_email(email, doctor, date.strftime('%Y-%m-%d'), time.strftime('%H:%M'))
                sms_message = f"Your appointment with {doctor} is confirmed for {date.strftime('%Y-%m-%d')} at {time.strftime('%H:%M')}"
                sms_sent = send_sms_fast2sms(phone, sms_message)
                
                if email_sent and sms_sent:
                    st.success("Appointment booked successfully! Check your email and phone for confirmation.")
                else:
                    st.warning("Appointment booked, but there were issues sending notifications.")
            else:
                st.error("Please fill in all fields.")

    if 'appointment_submitted' in st.session_state:
        st.subheader("Your Appointments")
        appointments = get_appointment_status(st.session_state['appointment_submitted'])
        
        if not appointments:
            st.info("No appointments found")
        else:
            for appointment in appointments:
                with st.expander(f"Appointment on {appointment['date']} at {appointment['time']}"):
                    st.write(f"**Doctor:** {appointment['doctor']}")
                    st.write(f"**Date:** {appointment['date']}")
                    st.write(f"**Time:** {appointment['time']}")
                    st.write(f"**Status:** {appointment['status'].upper()}")
                    
                    if appointment['status'] == 'approved':
                        st.success("Your appointment has been approved!")
                    elif appointment['status'] == 'rejected':
                        st.error("Your appointment has been rejected")
                    else:
                        st.info("Your appointment is pending approval")

# FAQ Page
elif st.session_state['page'] == "FAQ":
    st.header("❓ Frequently Asked Questions")
    faqs = {
        "What is acne?": "Acne is a common skin condition that occurs when hair follicles become clogged with oil and dead skin cells, causing pimples, blackheads, and whiteheads.",
        "What is eczema?": "Eczema is a condition that makes your skin red, inflamed, and itchy. It is also known as atopic dermatitis.",
        "What is melanoma?": "Melanoma is a serious form of skin cancer that develops in the cells that produce melanin (the pigment that gives your skin its color).",
        "How can I prevent skin diseases?": "Maintain good hygiene, use sunscreen, avoid sharing personal items, and consult a dermatologist for persistent issues.",
        "When should I see a doctor?": "If you notice new, changing, or unusual skin lesions, or if a rash is painful, spreading, or not improving, see a doctor." 
    }
    for question, answer in faqs.items():
        st.subheader(question)
        st.write(answer)

# History Page
elif st.session_state['page'] == "History":
    st.header("📋 Your History")
    
    # Tabs for different types of history
    tab1, tab2, tab3 = st.tabs(["Predictions", "Appointments", "Chat History"])
    
    with tab1:
        st.subheader("Recent Predictions")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prediction_history ORDER BY timestamp DESC LIMIT 10")
        predictions = cursor.fetchall()
        
        for pred in predictions:
            st.write(f"**Disease:** {pred[2]}")
            st.write(f"**Confidence:** {pred[3]:.2f}%")
            st.write(f"**Date:** {pred[4]}")
            st.write("---")
    
    with tab2:
        st.subheader("Appointment History")
        cursor.execute("SELECT * FROM appointments ORDER BY date DESC, time DESC LIMIT 10")
        appointments = cursor.fetchall()
        
        for apt in appointments:
            st.write(f"**Doctor:** {apt[4]}")
            st.write(f"**Date:** {apt[5]}")
            st.write(f"**Time:** {apt[6]}")
            st.write(f"**Status:** {apt[7]}")
            st.write("---")
    
    with tab3:
        st.subheader("Chat History")
        cursor.execute("SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT 10")
        chats = cursor.fetchall()
        
        for chat in chats:
            st.write(f"**You:** {chat[1]}")
            st.write(f"**AI:** {chat[2]}")
            st.write(f"**Time:** {chat[3]}")
            st.write("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>© 2025 Skin Disease Detection System | For medical emergencies, please call your local emergency services</p>
</div>
""", unsafe_allow_html=True) 
