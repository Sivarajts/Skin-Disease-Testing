import streamlit as st
import sqlite3
from datetime import datetime

def get_db_connection():
    conn = sqlite3.connect('skin_disease.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_doctor_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT NOT NULL,
        hospital TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()
def add_default_doctor():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO doctors (username, password, name, hospital) 
        VALUES (?, ?, ?, ?)
        """, ("doctor", "doctor123", "Dr. John Doe", "City Hospital"))
        conn.commit()
    except sqlite3.IntegrityError:
        # Username already exists
        pass
    finally:
        conn.close()

add_default_doctor()


def verify_doctor(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM doctors WHERE username = ? AND password = ?", (username, password))
    doctor = cursor.fetchone()
    conn.close()
    return doctor

def get_pending_appointments():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM appointments 
    WHERE status = 'pending' 
    ORDER BY date, time
    """)
    appointments = cursor.fetchall()
    conn.close()
    return appointments

def update_appointment_status(appointment_id, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE appointments 
    SET status = ? 
    WHERE id = ?
    """, (status, appointment_id))
    conn.commit()
    conn.close()

# Initialize the database
init_doctor_db()

# Set page config
st.set_page_config(
    page_title="Doctor's Portal - Skin Disease Detection System",
    page_icon="👨‍⚕️",
    layout="wide"
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
    </style>
    """, unsafe_allow_html=True)

# Login form
st.title("Doctor's Portal")
st.subheader("Login to manage appointments")

with st.form("doctor_login"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

if submit:
    doctor = verify_doctor(username, password)
    if doctor:
        st.session_state['doctor'] = doctor
        st.success("Login successful!")
    else:
        st.error("Invalid username or password")

# If logged in, show appointment management
if 'doctor' in st.session_state:
    st.write(f"Welcome, Dr. {st.session_state['doctor']['name']}")
    
    # Show pending appointments
    st.subheader("Pending Appointments")
    appointments = get_pending_appointments()
    
    if not appointments:
        st.info("No pending appointments")
    else:
        for appointment in appointments:
            with st.expander(f"Appointment with {appointment['name']} on {appointment['date']} at {appointment['time']}"):
                st.write(f"**Patient Name:** {appointment['name']}")
                st.write(f"**Email:** {appointment['email']}")
                st.write(f"**Phone:** {appointment['phone']}")
                st.write(f"**Date:** {appointment['date']}")
                st.write(f"**Time:** {appointment['time']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Approve", key=f"approve_{appointment['id']}"):
                        update_appointment_status(appointment['id'], 'approved')
                        st.success("Appointment approved!")
                        st.rerun()
                with col2:
                    if st.button("Reject", key=f"reject_{appointment['id']}"):
                        update_appointment_status(appointment['id'], 'rejected')
                        st.error("Appointment rejected!")
                        st.rerun()
    
    # Logout button
    if st.button("Logout"):
        del st.session_state['doctor']
        st.rerun() 
