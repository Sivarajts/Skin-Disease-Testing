import sqlite3

def add_default_doctor():
    conn = sqlite3.connect('skin_disease.db')
    cursor = conn.cursor()
    
    # Create doctors table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT NOT NULL,
        hospital TEXT NOT NULL
    )
    """)
    
    # Add default doctor
    try:
        cursor.execute("""
        INSERT INTO doctors (username, password, name, hospital)
        VALUES (?, ?, ?, ?)
        """, ('doctor', 'doctor123', 'Dr. Ravi Kumar', 'Apollo Hospital, Chennai'))
        conn.commit()
        print("Default doctor account created successfully!")
    except sqlite3.IntegrityError:
        print("Doctor account already exists!")
    
    conn.close()

if __name__ == "__main__":
    add_default_doctor() 