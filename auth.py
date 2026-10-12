"""
DeepGuard AI - Local Authentication Module
==========================================
Provides secure local SQLite-based authentication with:
- Per-user cryptographic salt generation (secrets.token_bytes)
- PBKDF2-HMAC-SHA256 password hashing (100,000 iterations)
- Zero plaintext password storage
- Defensive input sanitization and duplicate handling
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Tuple, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")
HASH_ITERATIONS = 100_000


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with row factory configured."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the users table if it does not already exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _hash_password(password: str, salt_hex: str) -> str:
    """Derives a secure hash using PBKDF2-HMAC-SHA256."""
    salt_bytes = bytes.fromhex(salt_hex)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        HASH_ITERATIONS
    )
    return derived.hex()


def register_user(username: str, password: str, db_path: str = DEFAULT_DB_PATH) -> Tuple[bool, str]:
    """
    Registers a new user securely.
    Returns: (success: bool, message: str)
    """
    init_db(db_path)
    
    # Validation
    username = (username or "").strip()
    if not username:
        return False, "Username cannot be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    if not username.replace("_", "").isalnum():
        return False, "Username can only contain letters, numbers, and underscores."
    
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."

    # Cryptographic salt (32 bytes = 256 bits)
    salt_hex = secrets.token_bytes(32).hex()
    password_hash = _hash_password(password, salt_hex)
    created_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt_hex, created_at)
        )
        conn.commit()
        return True, f"User '{username}' registered successfully."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists. Please choose a different username."
    except Exception as e:
        return False, f"Database error during registration: {str(e)}"
    finally:
        conn.close()


def authenticate_user(username: str, password: str, db_path: str = DEFAULT_DB_PATH) -> Tuple[bool, str]:
    """
    Authenticates an existing user.
    Returns: (success: bool, message: str)
    """
    init_db(db_path)
    
    username = (username or "").strip()
    if not username or not password:
        return False, "Please enter both username and password."

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            return False, "Invalid username or password."
            
        stored_hash = row["password_hash"]
        salt_hex = row["salt"]
        
        # Hash provided password with stored salt
        candidate_hash = _hash_password(password, salt_hex)
        
        # Constant-time comparison to prevent timing attacks
        if secrets.compare_digest(stored_hash, candidate_hash):
            return True, f"Authentication successful for user '{username}'."
        else:
            return False, "Invalid username or password."
    except Exception as e:
        return False, f"Database error during authentication: {str(e)}"
    finally:
        conn.close()


def user_exists(username: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Checks whether a username exists in the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username.strip(),))
    row = cursor.fetchone()
    conn.close()
    return row is not None


if __name__ == "__main__":
    print("[*] Testing auth.py SQLite PBKDF2 authentication system...")
    test_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_users.db")
    if os.path.exists(test_db):
        os.remove(test_db)
        
    init_db(test_db)
    print("[*] Initialized test database.")
    
    # Test registration
    ok, msg = register_user("student_admin", "secure_pass123", db_path=test_db)
    print(f"[*] Register student_admin: ok={ok}, msg='{msg}'")
    assert ok, "Registration failed!"

    # Test duplicate
    ok_dup, msg_dup = register_user("student_admin", "another_pass", db_path=test_db)
    print(f"[*] Register duplicate: ok={ok_dup}, msg='{msg_dup}'")
    assert not ok_dup, "Duplicate registration should fail!"

    # Test valid login
    ok_login, msg_login = authenticate_user("student_admin", "secure_pass123", db_path=test_db)
    print(f"[*] Valid login: ok={ok_login}, msg='{msg_login}'")
    assert ok_login, "Valid login failed!"

    # Test invalid password
    ok_bad, msg_bad = authenticate_user("student_admin", "wrong_password", db_path=test_db)
    print(f"[*] Invalid login: ok={ok_bad}, msg='{msg_bad}'")
    assert not ok_bad, "Invalid login should fail!"

    # Clean up test database
    if os.path.exists(test_db):
        os.remove(test_db)

    # Initialize production users.db with a default demo account for immediate viva use
    prod_ok, prod_msg = register_user("admin", "admin123", db_path=DEFAULT_DB_PATH)
    print(f"[*] Production account setup: {prod_msg}")
    print("[OK] auth.py self-test completed successfully!")
