#!/usr/bin/env python3
"""
Quick test to check iMessage database access
Save this as: test_db_access.py in your root directory
"""

from pathlib import Path
import os

def test_db_access():
    print("🔍 Testing iMessage Database Access")
    print("=" * 40)
    
    # Test 1: Check if database file exists
    db_path = Path("~/Library/Messages/chat.db").expanduser()
    print(f"Database path: {db_path}")
    print(f"Exists: {db_path.exists()}")
    
    if not db_path.exists():
        print("❌ Database file not found!")
        print("This could mean:")
        print("  - You're not on macOS")
        print("  - iMessage is not set up")
        print("  - Messages app has never been opened")
        return False
    
    # Test 2: Check WAL file
    wal_path = db_path.with_suffix(".db-wal")
    print(f"WAL file exists: {wal_path.exists()}")
    
    # Test 3: Check file permissions
    print(f"Database readable: {os.access(db_path, os.R_OK)}")
    
    # Test 4: Try basic SQLite connection
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5;")
        tables = cursor.fetchall()
        conn.close()
        
        print(f"✅ Successfully connected to database")
        print(f"Found tables: {[t[0] for t in tables]}")
        return True
        
    except Exception as e:
        print(f"❌ Cannot connect to database: {e}")
        print("You probably need to grant 'Full Disk Access' permission")
        return False

if __name__ == "__main__":
    success = test_db_access()
    
    if success:
        print("\n🎉 Database access working!")
        print("Next: Test the iMessage reader")
    else:
        print("\n❌ Database access failed")
        print("Fix permissions first:")
        print("1. System Preferences > Security & Privacy > Privacy")
        print("2. Select 'Full Disk Access'")
        print("3. Add Terminal or your IDE")
        print("4. Restart your terminal/IDE")