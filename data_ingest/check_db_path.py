# scripts/check_db_path.py
from pathlib import Path

def test_db_path():
    db_path = Path("~/Library/Messages/chat.db").expanduser()
    print("Resolved path:", db_path)
    print("Exists?      ", db_path.exists())

    # Also check the WAL file (Write-Ahead Log)
    wal_path = db_path.with_suffix(".db-wal")
    print("WAL exists?  ", wal_path.exists())

if __name__ == "__main__":
    test_db_path()

