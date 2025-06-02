import sys
import os
import sqlite3
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Dict  # <-- ADDED Dict HERE
from datetime import datetime

import pandas as pd
from pydantic import BaseModel

# Ensure project root is in Python path for config imports
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings


class iMessage(BaseModel):
    """Pydantic model for iMessage data"""
    text: Optional[str]
    is_from_me: bool
    contact_id: Optional[str]
    contact_name: Optional[str]
    timestamp: datetime
    chat_id: str
    service: str  # iMessage, SMS, etc.


class Contact(BaseModel):
    """Pydantic model for contact information"""
    identifier: str  # phone number or email
    display_name: Optional[str]
    message_count: int = 0
    relationship_type: Optional[str] = None


class iMessageCollector:
    """Collects and processes iMessage data from macOS chat.db"""

    def __init__(self, db_path: Optional[str] = None):
        # Expand ~ and fallback to settings
        self.db_path = Path(db_path or settings.IMESSAGE_DB_PATH).expanduser()
        self.logger = logging.getLogger(__name__)

    def verify_access(self) -> bool:
        """Verify the database file exists"""
        if not self.db_path.exists():
            self.logger.error(f"iMessage database not found at {self.db_path}")
            return False
        return True

    def _get_connection(self) -> sqlite3.Connection:
        """
        Copy the main DB + WAL (if exists) to a temp directory,
        then open a read-only URI connection.
        """
        tmpdir = tempfile.mkdtemp()
        tmp_db = Path(tmpdir) / "chat.db"
        tmp_wal = tmp_db.with_suffix(".db-wal")

        # Copy main DB
        shutil.copy(self.db_path, tmp_db)
        # Copy WAL if present
        wal = self.db_path.with_suffix(".db-wal")
        if wal.exists():
            shutil.copy(wal, tmp_wal)

        uri = f"file:{tmp_db}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        return conn

    def collect_messages(self, limit: Optional[int] = None) -> List[iMessage]:
        """Collect last `limit` messages from iMessage database"""
        if not self.verify_access():
            raise RuntimeError("Cannot access iMessage database. Check permissions.")

        query = (
            """
            SELECT
                m.text,
                m.is_from_me,
                h.id as contact_id,
                h.uncanonicalized_id as contact_name,
                datetime(m.date/1000000000 + strftime('%s','2001-01-01'), 'unixepoch') as timestamp,
                m.cache_roomnames as chat_id,
                m.service
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL AND m.text != ''
            ORDER BY m.date DESC
            """
        )
        if limit:
            query += f" LIMIT {limit}"

        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()

        messages: List[iMessage] = []
        for _, row in df.iterrows():
            try:
                messages.append(
                    iMessage(
                        text=row['text'],
                        is_from_me=bool(row['is_from_me']),
                        contact_id=row['contact_id'],
                        contact_name=row['contact_name'],
                        timestamp=pd.to_datetime(row['timestamp']),
                        chat_id=row['chat_id'] or 'unknown',
                        service=row['service'] or 'iMessage'
                    )
                )
            except Exception as e:
                self.logger.warning(f"Skipping malformed message: {e}")
        self.logger.info(f"Collected {len(messages)} messages")
        return messages

    def get_contacts(self) -> List[Contact]:
        """Extract contact info and message counts"""
        if not self.verify_access():
            raise RuntimeError("Cannot access iMessage database. Check permissions.")

        query = (
            """
            SELECT
                h.id as identifier,
                h.uncanonicalized_id as display_name,
                COUNT(m.ROWID) as message_count
            FROM handle h
            LEFT JOIN message m ON h.ROWID = m.handle_id
            GROUP BY h.id, h.uncanonicalized_id
            HAVING message_count > 0
            ORDER BY message_count DESC
            """
        )

        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()

        contacts: List[Contact] = []
        for _, row in df.iterrows():
            contacts.append(
                Contact(
                    identifier=row['identifier'],
                    display_name=row['display_name'],
                    message_count=int(row['message_count'])
                )
            )
        self.logger.info(f"Found {len(contacts)} contacts")
        return contacts

    def analyze_your_messages(self, messages: List[iMessage]) -> Dict:
        """Analyze patterns in messages you sent"""
        your_msgs = [m for m in messages if m.is_from_me]
        if not your_msgs:
            return {"error": "No messages from you found"}

        analysis = {
            "total_messages": len(your_msgs),
            "date_range": {
                "earliest": min(m.timestamp for m in your_msgs),
                "latest": max(m.timestamp for m in your_msgs)
            },
            "avg_message_length": sum(len(m.text or "") for m in your_msgs) / len(your_msgs),
            "services_used": list({m.service for m in your_msgs}),
        }
        return analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = iMessageCollector()
    try:
        print("🧪 Testing iMessage Collection...")
        msgs = collector.collect_messages(limit=10)
        print(f"✅ Successfully collected {len(msgs)} messages")
        
        your_msgs = [m for m in msgs if m.is_from_me]
        print(f"📱 Found {len(your_msgs)} messages from you")
        
        # Show a few sample messages (safely)
        for i, msg in enumerate(msgs[:3], 1):
            sender = "You" if msg.is_from_me else "Contact"
            preview = (msg.text[:30] + "...") if len(msg.text) > 30 else msg.text
            print(f"{i:2d}> [{sender}] {preview}")
            
    except Exception as e:
        print("❌ Error fetching messages:", e)
        print("\nTroubleshooting:")
        print("1. Grant 'Full Disk Access' to your terminal/IDE")
        print("2. Make sure iMessage is set up on this Mac")
        print("3. Try running: python test_db_access.py")