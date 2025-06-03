import sys
import os
import sqlite3
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Dict
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
                m.service,
                -- Better chat identification
                CASE 
                    WHEN c.chat_identifier IS NOT NULL THEN c.chat_identifier
                    WHEN c.display_name IS NOT NULL THEN c.display_name
                    WHEN h.id IS NOT NULL THEN h.id
                    ELSE 'unknown_' || m.ROWID
                END as chat_id,
                c.display_name as chat_display_name,
                c.group_id as group_chat_id
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
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
                # Create a better chat_id
                chat_id = self._determine_chat_id(row)
                
                messages.append(
                    iMessage(
                        text=row['text'],
                        is_from_me=bool(row['is_from_me']),
                        contact_id=row['contact_id'],
                        contact_name=row['contact_name'],
                        timestamp=pd.to_datetime(row['timestamp']),
                        chat_id=chat_id,
                        service=row['service'] or 'iMessage'
                    )
                )
            except Exception as e:
                self.logger.warning(f"Skipping malformed message: {e}")
        self.logger.info(f"Collected {len(messages)} messages")
        return messages

    def _determine_chat_id(self, row) -> str:
        """
        Determine the best chat identifier from available data
        Priority: chat_identifier > display_name > contact_name > contact_id > fallback
        """
        # For group chats
        if pd.notna(row.get('chat_display_name')) and row['chat_display_name']:
            return f"group_{row['chat_display_name']}"
        
        # For individual chats with proper chat identifier
        if pd.notna(row.get('chat_id')) and row['chat_id'] != 'unknown_' + str(row.name):
            return str(row['chat_id'])
        
        # For individual chats, use contact information
        if pd.notna(row.get('contact_name')) and row['contact_name']:
            return f"chat_{row['contact_name']}"
        
        if pd.notna(row.get('contact_id')) and row['contact_id']:
            return f"chat_{row['contact_id']}"
        
        # Fallback
        return f"unknown_chat_{row.name if hasattr(row, 'name') else 'unnamed'}"

    def get_chat_participants(self) -> Dict[str, List[str]]:
        """Get chat participants for better context"""
        if not self.verify_access():
            raise RuntimeError("Cannot access iMessage database. Check permissions.")

        query = """
            SELECT 
                c.chat_identifier,
                c.display_name as chat_name,
                GROUP_CONCAT(h.id, ', ') as participants,
                COUNT(DISTINCT h.id) as participant_count
            FROM chat c
            LEFT JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
            LEFT JOIN handle h ON chj.handle_id = h.ROWID
            GROUP BY c.ROWID, c.chat_identifier, c.display_name
            HAVING participant_count > 0
            ORDER BY participant_count DESC
        """

        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()

        chat_info = {}
        for _, row in df.iterrows():
            chat_key = row['chat_identifier'] or row['chat_name'] or 'unknown'
            chat_info[chat_key] = {
                'participants': row['participants'].split(', ') if row['participants'] else [],
                'participant_count': row['participant_count'],
                'chat_name': row['chat_name']
            }

        self.logger.info(f"Found {len(chat_info)} chats with participants")
        return chat_info

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
            "unique_chats": len(set(m.chat_id for m in your_msgs)),
            "most_active_chats": self._get_most_active_chats(your_msgs)
        }
        return analysis

    def _get_most_active_chats(self, messages: List[iMessage], top_n: int = 10) -> List[Dict]:
        """Get the most active chats by message count"""
        chat_counts = {}
        for msg in messages:
            chat_id = msg.chat_id
            if chat_id not in chat_counts:
                chat_counts[chat_id] = {'count': 0, 'latest_message': msg.timestamp}
            chat_counts[chat_id]['count'] += 1
            if msg.timestamp > chat_counts[chat_id]['latest_message']:
                chat_counts[chat_id]['latest_message'] = msg.timestamp

        # Sort by message count and return top N
        sorted_chats = sorted(chat_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:top_n]
        
        return [
            {
                'chat_id': chat_id,
                'message_count': data['count'],
                'latest_message': data['latest_message'].isoformat()
            }
            for chat_id, data in sorted_chats
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = iMessageCollector()
    try:
        print("🧪 Testing Enhanced iMessage Collection...")
        msgs = collector.collect_messages(limit=10)
        print(f"✅ Successfully collected {len(msgs)} messages")
        
        your_msgs = [m for m in msgs if m.is_from_me]
        print(f"📱 Found {len(your_msgs)} messages from you")
        
        # Show chat IDs
        unique_chats = set(m.chat_id for m in msgs)
        print(f"💬 Found {len(unique_chats)} unique chats:")
        for chat_id in list(unique_chats)[:5]:
            print(f"   - {chat_id}")
        
        # Show a few sample messages with better chat info
        print(f"\n📝 Sample messages:")
        for i, msg in enumerate(msgs[:3], 1):
            sender = "You" if msg.is_from_me else f"Contact({msg.contact_name or msg.contact_id})"
            preview = (msg.text[:30] + "...") if len(msg.text) > 30 else msg.text
            print(f"{i:2d}> [{sender}] in {msg.chat_id}: {preview}")

        # Test chat participants
        print(f"\n🔍 Testing chat participant detection...")
        chat_info = collector.get_chat_participants()
        print(f"Found {len(chat_info)} chats with participant info")
        
        # Show analysis
        analysis = collector.analyze_your_messages(msgs)
        if 'unique_chats' in analysis:
            print(f"\n📊 Your messaging analysis:")
            print(f"   Unique chats: {analysis['unique_chats']}")
            print(f"   Most active chats:")
            for chat in analysis['most_active_chats'][:3]:
                print(f"     - {chat['chat_id']}: {chat['message_count']} messages")
            
    except Exception as e:
        print("❌ Error fetching messages:", e)
        print("\nTroubleshooting:")
        print("1. Grant 'Full Disk Access' to your terminal/IDE")
        print("2. Make sure iMessage is set up on this Mac")
        import traceback
        traceback.print_exc()