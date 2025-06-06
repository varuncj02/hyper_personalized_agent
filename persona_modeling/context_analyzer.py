#!/usr/bin/env python3
"""
Context Analyzer - Enhanced context understanding for Phase 1
Analyzes WHO you're talking to and WHY for better response generation
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

# Add project root to Python path
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_ingest.imessage_reader import iMessage

@dataclass
class ConversationContext:
    """Rich context for a conversation"""
    chat_id: str
    primary_contact: str
    relationship_type: str  # family, friend, work, romantic, etc.
    conversation_topic: Optional[str]
    emotional_tone: str  # excited, casual, serious, etc.
    time_context: str  # morning, work_hours, late_night, etc.
    conversation_urgency: str  # urgent, normal, casual
    message_frequency: str  # high, medium, low
    recent_message_count: int
    last_interaction: datetime
    formality_level: str  # formal, casual, very_casual

@dataclass
class MessageAnalysis:
    """Analysis of an incoming message"""
    message_type: str  # question, greeting, statement, request, etc.
    emotional_tone: str
    urgency_level: str
    requires_response: bool
    key_topics: List[str]
    sentiment_score: float  # -1 to 1
    formality_level: str

class ContextAnalyzer:
    """Analyzes conversation context and message intent"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Relationship classification patterns
        self.relationship_patterns = {
            'family': {
                'keywords': ['mom', 'dad', 'family', 'home', 'parents', 'sister', 'brother'],
                'formality': 'casual',
                'typical_topics': ['family', 'home', 'health', 'plans']
            },
            'close_friend': {
                'keywords': ['bro', 'dude', 'bestie', 'lol', 'haha'],
                'formality': 'very_casual',
                'typical_topics': ['hangout', 'plans', 'gossip', 'fun']
            },
            'work': {
                'keywords': ['meeting', 'project', 'deadline', 'office', 'work', 'team'],
                'formality': 'formal',
                'typical_topics': ['work', 'business', 'schedule']
            },
            'casual_friend': {
                'keywords': ['hey', 'what\'s up', 'cool', 'nice'],
                'formality': 'casual',
                'typical_topics': ['general', 'plans', 'social']
            }
        }
        
        # Message type patterns
        self.message_patterns = {
            'question': r'\?|what|how|when|where|why|who|can you|would you|do you',
            'greeting': r'^(hey|hi|hello|good morning|good evening|what\'s up)',
            'request': r'can you|could you|please|would you mind',
            'plan_making': r'want to|let\'s|should we|how about|tonight|tomorrow|weekend',
            'gratitude': r'thank|thanks|appreciate',
            'apology': r'sorry|apologize|my bad|my fault'
        }
        
    def analyze_conversation_context(self, 
                                   chat_id: str, 
                                   recent_messages: List[iMessage],
                                   all_messages: List[iMessage]) -> ConversationContext:
        """Analyze the full context of a conversation"""
        
        if not recent_messages:
            return self._create_default_context(chat_id)
        
        # Get primary contact
        primary_contact = self._identify_primary_contact(recent_messages)
        
        # Classify relationship
        relationship_type = self._classify_relationship(all_messages, primary_contact)
        
        # Analyze conversation topic
        conversation_topic = self._extract_conversation_topic(recent_messages)
        
        # Determine emotional tone
        emotional_tone = self._analyze_emotional_tone(recent_messages)
        
        # Get time context
        time_context = self._get_time_context()
        
        # Assess urgency
        conversation_urgency = self._assess_urgency(recent_messages)
        
        # Calculate message frequency
        message_frequency = self._calculate_message_frequency(all_messages)
        
        # Determine formality
        formality_level = self._analyze_formality(recent_messages)
        
        return ConversationContext(
            chat_id=chat_id,
            primary_contact=primary_contact,
            relationship_type=relationship_type,
            conversation_topic=conversation_topic,
            emotional_tone=emotional_tone,
            time_context=time_context,
            conversation_urgency=conversation_urgency,
            message_frequency=message_frequency,
            recent_message_count=len(recent_messages),
            last_interaction=recent_messages[0].timestamp if recent_messages else datetime.now(),
            formality_level=formality_level
        )
    
    def analyze_incoming_message(self, message: str, context: ConversationContext) -> MessageAnalysis:
        """Analyze an incoming message for intent and characteristics"""
        
        message_lower = message.lower()
        
        # Classify message type
        message_type = self._classify_message_type(message)
        
        # Analyze emotional tone
        emotional_tone = self._detect_message_emotional_tone(message)
        
        # Assess urgency
        urgency_level = self._detect_urgency(message)
        
        # Check if response required
        requires_response = self._requires_response(message, message_type)
        
        # Extract key topics
        key_topics = self._extract_message_topics(message)
        
        # Calculate sentiment
        sentiment_score = self._calculate_sentiment(message)
        
        # Determine formality
        formality_level = self._detect_message_formality(message)
        
        return MessageAnalysis(
            message_type=message_type,
            emotional_tone=emotional_tone,
            urgency_level=urgency_level,
            requires_response=requires_response,
            key_topics=key_topics,
            sentiment_score=sentiment_score,
            formality_level=formality_level
        )
    
    def _identify_primary_contact(self, messages: List[iMessage]) -> str:
        """Identify the primary contact in the conversation"""
        contacts = [m.contact_name or m.contact_id for m in messages if not m.is_from_me]
        if contacts:
            # Return most frequent contact
            return Counter(contacts).most_common(1)[0][0]
        return "unknown"
    
    def _classify_relationship(self, messages: List[iMessage], contact: str) -> str:
        """Classify relationship type based on message history"""
        
        # Get messages with this contact
        contact_messages = [m for m in messages if 
                          (m.contact_name == contact or m.contact_id == contact) and m.is_from_me]
        
        if not contact_messages:
            return "unknown"
        
        # Analyze message content for relationship indicators
        all_text = " ".join(m.text.lower() for m in contact_messages[-50:])  # Last 50 messages
        
        # Score each relationship type
        scores = {}
        for rel_type, patterns in self.relationship_patterns.items():
            score = sum(1 for keyword in patterns['keywords'] if keyword in all_text)
            scores[rel_type] = score
        
        # Return highest scoring relationship
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        # Fallback classification based on message characteristics
        avg_length = sum(len(m.text.split()) for m in contact_messages) / len(contact_messages)
        
        if avg_length < 3:
            return "casual_friend"
        elif avg_length > 15:
            return "work"
        else:
            return "friend"
    
    def _extract_conversation_topic(self, messages: List[iMessage]) -> Optional[str]:
        """Extract the main topic of recent conversation"""
        
        if not messages:
            return None
        
        # Combine recent message text
        recent_text = " ".join(m.text.lower() for m in messages[:5])
        
        # Topic keywords
        topics = {
            'plans': ['meet', 'hang out', 'tonight', 'tomorrow', 'weekend', 'when', 'where'],
            'work': ['work', 'job', 'office', 'meeting', 'project', 'deadline'],
            'food': ['dinner', 'lunch', 'eat', 'restaurant', 'food', 'hungry'],
            'travel': ['trip', 'vacation', 'travel', 'flight', 'hotel'],
            'family': ['family', 'home', 'parents', 'mom', 'dad'],
            'health': ['sick', 'doctor', 'feel', 'better', 'health'],
            'social': ['party', 'event', 'friends', 'celebration']
        }
        
        # Score topics
        topic_scores = {}
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in recent_text)
            if score > 0:
                topic_scores[topic] = score
        
        return max(topic_scores, key=topic_scores.get) if topic_scores else None
    
    def _analyze_emotional_tone(self, messages: List[iMessage]) -> str:
        """Analyze overall emotional tone of recent conversation"""
        
        if not messages:
            return "neutral"
        
        recent_text = " ".join(m.text.lower() for m in messages[:3])
        
        # Emotional indicators
        excited_indicators = ['!', '😊', '😂', '🎉', 'awesome', 'amazing', 'excited', 'love']
        sad_indicators = ['😢', '😔', 'sad', 'upset', 'disappointed', 'down']
        angry_indicators = ['😠', '😡', 'angry', 'frustrated', 'annoyed', 'mad']
        casual_indicators = ['lol', 'haha', 'cool', 'nice', 'chill']
        serious_indicators = ['important', 'serious', 'urgent', 'need to talk']
        
        # Count indicators
        excited_count = sum(1 for indicator in excited_indicators if indicator in recent_text)
        sad_count = sum(1 for indicator in sad_indicators if indicator in recent_text)
        angry_count = sum(1 for indicator in angry_indicators if indicator in recent_text)
        casual_count = sum(1 for indicator in casual_indicators if indicator in recent_text)
        serious_count = sum(1 for indicator in serious_indicators if indicator in recent_text)
        
        # Determine dominant tone
        scores = {
            'excited': excited_count,
            'sad': sad_count,
            'angry': angry_count,
            'casual': casual_count,
            'serious': serious_count
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        
        return "neutral"
    
    def _get_time_context(self) -> str:
        """Get current time context"""
        hour = datetime.now().hour
        
        if 6 <= hour < 9:
            return "early_morning"
        elif 9 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 20:
            return "evening"
        elif 20 <= hour < 23:
            return "night"
        else:
            return "late_night"
    
    def _assess_urgency(self, messages: List[iMessage]) -> str:
        """Assess conversation urgency"""
        
        if not messages:
            return "normal"
        
        recent_text = " ".join(m.text.lower() for m in messages[:2])
        
        urgent_indicators = ['urgent', 'emergency', 'asap', 'now', 'immediately', '!!!']
        high_indicators = ['important', 'soon', 'quickly', '!!']
        
        if any(indicator in recent_text for indicator in urgent_indicators):
            return "urgent"
        elif any(indicator in recent_text for indicator in high_indicators):
            return "high"
        else:
            return "normal"
    
    def _calculate_message_frequency(self, messages: List[iMessage]) -> str:
        """Calculate messaging frequency"""
        
        if len(messages) < 2:
            return "unknown"
        
        # Get messages from last 7 days
        week_ago = datetime.now() - timedelta(days=7)
        recent_messages = [m for m in messages if m.timestamp > week_ago]
        
        if not recent_messages:
            return "low"
        
        messages_per_day = len(recent_messages) / 7
        
        if messages_per_day > 10:
            return "very_high"
        elif messages_per_day > 5:
            return "high"
        elif messages_per_day > 2:
            return "medium"
        elif messages_per_day > 0.5:
            return "low"
        else:
            return "very_low"
    
    def _analyze_formality(self, messages: List[iMessage]) -> str:
        """Analyze formality level of conversation"""
        
        if not messages:
            return "casual"
        
        # Look at your messages only
        your_messages = [m for m in messages if m.is_from_me]
        if not your_messages:
            return "casual"
        
        text = " ".join(m.text.lower() for m in your_messages[:5])
        
        formal_indicators = ['please', 'thank you', 'would you', 'could you', 'sir', 'ma\'am']
        casual_indicators = ['hey', 'yo', 'sup', 'gonna', 'wanna', 'ur', 'u']
        very_casual_indicators = ['lol', 'omg', 'wtf', 'haha', 'lmao']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text)
        casual_count = sum(1 for indicator in casual_indicators if indicator in text)
        very_casual_count = sum(1 for indicator in very_casual_indicators if indicator in text)
        
        if formal_count > casual_count + very_casual_count:
            return "formal"
        elif very_casual_count > 0:
            return "very_casual"
        else:
            return "casual"
    
    def _classify_message_type(self, message: str) -> str:
        """Classify the type of incoming message"""
        
        message_lower = message.lower()
        
        # Check patterns
        for msg_type, pattern in self.message_patterns.items():
            if re.search(pattern, message_lower):
                return msg_type
        
        # Default classification
        if '?' in message:
            return "question"
        elif any(word in message_lower for word in ['!', 'awesome', 'amazing']):
            return "excitement"
        else:
            return "statement"
    
    def _detect_message_emotional_tone(self, message: str) -> str:
        """Detect emotional tone of specific message"""
        
        message_lower = message.lower()
        
        # Emotional patterns
        if any(indicator in message_lower for indicator in ['!', '😊', '😂', 'awesome', 'amazing']):
            return "excited"
        elif any(indicator in message_lower for indicator in ['😢', 'sad', 'upset']):
            return "sad"
        elif any(indicator in message_lower for indicator in ['😠', 'angry', 'frustrated']):
            return "angry"
        elif any(indicator in message_lower for indicator in ['lol', 'haha', 'funny']):
            return "playful"
        else:
            return "neutral"
    
    def _detect_urgency(self, message: str) -> str:
        """Detect urgency level of message"""
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['urgent', 'emergency', 'asap', 'now', '!!!']):
            return "urgent"
        elif any(word in message_lower for word in ['important', 'soon', '!!']):
            return "high"
        else:
            return "normal"
    
    def _requires_response(self, message: str, message_type: str) -> bool:
        """Determine if message requires a response"""
        
        # Questions always need responses
        if message_type in ['question', 'request']:
            return True
        
        # Greetings usually need responses
        if message_type == 'greeting':
            return True
        
        # Plan making needs response
        if message_type == 'plan_making':
            return True
        
        # Direct mentions need response
        if any(word in message.lower() for word in ['what do you think', 'let me know']):
            return True
        
        return False
    
    def _extract_message_topics(self, message: str) -> List[str]:
        """Extract key topics from message"""
        
        message_lower = message.lower()
        topics = []
        
        # Topic detection
        topic_keywords = {
            'food': ['eat', 'dinner', 'lunch', 'restaurant', 'food'],
            'plans': ['meet', 'hang out', 'tonight', 'tomorrow'],
            'work': ['work', 'meeting', 'project', 'office'],
            'travel': ['trip', 'vacation', 'flight', 'hotel'],
            'time': ['when', 'time', 'schedule', 'calendar']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _calculate_sentiment(self, message: str) -> float:
        """Calculate sentiment score (-1 to 1)"""
        
        message_lower = message.lower()
        
        positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'happy', 'excited', 'nice']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'frustrated', 'annoyed']
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        words = message.split()
        if not words:
            return 0.0
        
        sentiment = (positive_count - negative_count) / len(words)
        return max(-1.0, min(1.0, sentiment))
    
    def _detect_message_formality(self, message: str) -> str:
        """Detect formality level of message"""
        
        message_lower = message.lower()
        
        formal_indicators = ['please', 'thank you', 'would you', 'could you']
        casual_indicators = ['hey', 'yo', 'sup', 'gonna', 'wanna']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in message_lower)
        casual_count = sum(1 for indicator in casual_indicators if indicator in message_lower)
        
        if formal_count > casual_count:
            return "formal"
        elif casual_count > 0:
            return "casual"
        else:
            return "neutral"
    
    def _create_default_context(self, chat_id: str) -> ConversationContext:
        """Create default context for new conversations"""
        
        return ConversationContext(
            chat_id=chat_id,
            primary_contact="unknown",
            relationship_type="unknown",
            conversation_topic=None,
            emotional_tone="neutral",
            time_context=self._get_time_context(),
            conversation_urgency="normal",
            message_frequency="unknown",
            recent_message_count=0,
            last_interaction=datetime.now(),
            formality_level="casual"
        )