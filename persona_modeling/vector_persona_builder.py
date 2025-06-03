#!/usr/bin/env python3
"""
Vector-Based Persona Builder with FAISS
Embeds your iMessages + preferences into searchable vector database for hyper-personalization
"""

import sys
import os
import json
import logging
import yaml
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple # Optional is already here
from datetime import datetime
from dataclasses import dataclass

# Add project root to Python path
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from data_ingest.imessage_reader import iMessageCollector, iMessage
from config.settings import settings

@dataclass
class PersonaVector:
    """Data class for storing vectors with metadata.

    The 'metadata' field can contain various pieces of information
    about the vector's source, including person-specific details like
    'source_chat_id', 'message_author_type', 'message_author_id',
    and 'message_author_name' for vectors derived from messages.
    """
    vector_id: str
    text_content: str
    vector_type: str  # 'message', 'preference', 'context', 'relationship'
    metadata: Dict
    embedding: Optional[np.ndarray] = None

class VectorPersonaBuilder:
    """
    Vector-based persona builder using FAISS for semantic similarity
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.default_preferences_path = settings.PREFERENCES_YAML_PATH
        self.vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # FAISS indices for different vector types
        self.message_index = None
        self.preference_index = None
        self.context_index = None
        self.relationship_index = None
        
        # Metadata storage
        self.vector_metadata = {}
        
        self.logger.info(f"Initialized with {embedding_model}, dimension: {self.vector_dimension}")
    
    def build_persona_vectors(self, messages: List[iMessage], preferences_path: Optional[str] = None) -> Dict:
        """
        Main function: Build vector-based persona from iMessages + preferences
        """
        self.logger.info("🧬 Building Vector-Based Persona...")
        
        # Determine preferences path
        actual_preferences_path = preferences_path if preferences_path is not None else self.default_preferences_path

        # Load preferences
        preferences = self._load_preferences(actual_preferences_path)
        
        # Filter to your messages
        your_messages = [m for m in messages if m.is_from_me and m.text]
        
        if not your_messages:
            return {"error": "No messages from you found"}
        
        # Build different types of vectors
        persona_vectors = {
            "message_vectors": self._build_message_vectors(your_messages),
            "preference_vectors": self._build_preference_vectors(preferences),
            "context_vectors": self._build_context_vectors(your_messages),
            "relationship_vectors": self._build_relationship_vectors(your_messages),
            "vector_stats": {
                "total_vectors": 0,
                "embedding_dimension": self.vector_dimension,
                "model_used": str(self.embedding_model),
                "created_at": datetime.now().isoformat()
            }
        }
        
        # Create FAISS indices
        self._create_faiss_indices(persona_vectors)
        
        # Save everything
        self._save_vector_store()
        
        # Calculate total vectors
        total_vectors = sum(len(vectors) for key, vectors in persona_vectors.items() if key != "vector_stats")
        persona_vectors["vector_stats"]["total_vectors"] = total_vectors
        
        return persona_vectors
    
    def _load_preferences(self, preferences_path: str) -> Dict:
        """Load preferences from YAML file"""
        path = Path(preferences_path)
        if not path.exists():
            self.logger.warning(f"Preferences file not found: {preferences_path}")
            return {"likes": [], "dislikes": []}
        
        with open(path, 'r') as f:
            preferences = yaml.safe_load(f)
        
        self.logger.info(f"Loaded preferences: {len(preferences.get('likes', []))} likes, {len(preferences.get('dislikes', []))} dislikes")
        return preferences
    
    def _build_message_vectors(self, your_messages: List[iMessage]) -> List[PersonaVector]:
        """Build vectors from your message patterns"""
        self.logger.info("📱 Building message style vectors...")
        
        message_vectors = []
        
        # Group messages by length for style analysis
        short_messages = [m for m in your_messages if len(m.text.split()) <= 3]
        medium_messages = [m for m in your_messages if 4 <= len(m.text.split()) <= 10]
        long_messages = [m for m in your_messages if len(m.text.split()) > 10]
        
        # Sample different message types
        message_samples = {
            "short_responses": short_messages[:100],
            "medium_responses": medium_messages[:100], 
            "long_responses": long_messages[:50],
            "question_responses": [m for m in your_messages if '?' in m.text][:50],
            "excited_responses": [m for m in your_messages if '!' in m.text or any(emoji in m.text for emoji in ['😊', '😂', '🎉', '💪'])][:50]
        }
        
        vector_id = 0
        for msg_type, messages in message_samples.items():
            for message in messages:
                if message.text and len(message.text.strip()) > 0:
                    vector = PersonaVector(
                        vector_id=f"msg_{vector_id}",
                        text_content=message.text,
                        vector_type="message",
                        metadata={
                            "message_type": msg_type,
                            "length": len(message.text),
                            "word_count": len(message.text.split()),
                            "timestamp": message.timestamp.isoformat(),
                            "service": message.service,
                            "contact": message.contact_name or "unknown", # This might be redundant if message.is_from_me is true
                            "source_chat_id": message.chat_id  # ID of the chat thread this message belongs to
                        }
                    )
                    # Add author information
                    if message.is_from_me:
                        vector.metadata["message_author_type"] = "self"      # Indicates the message is from the user running the script
                        vector.metadata["message_author_id"] = None         # No external ID for self
                        vector.metadata["message_author_name"] = None       # No external name for self
                    else:
                        vector.metadata["message_author_type"] = "external"  # Indicates the message is from another contact
                        vector.metadata["message_author_id"] = message.contact_id # ID of the contact
                        vector.metadata["message_author_name"] = message.contact_name or message.contact_id # Name of the contact
                    message_vectors.append(vector)
                    vector_id += 1
        
        self.logger.info(f"Created {len(message_vectors)} message vectors")
        return message_vectors
    
    def _build_preference_vectors(self, preferences: Dict) -> List[PersonaVector]:
        """Build vectors from preferences.yaml"""
        self.logger.info("❤️ Building preference vectors...")
        
        preference_vectors = []
        vector_id = 0
        
        # Process likes
        for like in preferences.get('likes', []):
            vector = PersonaVector(
                vector_id=f"pref_like_{vector_id}",
                text_content=f"I really enjoy {like}",
                vector_type="preference",
                metadata={
                    "preference_type": "like",
                    "original_text": like,
                    "sentiment": "positive"
                }
            )
            preference_vectors.append(vector)
            vector_id += 1
        
        # Process dislikes
        for dislike in preferences.get('dislikes', []):
            vector = PersonaVector(
                vector_id=f"pref_dislike_{vector_id}",
                text_content=f"I really don't like {dislike}",
                vector_type="preference",
                metadata={
                    "preference_type": "dislike", 
                    "original_text": dislike,
                    "sentiment": "negative"
                }
            )
            preference_vectors.append(vector)
            vector_id += 1
        
        # Create expanded preference contexts
        expanded_preferences = self._expand_preferences(preferences)
        for expanded in expanded_preferences:
            vector = PersonaVector(
                vector_id=f"pref_expanded_{vector_id}",
                text_content=expanded["text"],
                vector_type="preference",
                metadata={
                    "preference_type": "expanded",
                    "category": expanded["category"],
                    "sentiment": expanded["sentiment"]
                }
            )
            preference_vectors.append(vector)
            vector_id += 1
        
        self.logger.info(f"Created {len(preference_vectors)} preference vectors")
        return preference_vectors
    
    def _expand_preferences(self, preferences: Dict) -> List[Dict]:
        """Expand preferences into conversation contexts"""
        expanded = []
        
        # Map preferences to conversation contexts
        preference_expansions = {
            "rock climbing": [
                {"text": "I love outdoor adventures and physical challenges", "category": "activities", "sentiment": "positive"},
                {"text": "I enjoy being active and staying fit", "category": "lifestyle", "sentiment": "positive"},
                {"text": "I like activities that require focus and problem-solving", "category": "personality", "sentiment": "positive"}
            ],
            "jazz piano": [
                {"text": "I appreciate complex and sophisticated music", "category": "music", "sentiment": "positive"},
                {"text": "I enjoy creative and artistic pursuits", "category": "arts", "sentiment": "positive"},
                {"text": "I like music that requires skill and improvisation", "category": "music", "sentiment": "positive"}
            ],
            "high-protein meals": [
                {"text": "I care about nutrition and healthy eating", "category": "food", "sentiment": "positive"},
                {"text": "I prefer meals that support an active lifestyle", "category": "health", "sentiment": "positive"},
                {"text": "I like food that helps with fitness goals", "category": "fitness", "sentiment": "positive"}
            ],
            "coding": [
                {"text": "I enjoy problem-solving and logical thinking", "category": "work", "sentiment": "positive"},
                {"text": "I like technology and building things", "category": "technology", "sentiment": "positive"},
                {"text": "I appreciate clean, efficient solutions", "category": "personality", "sentiment": "positive"}
            ],
            "travel like a backpacker": [
                {"text": "I enjoy authentic and adventurous travel experiences", "category": "travel", "sentiment": "positive"},
                {"text": "I prefer budget-friendly and immersive travel", "category": "travel", "sentiment": "positive"},
                {"text": "I like exploring places off the beaten path", "category": "lifestyle", "sentiment": "positive"}
            ],
            "pineapple on pizza": [
                {"text": "I have strong opinions about food combinations", "category": "food", "sentiment": "negative"},
                {"text": "I prefer traditional food preparations", "category": "food", "sentiment": "negative"}
            ],
            "slow walkers": [
                {"text": "I prefer efficient movement and good pace", "category": "personality", "sentiment": "negative"},
                {"text": "I value time and don't like unnecessary delays", "category": "personality", "sentiment": "negative"}
            ]
        }
        
        for like in preferences.get('likes', []):
            if like in preference_expansions:
                expanded.extend(preference_expansions[like])
        
        for dislike in preferences.get('dislikes', []):
            if dislike in preference_expansions:
                expanded.extend(preference_expansions[dislike])
        
        return expanded
    
    def _build_context_vectors(self, your_messages: List[iMessage]) -> List[PersonaVector]:
        """Build vectors for different conversation contexts"""
        self.logger.info("🎯 Building context vectors...")
        
        context_vectors = []
        vector_id = 0
        
        # Identify different conversation contexts
        contexts = self._identify_conversation_contexts(your_messages)
        
        for context_type, messages in contexts.items():
            # Sample representative messages for each context
            sample_messages = messages[:20]  # Limit for MVP
            
            for message in sample_messages:
                vector = PersonaVector(
                    vector_id=f"ctx_{vector_id}",
                    text_content=message.text,
                    vector_type="context",
                    metadata={
                        "context_type": context_type,
                        "message_length": len(message.text),
                        "timestamp": message.timestamp.isoformat(),
                        "contact": message.contact_name or "unknown", # This might be redundant if message.is_from_me is true
                        "source_chat_id": message.chat_id  # ID of the chat thread this message belongs to
                    }
                )
                # Add author information
                if message.is_from_me:
                    vector.metadata["message_author_type"] = "self"      # Indicates the message is from the user running the script
                    vector.metadata["message_author_id"] = None         # No external ID for self
                    vector.metadata["message_author_name"] = None       # No external name for self
                else:
                    vector.metadata["message_author_type"] = "external"  # Indicates the message is from another contact
                    vector.metadata["message_author_id"] = message.contact_id # ID of the contact
                    vector.metadata["message_author_name"] = message.contact_name or message.contact_id # Name of the contact
                context_vectors.append(vector)
                vector_id += 1
        
        self.logger.info(f"Created {len(context_vectors)} context vectors")
        return context_vectors
    
    def _identify_conversation_contexts(self, messages: List[iMessage]) -> Dict[str, List[iMessage]]:
        """Identify different types of conversation contexts"""
        contexts = {
            "casual_greeting": [],
            "making_plans": [],
            "asking_questions": [],
            "sharing_excitement": [],
            "problem_solving": [],
            "expressing_opinions": []
        }
        
        for message in messages:
            text_lower = message.text.lower()
            
            # Simple keyword-based context identification (MVP approach)
            if any(word in text_lower for word in ['hey', 'hi', 'hello', 'what\'s up', 'how are']):
                contexts["casual_greeting"].append(message)
            elif any(word in text_lower for word in ['let\'s', 'want to', 'should we', 'when', 'where']):
                contexts["making_plans"].append(message)
            elif '?' in message.text:
                contexts["asking_questions"].append(message)
            elif any(word in text_lower for word in ['!', 'awesome', 'amazing', 'excited', 'love']):
                contexts["sharing_excitement"].append(message)
            elif any(word in text_lower for word in ['think', 'believe', 'opinion', 'feel like']):
                contexts["expressing_opinions"].append(message)
            else:
                contexts["problem_solving"].append(message)
        
        return contexts
    
    def _build_relationship_vectors(self, your_messages: List[iMessage]) -> List[PersonaVector]:
        """Build vectors for different relationship communication patterns"""
        self.logger.info("👥 Building relationship vectors...")
        
        relationship_vectors = []
        vector_id = 0
        
        # Group messages by contact
        contact_messages = {}
        for message in your_messages:
            contact = message.contact_name or message.contact_id or "unknown"
            if contact not in contact_messages:
                contact_messages[contact] = []
            contact_messages[contact].append(message)
        
        # Get top contacts and sample their messages
        top_contacts = sorted(contact_messages.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        
        for contact, messages in top_contacts:
            # Sample messages for this relationship
            sample_size = min(10, len(messages))
            sample_messages = messages[:sample_size]
            
            for message in sample_messages:
                vector = PersonaVector(
                    vector_id=f"rel_{vector_id}",
                    text_content=message.text,
                    vector_type="relationship",
                    metadata={
                        "contact": contact, # This is the contact the message is with, not necessarily the author
                        "total_messages_with_contact": len(messages),
                        "message_length": len(message.text),
                        "timestamp": message.timestamp.isoformat(),
                        "source_chat_id": message.chat_id  # ID of the chat thread this message belongs to
                    }
                )
                # Add author information
                if message.is_from_me:
                    vector.metadata["message_author_type"] = "self"      # Indicates the message is from the user running the script
                    vector.metadata["message_author_id"] = None         # No external ID for self
                    vector.metadata["message_author_name"] = None       # No external name for self
                else:
                    vector.metadata["message_author_type"] = "external"  # Indicates the message is from another contact
                    vector.metadata["message_author_id"] = message.contact_id # ID of the contact (should align with 'contact' variable)
                    vector.metadata["message_author_name"] = message.contact_name or message.contact_id # Name of the contact
                relationship_vectors.append(vector)
                vector_id += 1
        
        self.logger.info(f"Created {len(relationship_vectors)} relationship vectors")
        return relationship_vectors
    
    def _create_faiss_indices(self, persona_vectors: Dict) -> None:
        """Create FAISS indices for fast similarity search"""
        self.logger.info("🔍 Creating FAISS indices...")
        
        # Create embeddings for all vectors
        all_vectors = []
        all_texts = []
        
        for vector_type, vectors in persona_vectors.items():
            if vector_type == "vector_stats":
                continue
                
            for vector in vectors:
                all_texts.append(vector.text_content)
                all_vectors.append(vector)
        
        if not all_texts:
            self.logger.warning("No texts to embed!")
            return
        
        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(all_texts)} texts...")
        embeddings = self.embedding_model.encode(all_texts, show_progress_bar=True)
        
        # Store embeddings in vectors
        for i, vector in enumerate(all_vectors):
            vector.embedding = embeddings[i]
            self.vector_metadata[vector.vector_id] = {
                "text": vector.text_content,
                "type": vector.vector_type,
                "metadata": vector.metadata
            }
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.main_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.main_index.add(embeddings.astype(np.float32))
        
        self.logger.info(f"Created FAISS index with {self.main_index.ntotal} vectors")
    
    def _save_vector_store(self) -> None:
        """Save FAISS indices and metadata"""
        output_dir = Path("data/vector_store")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.main_index, str(output_dir / "persona.index"))
        
        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(self.vector_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Vector store saved to {output_dir}")
    
    def search_similar(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors given a query"""
        if self.main_index is None:
            raise ValueError("No index loaded. Build persona first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.main_index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.vector_metadata):
                vector_id = list(self.vector_metadata.keys())[idx]
                metadata = self.vector_metadata[vector_id]
                results.append((metadata["text"], float(score), metadata))
        
        return results

def main():
    """Main function to build vector-based persona"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🧬 Building Vector-Based Persona with FAISS")
    print("=" * 60)
    
    # Collect messages
    collector = iMessageCollector()
    
    try:
        print("📱 Collecting iMessage history...")
        messages = collector.collect_messages(limit=30000)  # Reasonable limit for MVP
        print(f"✅ Collected {len(messages)} messages")
        
        # Build vector persona
        print("🔍 Building vector embeddings...")
        persona_builder = VectorPersonaBuilder()
        persona_vectors = persona_builder.build_persona_vectors(messages)
        
        if "error" in persona_vectors:
            print(f"❌ Error: {persona_vectors['error']}")
            return
        
        # Display results
        print("\n📊 Vector Persona Summary:")
        print("=" * 50)
        
        stats = persona_vectors["vector_stats"]
        print(f"🔢 Total vectors created: {stats['total_vectors']}")
        print(f"📐 Embedding dimension: {stats['embedding_dimension']}")
        print(f"🤖 Model used: {stats['model_used']}")
        
        for vector_type, vectors in persona_vectors.items():
            if vector_type != "vector_stats":
                print(f"   {vector_type}: {len(vectors)} vectors")
        
        # Test similarity search
        print("\n🔍 Testing Vector Search:")
        test_queries = [
            "want to grab dinner?",
            "how was your day?",
            "let's go climbing",
            "what do you think about jazz?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = persona_builder.search_similar(query, k=3)
            for i, (text, score, metadata) in enumerate(results, 1):
                print(f"  {i}. [{score:.3f}] {text[:50]}...")
        
        print(f"\n💾 Vector store saved to data/vector_store/")
        print("🎉 Vector-based persona building complete!")
        print("\nNext: Use this vector store for hyper-personalized response generation")
        
    except Exception as e:
        print(f"❌ Error building vector persona: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()