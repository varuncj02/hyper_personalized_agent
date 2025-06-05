# Vector-Based Persona Builder with FAISS

## Description

This project provides tools to build a vector-based persona from your iMessage history and user-defined preferences. It embeds textual data into a searchable vector database using FAISS (Facebook AI Similarity Search) and sentence transformers. This allows for hyper-personalization by enabling semantic search over your communication patterns, preferences, and interaction styles.

The core idea is to represent different aspects of your persona (message styles, stated preferences, common conversation contexts, and relationship dynamics) as numerical vectors. These vectors can then be used to find similar items, understand behavioral patterns, and potentially power personalized response generation systems.

## Features

*   **iMessage Ingestion**: Reads and processes iMessage history from `chat.db`.
*   **Preference Integration**: Incorporates user-defined likes and dislikes from a YAML file.
*   **Multi-Faceted Persona Modeling**:
    *   `message_vectors`: Captures a variety of your messaging styles (short, medium, long responses, questions, exclamations).
    *   `preference_vectors`: Represents your explicitly stated likes and dislikes, along with expanded contexts.
    *   `context_vectors`: Models your communication patterns in different conversational contexts (greetings, planning, problem-solving).
    *   `relationship_vectors`: Analyzes communication styles with your top contacts.
*   **Sentence Embeddings**: Uses `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to generate high-quality semantic embeddings.
*   **FAISS-Powered Search**: Builds a FAISS index for efficient similarity search over all persona vectors.
*   **Detailed Metadata**: Each `PersonaVector` stores comprehensive metadata, including:
    *   `vector_id`: Unique identifier for the vector.
    *   `text_content`: The original text that was embedded.
    *   `vector_type`: The category of the vector (e.g., 'message', 'preference').
    *   Specific metadata related to the vector type (e.g., message length, preference sentiment).
    *   **Person-Specific Details (for message-derived vectors)**:
        *   `source_chat_id`: The ID of the chat thread the message belongs to.
        *   `message_author_type`: "self" (for messages you sent) or "external" (for messages received).
        *   `message_author_id`: The contact ID of the sender if `message_author_type` is "external".
        *   `message_author_name`: The contact name (or ID) of the sender if `message_author_type` is "external".
*   **Configurable Settings**: Leverages a `.env` file for API keys and paths, and `config/settings.py` for other configurations.
*   **Persistent Vector Store**: Saves the FAISS index and metadata to disk for later use.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The project likely has a `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file lists all necessary packages. Key dependencies include `pydantic-settings`, `sentence-transformers`, `faiss-cpu`, `PyYAML`, and `numpy`.

## Configuration

1.  **Environment Variables (`.env` file):**
    Create a `.env` file in the project root directory. This file is used to store sensitive information like API keys and potentially user-specific paths. Example:
    ```dotenv
    # Database paths (if overriding defaults)
    # IMESSAGE_DB_PATH="~/Library/Messages/chat.db"
    # PERSONA_DB_PATH="./data/persona_vectors.faiss" # This seems to be an output path, not input

    # API Keys (if needed for future features like Reddit integration)
    # REDDIT_CLIENT_ID="your_reddit_client_id"
    # REDDIT_CLIENT_SECRET="your_reddit_client_secret"
    # REDDIT_USER_AGENT="PersonaAgent/1.0"
    ```
    Refer to `config/settings.py` for all possible environment variables.

2.  **Application Settings (`config/settings.py`):**
    This file defines various settings for the application. Key settings include:
    *   `IMESSAGE_DB_PATH`: Path to your iMessage `chat.db` file. Defaults to `~/Library/Messages/chat.db`.
    *   `PERSONA_DB_PATH`: Path where the FAISS index for persona vectors will be saved (this is an output path). Defaults to `./data/persona_vectors.faiss`.
    *   `PREFERENCES_YAML_PATH`: Path to the YAML file containing user preferences. Defaults to `config/preferences.yaml`.
    *   `EMBEDDING_MODEL`: The SentenceTransformer model to use.
    *   `VECTOR_DIMENSION`: The dimension of the embeddings (determined by the model).

3.  **User Preferences (`config/preferences.yaml`):**
    Create a `preferences.yaml` file in the `config/` directory (or the path specified by `PREFERENCES_YAML_PATH`). This file defines your likes and dislikes. Example:
    ```yaml
    likes:
      - "rock climbing"
      - "jazz piano"
      - "coding complex algorithms"
      - "spicy food"
      - "traveling to new countries"

    dislikes:
      - "pineapple on pizza"
      - "slow walkers"
      - "loud noises in the morning"
    ```

## Usage

The main script for building the persona is `persona_modeling/vector_persona_builder.py`.

1.  **Ensure Configuration:** Verify that your iMessage database path is correctly set in `config/settings.py` or your `.env` file, and your `preferences.yaml` is populated.

2.  **Run the script:**
    ```bash
    python persona_modeling/vector_persona_builder.py
    ```

    The script will perform the following steps:
    *   Collect iMessages.
    *   Load preferences from the YAML file.
    *   Build various types of persona vectors (messages, preferences, contexts, relationships).
    *   Generate embeddings for all text content.
    *   Create a FAISS index.
    *   Save the FAISS index and associated metadata to the `data/vector_store/` directory (by default).

3.  **Output:**
    *   **Console Logs:** The script will print progress information, statistics about the vectors created, and results from test similarity searches.
    *   **Vector Store:**
        *   `data/vector_store/persona.index`: The saved FAISS index file.
        *   `data/vector_store/metadata.json`: A JSON file containing the metadata for all vectors in the index.

## Data Model: `PersonaVector`

The core data structure for representing embedded information is the `PersonaVector` class, defined in `persona_modeling/vector_persona_builder.py`.

```python
@dataclass
class PersonaVector:
    """Data class for storing vectors with metadata.

    The 'metadata' field can contain various pieces of information
    about the vector's source, including person-specific details like
    'source_chat_id', 'message_author_type', 'message_author_id',
    and 'message_author_name' for vectors derived from messages.
    """
    vector_id: str      # Unique ID for the vector (e.g., "msg_0", "pref_like_1")
    text_content: str   # The original text content that was embedded
    vector_type: str    # Type of vector (e.g., 'message', 'preference', 'context', 'relationship')
    metadata: Dict      # Dictionary holding various metadata fields
    embedding: Optional[np.ndarray] = None # The numerical embedding (populated after encoding)
```

### Metadata Details

The `metadata` dictionary is crucial for understanding and filtering vectors. Common fields include:

*   **For `message` vectors:**
    *   `message_type`: Category of message style (e.g., "short_responses").
    *   `length`: Character length of the message.
    *   `word_count`: Word count of the message.
    *   `timestamp`: ISO format timestamp of the message.
    *   `service`: Messaging service (e.g., "iMessage").
    *   `contact`: Contact name associated with the message (can be "unknown").
    *   `source_chat_id`: ID of the chat thread.
    *   `message_author_type`: "self" or "external".
    *   `message_author_id`: Contact ID if external, else `None`.
    *   `message_author_name`: Contact name if external, else `None`.

*   **For `preference` vectors:**
    *   `preference_type`: "like", "dislike", or "expanded".
    *   `original_text`: The text from `preferences.yaml` (for likes/dislikes).
    *   `sentiment`: "positive" or "negative".
    *   `category`: For expanded preferences (e.g., "activities", "food").

*   **For `context` vectors:**
    *   `context_type`: Identified context (e.g., "casual_greeting", "making_plans").
    *   `message_length`: Character length of the sample message.
    *   `timestamp`: ISO format timestamp of the sample message.
    *   `contact`: Contact name from the sample message.
    *   `source_chat_id`: ID of the chat thread for the sample message.
    *   `message_author_type`: "self" or "external" for the sample message.
    *   `message_author_id`: Contact ID if external for the sample message, else `None`.
    *   `message_author_name`: Contact name if external for the sample message, else `None`.

*   **For `relationship` vectors:**
    *   `contact`: The primary contact this vector's message relates to.
    *   `total_messages_with_contact`: Total messages exchanged with this contact in the input data.
    *   `message_length`: Character length of the sample message.
    *   `timestamp`: ISO format timestamp of the sample message.
    *   `source_chat_id`: ID of the chat thread for the sample message.
    *   `message_author_type`: "self" or "external" for the sample message.
    *   `message_author_id`: Contact ID if external for the sample message, else `None`.
    *   `message_author_name`: Contact name if external for the sample message, else `None`.


## Vector Store

The generated FAISS index and metadata are saved to the directory specified by `output_dir` in `_save_vector_store` (defaults to `data/vector_store/`).
*   `persona.index`: The binary FAISS index file.
*   `metadata.json`: A JSON file mapping `vector_id` to its text, type, and detailed metadata. This file is essential for interpreting search results from the FAISS index.

The `VectorPersonaBuilder` class also includes a `search_similar` method that demonstrates how to load the index and query it to find vectors semantically similar to a given text string.

## Contributing

Contributions are welcome! If you have ideas for improvements, new features, or bug fixes, please:
1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is likely distributed under the MIT License or a similar open-source license. Please check for a `LICENSE` file in the repository. If one does not exist, the project owner should add one.

---

*This README was generated based on the observed structure and functionality of the project files. Specific details might need adjustments by the project maintainers.*