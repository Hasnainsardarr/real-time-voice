# VoiceBot NLU System

A real-time Natural Language Understanding (NLU) system for voice bot applications using **FAISS+SBERT semantic search** with hybrid keyword matching. This system processes transcribed user input and returns appropriate WAV filenames for audio playback in phone call scenarios.

## 🚀 Key Features

- **Advanced Semantic Understanding**: FAISS+SBERT embeddings for accurate intent matching
- **Hybrid Matching**: Combines semantic search with keyword-based objection handling
- **High Performance**: Fast vector similarity search with FAISS indexing
- **Offline Capability**: Works fully offline after initial model download
- **Production Ready**: Comprehensive error handling, logging, and fallback mechanisms
- **Scalable**: Handles hundreds of intents with minimal latency

---

## System Architecture

### Overview

The VoiceBot NLU system uses a **hybrid approach** combining:

1. **FAISS+SBERT Semantic Search**: Primary intent matching using transformer-based embeddings
2. **Keyword Matching**: Preserved for critical objections and special cases
3. **spaCy Fallback**: Safety net for edge cases when semantic matching fails

### Matching Priority

```
1. Keyword Objection Detection (Critical cases like "TPS list")
2. FAISS+SBERT Semantic Search (≥0.65 confidence threshold)
3. spaCy+Keyword Fallback (Low confidence cases)
4. Default Fallback Response (No match found)
```

### Core Components

- **`main.py`**: FastAPI server with REST endpoints
- **`matcher.py`**: Hybrid matching orchestrator
- **`faiss_matcher.py`**: FAISS+SBERT semantic search implementation
- **`intent.json`**: Intent definitions and response mappings

---

## System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended for large intent sets)
- **Storage**: 1GB free space for dependencies and models
- **Internet**: Required for initial model download only

---

## Installation

### 1. Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd voice-bot

# Or download and extract the ZIP file
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Download spaCy language model (for fallback)
python -m spacy download en_core_web_md
```

### 4. Verify Installation

```bash
# Test all dependencies
python -c "import sentence_transformers; import faiss; import spacy; print('✓ All dependencies installed successfully')"
```

---

## Running the System

### Start the Server

```bash
# Method 1: Using uvicorn directly
python -m uvicorn main:app --reload

# Method 2: Using the main.py file
python main.py

# The server will start on http://127.0.0.1:8000
```

### First Run Setup

On the first run, the system will:
1. Download the SBERT model (`all-MiniLM-L6-v2`, ~91MB)
2. Extract all intent phrases from `intent.json`
3. Build and cache the FAISS index for fast similarity search

**Note**: This is a one-time setup. Subsequent runs will use cached models and index.

### Verify Server is Running

1. Open your browser and go to `http://127.0.0.1:8000`
2. You should see: `{"message": "VoiceBot NLU API is running"}`
3. Check the interactive documentation at `http://127.0.0.1:8000/docs`

---

## API Documentation

### Endpoints

#### 1. Health Check

```
GET /health
```

Returns server status and system information.

#### 2. Root Endpoint

```
GET /
```

Returns basic API information.

#### 3. Intent Matching (Main Endpoint)

```
POST /match_intent
```

**Request Body:**

```json
{
  "prompt": "2sol",
  "text": "how expensive is it?"
}
```

**Response:**

```json
{
  "action": "wav_response",
  "say": "savings_with_solar.wav",
  "confidence": 0.9288079345226288,
  "matched_intent": "how much do people save with solar?"
}
```

#### Response Fields

- **action**: Action type (`wav_response`, `wav_response_hangup`, `live_transfer`, `hangup`)
- **say**: WAV filename to play (empty string for hangup actions)
- **confidence**: Confidence score (0.0 to 1.0)
- **matched_intent**: Matched intent text for debugging

---

## Sample Usage

### Using cURL

```bash
# Test price question (semantic matching)
curl -X POST http://127.0.0.1:8000/match_intent \
  -H "Content-Type: application/json" \
  -d '{"prompt": "2sol", "text": "how expensive is it?"}'

# Test greeting (exact match)
curl -X POST http://127.0.0.1:8000/match_intent \
  -H "Content-Type: application/json" \
  -d '{"prompt": "1sol", "text": "yes i can hear you"}'

# Test objection (keyword matching)
curl -X POST http://127.0.0.1:8000/match_intent \
  -H "Content-Type: application/json" \
  -d '{"prompt": "2sol", "text": "i'\''m on the TPS list"}'
```

### Using Python

```python
import requests

# Test the API
response = requests.post(
    "http://127.0.0.1:8000/match_intent",
    json={"prompt": "2sol", "text": "how expensive is it?"}
)

result = response.json()
print(f"Action: {result['action']}")
print(f"WAV file: {result['say']}")
print(f"Confidence: {result['confidence']}")
print(f"Matched intent: {result['matched_intent']}")
```

---

## Testing

### Comprehensive Test Cases

| Test Case                    | Prompt | Input Text                         | Expected Result                         | Matching Method |
| ---------------------------- | ------ | ---------------------------------- | --------------------------------------- | --------------- |
| Price Question               | 2sol   | "how expensive is it?"             | savings_with_solar.wav (92% confidence) | FAISS+SBERT     |
| Alternative Price Phrasing   | 2sol   | "what does it cost?"               | Cost-related response                   | FAISS+SBERT     |
| Greeting                     | 1sol   | "yes i can hear you"               | 1sol.wav (100% confidence)             | FAISS+SBERT     |
| TPS Objection                | 2sol   | "i'm on the TPS list"              | NI_endCall.wav + hangup                 | Keyword         |
| Voicemail Detection          | 1sol   | "Welcome to BT voicemail service"  | Hangup action                           | Keyword         |
| Solar Type Question          | 2sol   | "what type of solar is it?"        | what_type_of_solar.wav                  | FAISS+SBERT     |
| Alternative Type Phrasing    | 2sol   | "what kind of solar panels?"       | what_type_of_solar.wav (95% confidence) | FAISS+SBERT     |

### Manual Testing

Use the interactive documentation:

1. Go to `http://127.0.0.1:8000/docs`
2. Click "Try it out" on the `/match_intent` endpoint
3. Enter test data and execute

---

## Configuration

### Intent Configuration

The system uses `intent.json` for intent definitions containing:

- **Prompts**: Different conversation stages (start, 1sol, 2sol, etc.)
- **Fulfil**: Positive responses that move the conversation forward
- **Objections**: Negative responses or concerns
- **Fallbacks**: Default responses when no good match is found

### Similarity Thresholds

Key configuration constants in `matcher.py`:

- `OBJECTION_THRESHOLD = 0.7`: Minimum score for keyword objection matching
- `FULFIL_THRESHOLD = 0.3`: Minimum score for spaCy fallback matching

Key configuration constants in `faiss_matcher.py`:

- `similarity_threshold = 0.65`: Minimum score for FAISS+SBERT matches
- `model_name = 'all-MiniLM-L6-v2'`: SBERT model for embeddings

### Matching Algorithm Details

#### 1. FAISS+SBERT Semantic Search (Primary)
- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Index**: FAISS IndexFlatIP (cosine similarity with normalized vectors)
- **Intents**: 512 intent phrases indexed for fast search
- **Threshold**: ≥0.65 confidence for acceptance

#### 2. Keyword Matching (Critical Cases)
- **Keyword Categories**: Price, type, savings, voicemail terms
- **Exact Phrase Matching**: Perfect matches get 1.0 confidence
- **Category Matching**: Related terms get 0.95 confidence
- **Preserved for**: Objections, special handling cases

#### 3. spaCy Fallback (Safety Net)
- **Hybrid Approach**: 70% keyword + 30% spaCy semantic similarity
- **Used When**: FAISS confidence < 0.65
- **Purpose**: Ensures system always provides a response

---

## File Structure

```
voice-bot/
├── main.py                 # FastAPI application server
├── matcher.py              # Hybrid matching orchestrator
├── faiss_matcher.py        # FAISS+SBERT semantic search
├── intent.json             # Intent definitions (2838 lines, 512 intents)
├── requirements.txt        # Python dependencies
├── intent_index.faiss      # FAISS index (auto-generated)
├── intent_metadata.pkl     # Intent metadata cache (auto-generated)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## Performance

### Benchmarks

- **Average Response Time**: 1.7s (including model loading)
- **Warm Response Time**: <500ms (after initial load)
- **Index Size**: 512 embeddings, 384 dimensions
- **Memory Usage**: ~200MB (including models)

### Scalability

- **FAISS Index**: Optimized for fast similarity search
- **Batch Processing**: Efficient embedding generation
- **Caching**: Models and index cached for fast startup
- **Horizontal Scaling**: Stateless design supports multiple instances

---

## Troubleshooting

### Common Issues

1. **"Could not load sentence transformer model" error**
   - Ensure internet connection for initial model download
   - Check available disk space (>1GB required)

2. **"Could not load spaCy model" error**
   ```bash
   python -m spacy download en_core_web_md
   ```

3. **"Port already in use" error**
   ```bash
   # Use different port
   python -m uvicorn main:app --reload --port 8001
   ```

4. **"intent.json file not found" error**
   - Ensure `intent.json` is in the project root directory
   - Check file permissions

5. **Low confidence scores**
   - Review intent definitions in `intent.json`
   - Adjust similarity thresholds in `faiss_matcher.py`
   - Consider retraining with different SBERT model

### Performance Optimization

- **Model Caching**: SBERT model and FAISS index cached after first load
- **Batch Processing**: Efficient embedding generation
- **Vector Normalization**: Optimized cosine similarity computation
- **Fallback Hierarchy**: Fast keyword matching before expensive semantic search

### Logging

The system provides detailed logging:

- **Startup**: Model loading and index building progress
- **Matching**: Which method (FAISS/keyword/fallback) matched each request
- **Performance**: Confidence scores and response times
- **Errors**: Detailed error messages for troubleshooting

---

## Deployment

### Production Considerations

1. **Pre-build Models**: Download and cache models during container build
2. **Resource Allocation**: Allocate sufficient RAM for model loading
3. **Load Balancing**: System is stateless and supports horizontal scaling
4. **Health Checks**: Use `/health` endpoint for monitoring

### Docker Deployment

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
# Pre-download models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -m spacy download en_core_web_md
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Development

### Adding New Intents

1. Edit `intent.json` to add new intents
2. Delete cached files (`intent_index.faiss`, `intent_metadata.pkl`)
3. Restart the server to rebuild the index
4. Test new intents using the interactive documentation

### Modifying Matching Logic

1. **FAISS+SBERT**: Edit `faiss_matcher.py` for semantic matching
2. **Keyword Logic**: Edit `matcher.py` for objection handling
3. **Thresholds**: Adjust confidence thresholds in respective files
4. **Models**: Change SBERT model in `faiss_matcher.py`

### System Integration

The system is designed to integrate with:
- **Speech-to-Text**: Processes transcribed input
- **Text-to-Speech**: Returns WAV filenames for audio playback
- **Call Centers**: Handles voice bot conversations
- **CRM Systems**: Can log matched intents and confidence scores

---

## Technical Details

### FAISS+SBERT Implementation

- **Embeddings**: 384-dimensional vectors from `all-MiniLM-L6-v2`
- **Similarity**: Cosine similarity with normalized vectors
- **Index Type**: FAISS IndexFlatIP for exact search
- **Caching**: Persistent storage for models and index

### Hybrid Matching Benefits

- **Accuracy**: Semantic understanding + keyword precision
- **Speed**: Fast vector search with FAISS
- **Reliability**: Multiple fallback mechanisms
- **Maintainability**: Clear separation of concerns

### Offline Capability

After initial setup, the system works completely offline:
- All models cached locally
- No internet required for inference
- Portable across environments
- Suitable for air-gapped deployments

---

## Support

For technical issues:

1. Check the server logs for detailed error messages
2. Verify all dependencies are installed correctly
3. Test with the provided examples in this README
4. Review the interactive API documentation at `/docs`
5. Check system resources (RAM, disk space)

---

## License

This project is proprietary software developed for voice bot applications using advanced semantic matching techniques.

