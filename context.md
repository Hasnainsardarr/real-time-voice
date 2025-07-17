# VoiceBot NLU Project Context

Welcome to the **VoiceBot NLU** project! This file provides all the context and setup instructions you need in Cursor for a smooth development experience.

---

## 🏗 Project Overview

We are building the NLU (intent‑matching) module of a real‑time voice bot for phone calls. Audio capture and transcription are handled elsewhere; your task is to:

1. Read JSON definitions of prompts & intents (`intents.json`).
2. Receive user text and prompt state via a FastAPI endpoint.
3. Compute the best intent match (using spaCy semantic similarity).
4. Return the corresponding action and response (e.g., WAV file name, hangup, or live‑transfer instruction).

## 📁 Repository Structure

```
voicebot-nlu/
├── context.md            # This context file
├── intents.json          # JSON flow tree of prompts, fulfil, objections, fallbacks
├── main.py               # FastAPI application entry point
├── matcher.py            # spaCy-based intent matching logic
├── requirements.txt      # Python dependencies
└── .gitignore            # Exclusions
```

## ⚙️ Prerequisites

- Python 3.8 or higher
- pip
- A virtual environment tool (venv, virtualenv, conda)

## 🚀 Setup & Installation

1. **Clone or create** this project in Cursor.
2. **Open a terminal** in the project root.
3. Create and activate your virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate    # Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_md
   ```

## 📝 File Templates & Snippets

### `requirements.txt`

```text
fastapi
uvicorn
spacy
```

### `main.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
from matcher import get_response

app = FastAPI()

class BotInput(BaseModel):
    prompt: str
    text: str

@app.post('/match_intent')
def match_intent(data: BotInput):
    return get_response(data.prompt, data.text)
```

### `matcher.py`

```python
import json
import spacy

nlp = spacy.load('en_core_web_md')
with open('intents.json', 'r', encoding='utf-8') as f:
    FLOW = json.load(f)['flow']

# Define helper functions: get_prompt_data, get_response
```

## 🔄 Running & Testing

1. **Start the API**:
   ```bash
   uvicorn main:app --reload
   ```
2. **Open API docs** in your browser:
   ```
   http://127.0.0.1:8000/docs
   ```
3. **Invoke** the `/match_intent` endpoint with sample JSON:
   ```json
   {
     "prompt": "2sol",
     "text": "how expensive is it?"
   }
   ```

## 🎯 Next Steps

- Implement full `matcher.py` logic:
  - Load prompt entries, `fulfil`, `objections`, `fallbacks` sections
  - Use spaCy to rank similarity scores
  - Match objections first (if score > threshold)
  - Fallback to default if no confident match
- Add logging and confidence scores to help debug
- Write unit tests for common phrases

---

Feel free to refer back to this context anytime. Happy coding!

