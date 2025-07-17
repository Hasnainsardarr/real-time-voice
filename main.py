from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from matcher import get_response
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VoiceBot NLU API",
    description="Natural Language Understanding service for voice bot intent matching - returns WAV filenames for audio playback",
    version="1.0.0"
)

class BotInput(BaseModel):
    prompt: str
    text: str

class BotResponse(BaseModel):
    action: str  # wav_response, wav_response_hangup, live_transfer, hangup
    say: str = ""  # WAV filename or empty string
    confidence: float = 0.0
    matched_intent: str = None

@app.get("/")
def root():
    return {"message": "VoiceBot NLU API is running"}

@app.post('/match_intent', response_model=BotResponse)
def match_intent(data: BotInput):
    """
    Match user text against defined intents for a given prompt.
    
    Args:
        data: BotInput containing prompt and user text
        
    Returns:
        BotResponse with action, WAV filename, and confidence score
    """
    try:
        logger.info(f"Processing intent matching for prompt: {data.prompt}, text: {data.text}")
        result = get_response(data.prompt, data.text)
        logger.info(f"Intent matching result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing intent matching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "VoiceBot NLU API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 