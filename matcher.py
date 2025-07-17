import json
import spacy
import logging
from typing import Dict, List, Optional, Tuple
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_md')
    logger.info("Successfully loaded spaCy model: en_core_web_md")
except IOError:
    logger.error("Could not load spaCy model. Please run: python -m spacy download en_core_web_md")
    raise

# Load intents configuration
try:
    with open('intent.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        FLOW = {item['prompt']: item for item in data['flow']}
    logger.info(f"Successfully loaded {len(FLOW)} prompt configurations")
except FileNotFoundError:
    logger.error("intent.json file not found")
    raise
except json.JSONDecodeError as e:
    logger.error(f"Error parsing intent.json: {e}")
    raise

# Configuration constants
OBJECTION_THRESHOLD = 0.7  # Minimum similarity score for objection matching
FULFIL_THRESHOLD = 0.3     # Minimum similarity score for fulfil matching
DEFAULT_CONFIDENCE = 0.8   # Confidence for direct fulfil responses

def get_prompt_data(prompt: str) -> Optional[Dict]:
    """
    Retrieve prompt data from the flow configuration.
    
    Args:
        prompt: The prompt identifier
        
    Returns:
        Dictionary containing prompt data or None if not found
    """
    return FLOW.get(prompt)

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate enhanced similarity between two texts using spaCy + keyword matching.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Keyword-based matching for better accuracy
        keyword_score = calculate_keyword_similarity(text1_lower, text2_lower)
        
        # Semantic similarity using spaCy
        doc1 = nlp(text1_lower)
        doc2 = nlp(text2_lower)
        
        # Handle empty or very short texts
        if len(doc1) == 0 or len(doc2) == 0:
            return keyword_score
            
        semantic_score = doc1.similarity(doc2)
        
        # Combine keyword and semantic scores (weighted towards keyword for better accuracy)
        combined_score = (keyword_score * 0.7) + (semantic_score * 0.3)
        
        return float(min(combined_score, 1.0))
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def calculate_keyword_similarity(text1: str, text2: str) -> float:
    """
    Calculate keyword-based similarity for better intent matching.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Keyword similarity score between 0 and 1
    """
    # Define keyword groups for better matching
    price_keywords = {
        'expensive', 'cost', 'price', 'money', 'afford', 'cheap', 'budget', 
        'pay', 'payment', 'finance', 'much', 'how much', 'cost of'
    }
    
    type_keywords = {
        'type', 'kind', 'what type', 'which type', 'what kind', 'solar type'
    }
    
    savings_keywords = {
        'save', 'savings', 'save money', 'how much save', 'people save', 'will i save'
    }
    
    voicemail_keywords = {
        'voicemail', 'voice mail', 'bt voicemail', 'voicemail service', 'mailbox'
    }
    
    # Check for exact phrase matches first
    if text1 in text2 or text2 in text1:
        return 1.0
    
    # Check for keyword category matches
    text1_words = set(text1.split())
    text2_words = set(text2.split())
    
    def check_keyword_group(keywords, text1_words, text2_words):
        text1_matches = any(kw in text1 for kw in keywords)
        text2_matches = any(kw in text2 for kw in keywords)
        return text1_matches and text2_matches
    
    # Price-related matching
    if check_keyword_group(price_keywords, text1_words, text2_words):
        return 0.95
    
    # Type-related matching
    if check_keyword_group(type_keywords, text1_words, text2_words):
        return 0.95
    
    # Savings-related matching
    if check_keyword_group(savings_keywords, text1_words, text2_words):
        return 0.95
    
    # Voicemail-related matching
    if check_keyword_group(voicemail_keywords, text1_words, text2_words):
        return 0.95
    
    # Word overlap similarity
    common_words = text1_words.intersection(text2_words)
    if common_words:
        total_words = len(text1_words.union(text2_words))
        return len(common_words) / total_words
    
    return 0.0

def find_best_objection_match(user_text: str, objections: List[Dict]) -> Tuple[Optional[Dict], float]:
    """
    Find the best matching objection for the user text.
    
    Args:
        user_text: The user's input text
        objections: List of objection definitions
        
    Returns:
        Tuple of (best_objection, confidence_score)
    """
    best_match = None
    best_score = 0.0
    
    for objection in objections:
        intent_text = objection.get('intent', '')
        
        similarity = calculate_similarity(user_text, intent_text)
        
        if similarity > best_score:
            best_score = similarity
            best_match = objection
                
    return best_match, best_score

def find_best_fulfil_match(user_text: str, fulfil_list: List[Dict]) -> Tuple[Optional[Dict], float]:
    """
    Find the best matching fulfil response for the user text.
    
    Args:
        user_text: The user's input text
        fulfil_list: List of fulfil definitions
        
    Returns:
        Tuple of (best_fulfil, confidence_score)
    """
    best_match = None
    best_score = 0.0
    
    for fulfil in fulfil_list:
        intent_text = fulfil.get('intent', '')
        
        similarity = calculate_similarity(user_text, intent_text)
        
        if similarity > best_score:
            best_score = similarity
            best_match = fulfil
                
    return best_match, best_score

def get_response(prompt: str, text: str) -> Dict:
    """
    Get the appropriate response for user text based on the prompt context.
    
    Args:
        prompt: The current prompt identifier
        text: The user's input text
        
    Returns:
        Dictionary containing action, say (WAV filename), confidence, and matched_intent
    """
    logger.info(f"Processing request - Prompt: {prompt}, Text: {text}")
    
    # Get prompt data
    prompt_data = get_prompt_data(prompt)
    if not prompt_data:
        logger.warning(f"Unknown prompt: {prompt}")
        return {
            "action": "hangup",
            "say": "",
            "confidence": 0.0,
            "matched_intent": "unknown_prompt"
        }
    
    # Clean and normalize user text
    user_text = text.strip().lower()
    
    # Check for objections first (highest priority)
    objections = prompt_data.get('objections', [])
    if objections:
        best_objection, objection_score = find_best_objection_match(user_text, objections)
        
        if best_objection and objection_score >= OBJECTION_THRESHOLD:
            action = best_objection.get('action', 'wav_response')
            wav_filename = best_objection.get('say', '')
            intent = best_objection.get('intent', '')
            
            logger.info(f"Matched objection: {intent} with score {objection_score}")
            
            return {
                "action": action,
                "say": wav_filename,
                "confidence": objection_score,
                "matched_intent": intent
            }
    
    # Check for fulfil responses (direct answers)
    fulfil_list = prompt_data.get('fulfil', [])
    if fulfil_list:
        best_fulfil, fulfil_score = find_best_fulfil_match(user_text, fulfil_list)
        
        if best_fulfil and fulfil_score >= FULFIL_THRESHOLD:
            action = best_fulfil.get('action', 'wav_response')
            wav_filename = best_fulfil.get('say', '')
            intent = best_fulfil.get('intent', '')
            
            logger.info(f"Matched fulfil response: {intent} with score {fulfil_score}")
            
            return {
                "action": action,
                "say": wav_filename,
                "confidence": fulfil_score,
                "matched_intent": intent
            }
    
    # Fall back to default responses
    fallbacks = prompt_data.get('fallback', [])
    if fallbacks:
        fallback = random.choice(fallbacks)
        action = fallback.get('action', 'wav_response')
        wav_filename = fallback.get('say', '')
        
        logger.info("Using fallback response")
        return {
            "action": action,
            "say": wav_filename,
            "confidence": 0.5,
            "matched_intent": "fallback"
        }
    
    # Ultimate fallback
    logger.warning("No appropriate response found, using ultimate fallback")
    return {
        "action": "hangup",
        "say": "",
        "confidence": 0.0,
        "matched_intent": "no_match"
    }

def get_available_prompts() -> List[str]:
    """
    Get list of available prompt identifiers.
    
    Returns:
        List of prompt identifiers
    """
    return list(FLOW.keys())

def validate_flow_structure() -> bool:
    """
    Validate the structure of the loaded flow configuration.
    
    Returns:
        True if structure is valid, False otherwise
    """
    try:
        for prompt_id, prompt_data in FLOW.items():
            if not isinstance(prompt_data, dict):
                logger.error(f"Invalid prompt data for {prompt_id}: not a dictionary")
                return False
                
            # Check required sections
            if 'fulfil' not in prompt_data and 'fallback' not in prompt_data:
                logger.error(f"Prompt {prompt_id} missing both fulfil and fallback sections")
                return False
                
            # Validate objections structure
            objections = prompt_data.get('objections', [])
            for objection in objections:
                if not isinstance(objection, dict):
                    logger.error(f"Invalid objection structure in {prompt_id}")
                    return False
                if 'intent' not in objection or 'action' not in objection:
                    logger.error(f"Missing required fields in objection for {prompt_id}")
                    return False
                    
            # Validate fulfil structure
            fulfil_list = prompt_data.get('fulfil', [])
            for fulfil in fulfil_list:
                if not isinstance(fulfil, dict):
                    logger.error(f"Invalid fulfil structure in {prompt_id}")
                    return False
                if 'intent' not in fulfil or 'action' not in fulfil:
                    logger.error(f"Missing required fields in fulfil for {prompt_id}")
                    return False
                    
            # Validate fallback structure
            fallback_list = prompt_data.get('fallback', [])
            for fallback in fallback_list:
                if not isinstance(fallback, dict):
                    logger.error(f"Invalid fallback structure in {prompt_id}")
                    return False
                if 'action' not in fallback:
                    logger.error(f"Missing required fields in fallback for {prompt_id}")
                    return False
        
        logger.info("Flow structure validation passed")
        return True
    except Exception as e:
        logger.error(f"Error validating flow structure: {e}")
        return False

# Validate flow structure on module load
if not validate_flow_structure():
    logger.error("Flow structure validation failed")
    raise ValueError("Invalid flow structure in intent.json") 