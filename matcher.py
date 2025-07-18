import json
import spacy
import logging
from typing import Dict, List, Optional, Tuple
import random
from faiss_matcher import get_faiss_matcher
from fuzzywuzzy import fuzz
import re

# Configure logging for both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intent_matcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy model (kept for keyword matching fallback)
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
OBJECTION_THRESHOLD_HANGUP = 0.65  # Minimum similarity score for wav_response_hangup objections
OBJECTION_THRESHOLD_REGULAR = 0.7  # Minimum similarity score for other objections
FULFIL_THRESHOLD = 0.7             # Minimum similarity score for fulfil matching
DEFAULT_CONFIDENCE = 0.8           # Confidence for direct fulfil responses
FUZZY_WEIGHT = 0.3                 # Weight for fuzzy matching (30%)
SBERT_WEIGHT = 0.7                 # Weight for SBERT matching (70%)

# Initialize FAISS matcher
_faiss_matcher = None

def get_matcher():
    """Get the FAISS matcher instance."""
    global _faiss_matcher
    if _faiss_matcher is None:
        _faiss_matcher = get_faiss_matcher()
    return _faiss_matcher

def preprocess_text(text: str) -> str:
    """
    Preprocess text with typo correction and normalization.
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text with typos corrected
    """
    # Convert to lowercase for processing
    text = text.lower().strip()
    
    # Common typo corrections
    typo_corrections = {
        'solo': 'solar',
        'soler': 'solar',
        'solor': 'solar',
        'solars': 'solar',
        'panals': 'panels',
        'panles': 'panels',
        'expencive': 'expensive',
        'expensiv': 'expensive',
        'dont': "don't",
        'wont': "won't",
        'cant': "can't",
        'im': "i'm",
        'youre': "you're",
        'thats': "that's",
        'its': "it's",
        'voicemal': 'voicemail',
        'voicmail': 'voicemail',
        'telefone': 'telephone',
        'tps': 'TPS'
    }
    
    # Apply typo corrections
    for typo, correction in typo_corrections.items():
        text = re.sub(r'\b' + typo + r'\b', correction, text)
    
    logger.debug(f"Preprocessed text: '{text}'")
    return text

def calculate_hybrid_similarity(text1: str, text2: str) -> Tuple[float, Dict]:
    """
    Calculate hybrid similarity combining fuzzy matching with SBERT similarity.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Tuple of (combined_score, score_breakdown)
    """
    try:
        # Fuzzy matching score
        fuzzy_score = fuzz.token_sort_ratio(text1, text2) / 100.0
        
        # SBERT similarity using FAISS matcher
        matcher = get_matcher()
        # Get SBERT embedding similarity
        user_embedding = matcher.model.encode([text1])
        intent_embedding = matcher.model.encode([text2])
        
        import numpy as np
        import faiss
        
        # Normalize embeddings
        faiss.normalize_L2(user_embedding)
        faiss.normalize_L2(intent_embedding)
        
        # Calculate cosine similarity
        sbert_score = float(np.dot(user_embedding[0], intent_embedding[0]))
        
        # Combine scores with weights
        combined_score = (fuzzy_score * FUZZY_WEIGHT) + (sbert_score * SBERT_WEIGHT)
        
        score_breakdown = {
            'fuzzy_score': fuzzy_score,
            'sbert_score': sbert_score,
            'combined_score': combined_score
        }
        
        logger.debug(f"Hybrid similarity: {text1} vs {text2} -> Fuzzy: {fuzzy_score:.3f}, SBERT: {sbert_score:.3f}, Combined: {combined_score:.3f}")
        
        return combined_score, score_breakdown
        
    except Exception as e:
        logger.error(f"Error calculating hybrid similarity: {e}")
        return 0.0, {'fuzzy_score': 0.0, 'sbert_score': 0.0, 'combined_score': 0.0}

def find_best_objection_match_faiss(user_text: str, objections: List[Dict]) -> Tuple[Optional[Dict], float, Dict]:
    """
    Find the best matching objection using FAISS+SBERT with fuzzy matching.
    
    Args:
        user_text: The user's input text
        objections: List of objection definitions
        
    Returns:
        Tuple of (best_objection, confidence_score, score_breakdown)
    """
    best_match = None
    best_score = 0.0
    best_breakdown = {}
    
    for objection in objections:
        intent_text = objection.get('intent', '')
        if not intent_text:
            continue
            
        # Calculate hybrid similarity
        similarity, breakdown = calculate_hybrid_similarity(user_text, intent_text)
        
        if similarity > best_score:
            best_score = similarity
            best_match = objection
            best_breakdown = breakdown
    
    logger.info(f"Best objection match: {best_match.get('intent', 'None') if best_match else 'None'} with score {best_score:.3f}")
    return best_match, best_score, best_breakdown

def find_best_fulfil_match_faiss(user_text: str, fulfil_list: List[Dict]) -> Tuple[Optional[Dict], float, Dict]:
    """
    Find the best matching fulfil response using FAISS+SBERT with fuzzy matching.
    
    Args:
        user_text: The user's input text
        fulfil_list: List of fulfil definitions
        
    Returns:
        Tuple of (best_fulfil, confidence_score, score_breakdown)
    """
    best_match = None
    best_score = 0.0
    best_breakdown = {}
    
    for fulfil in fulfil_list:
        intent_text = fulfil.get('intent', '')
        if not intent_text or intent_text == '_default_':
            continue
            
        # Calculate hybrid similarity
        similarity, breakdown = calculate_hybrid_similarity(user_text, intent_text)
        
        if similarity > best_score:
            best_score = similarity
            best_match = fulfil
            best_breakdown = breakdown
    
    logger.info(f"Best fulfil match: {best_match.get('intent', 'None') if best_match else 'None'} with score {best_score:.3f}")
    return best_match, best_score, best_breakdown

def get_prompt_data(prompt: str) -> Optional[Dict]:
    """
    Retrieve prompt data from the flow configuration.
    
    Args:
        prompt: The prompt identifier
        
    Returns:
        Dictionary containing prompt data or None if not found
    """
    return FLOW.get(prompt)

def calculate_keyword_similarity(text1: str, text2: str) -> float:
    """
    Calculate keyword-based similarity for better intent matching.
    Preserved for special cases.
    
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

def calculate_similarity_fallback(text1: str, text2: str) -> float:
    """
    Calculate similarity using original spaCy + keyword approach.
    Used as fallback when FAISS doesn't find good matches.
    
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

def get_response(prompt: str, text: str) -> Dict:
    """
    Get the appropriate response using improved hybrid matching with typo correction.
    
    Args:
        prompt: The current prompt identifier
        text: The user's input text
        
    Returns:
        Dictionary containing action, say (WAV filename), confidence, and matched_intent
    """
    logger.info(f"Processing request - Prompt: {prompt}, Text: '{text}'")
    
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
    
    # Preprocess text with typo correction
    user_text = preprocess_text(text)
    logger.info(f"Preprocessed text: '{user_text}'")
    
    # PRIORITY 1: Check for objections using FAISS+SBERT+Fuzzy matching
    objections = prompt_data.get('objections', [])
    if objections:
        best_objection, objection_score, score_breakdown = find_best_objection_match_faiss(user_text, objections)
        
        if best_objection:
            action = best_objection.get('action', 'wav_response')
            
            # Different thresholds based on action type
            threshold = OBJECTION_THRESHOLD_HANGUP if action == 'wav_response_hangup' else OBJECTION_THRESHOLD_REGULAR
            
            if objection_score >= threshold:
                wav_filename = best_objection.get('say', '')
                intent = best_objection.get('intent', '')
                
                logger.info(f"Matched objection: '{intent}' with score {objection_score:.3f} (threshold: {threshold})")
                logger.info(f"Score breakdown - Fuzzy: {score_breakdown.get('fuzzy_score', 0):.3f}, SBERT: {score_breakdown.get('sbert_score', 0):.3f}")
                
                return {
                    "action": action,
                    "say": wav_filename,
                    "confidence": objection_score,
                    "matched_intent": intent
                }
            else:
                logger.info(f"Objection score {objection_score:.3f} below threshold {threshold}, continuing to fulfil matching")
    
    # PRIORITY 2: Check for fulfil responses using FAISS+SBERT+Fuzzy matching
    fulfil_list = prompt_data.get('fulfil', [])
    if fulfil_list:
        best_fulfil, fulfil_score, score_breakdown = find_best_fulfil_match_faiss(user_text, fulfil_list)
        
        if best_fulfil and fulfil_score >= FULFIL_THRESHOLD:
            action = best_fulfil.get('action', 'wav_response')
            wav_filename = best_fulfil.get('say', '')
            intent = best_fulfil.get('intent', '')
            
            logger.info(f"Matched fulfil response: '{intent}' with score {fulfil_score:.3f}")
            logger.info(f"Score breakdown - Fuzzy: {score_breakdown.get('fuzzy_score', 0):.3f}, SBERT: {score_breakdown.get('sbert_score', 0):.3f}")
            
            return {
                "action": action,
                "say": wav_filename,
                "confidence": fulfil_score,
                "matched_intent": intent
            }
        else:
            logger.info(f"Fulfil score {fulfil_score:.3f} below threshold {FULFIL_THRESHOLD}, trying fallback")
    
    # PRIORITY 3: Use fallback responses from intent.json
    fallbacks = prompt_data.get('fallback', [])
    if fallbacks:
        fallback = random.choice(fallbacks)
        action = fallback.get('action', 'wav_response')
        wav_filename = fallback.get('say', '')
        
        logger.info("Using fallback response from intent.json")
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

def get_matcher_stats() -> Dict:
    """Get statistics about the matching system."""
    try:
        matcher = get_matcher()
        faiss_stats = matcher.get_stats()
        
        return {
            "system": "Hybrid FAISS+SBERT + Fuzzy + Keyword Matching",
            "faiss_stats": faiss_stats,
            "total_prompts": len(FLOW),
            "objection_threshold_hangup": OBJECTION_THRESHOLD_HANGUP,
            "objection_threshold_regular": OBJECTION_THRESHOLD_REGULAR,
            "fulfil_threshold": FULFIL_THRESHOLD,
            "fuzzy_weight": FUZZY_WEIGHT,
            "sbert_weight": SBERT_WEIGHT
        }
    except Exception as e:
        logger.error(f"Error getting matcher stats: {e}")
        return {"error": str(e)}

# Validate flow structure on module load
if not validate_flow_structure():
    logger.error("Flow structure validation failed")
    raise ValueError("Invalid flow structure in intent.json") 