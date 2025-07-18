import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import Dict, List, Optional, Tuple, Any
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSIntentMatcher:
    """
    FAISS-based semantic intent matching using SBERT embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.65):
        """
        Initialize the FAISS intent matcher.
        
        Args:
            model_name: Name of the sentence transformer model to use
            similarity_threshold: Minimum similarity score for valid matches
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.index = None
        self.intent_data = []  # Store intent metadata
        self.intent_embeddings = None
        self.flow_data = {}
        
        # File paths for caching
        self.index_path = "intent_index.faiss"
        self.metadata_path = "intent_metadata.pkl"
        
        # Initialize the system
        self._load_model()
        self._load_intents()
        self._build_or_load_index()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Successfully loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise
    
    def _load_intents(self):
        """Load and extract intent phrases from intent.json."""
        try:
            with open('intent.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Store flow data for responses
            self.flow_data = {item['prompt']: item for item in data['flow']}
            
            # Extract all intent phrases with metadata
            self.intent_data = []
            intent_phrases = []
            
            for flow_item in data['flow']:
                prompt = flow_item['prompt']
                
                # Extract fulfil intents
                for fulfil in flow_item.get('fulfil', []):
                    intent_text = fulfil.get('intent', '')
                    if intent_text and intent_text != '_default_':
                        self.intent_data.append({
                            'prompt': prompt,
                            'type': 'fulfil',
                            'intent': intent_text,
                            'action': fulfil.get('action', 'wav_response'),
                            'say': fulfil.get('say', ''),
                            'original_data': fulfil
                        })
                        intent_phrases.append(intent_text)
                
                # Extract objection intents
                for objection in flow_item.get('objections', []):
                    intent_text = objection.get('intent', '')
                    if intent_text:
                        self.intent_data.append({
                            'prompt': prompt,
                            'type': 'objection',
                            'intent': intent_text,
                            'action': objection.get('action', 'wav_response'),
                            'say': objection.get('say', ''),
                            'original_data': objection
                        })
                        intent_phrases.append(intent_text)
            
            logger.info(f"Extracted {len(self.intent_data)} intent phrases from {len(self.flow_data)} prompts")
            self.intent_phrases = intent_phrases
            
        except Exception as e:
            logger.error(f"Error loading intents: {e}")
            raise
    
    def _build_or_load_index(self):
        """Build FAISS index or load from cache if available."""
        # Check if cached index exists
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                logger.info("Loading cached FAISS index...")
                self.index = faiss.read_index(self.index_path)
                
                with open(self.metadata_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Verify cache is still valid
                if (len(cached_data['intent_data']) == len(self.intent_data) and
                    cached_data['model_name'] == self.model_name):
                    logger.info("Using cached FAISS index")
                    return
                else:
                    logger.info("Cache is outdated, rebuilding index...")
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}, rebuilding...")
        
        # Build new index
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index from intent phrases."""
        try:
            logger.info("Building FAISS index with sentence embeddings...")
            
            # Generate embeddings for all intent phrases
            embeddings = self.model.encode(self.intent_phrases, convert_to_tensor=False)
            embeddings = np.array(embeddings).astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.intent_embeddings = embeddings
            
            # Build FAISS index using inner product (cosine similarity with normalized vectors)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            self.index.add(embeddings)
            
            logger.info(f"Built FAISS index with {self.index.ntotal} embeddings, dimension {dimension}")
            
            # Cache the index and metadata
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise
    
    def _save_index(self):
        """Save FAISS index and metadata to cache."""
        try:
            faiss.write_index(self.index, self.index_path)
            
            cache_data = {
                'intent_data': self.intent_data,
                'model_name': self.model_name,
                'intent_phrases': self.intent_phrases
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info("Saved FAISS index and metadata to cache")
        except Exception as e:
            logger.warning(f"Failed to save index cache: {e}")
    
    def find_best_match(self, user_text: str, prompt: str = None) -> Tuple[Optional[Dict], float]:
        """
        Find the best matching intent using FAISS semantic search.
        
        Args:
            user_text: The user's input text
            prompt: Optional prompt context to filter results
            
        Returns:
            Tuple of (best_match_data, confidence_score)
        """
        try:
            # Embed the user text
            user_embedding = self.model.encode([user_text], convert_to_tensor=False)
            user_embedding = np.array(user_embedding).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(user_embedding)
            
            # Search in FAISS index
            k = min(10, self.index.ntotal)  # Top-k results
            similarities, indices = self.index.search(user_embedding, k)
            
            # Filter results by prompt if specified
            best_match = None
            best_score = 0.0
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                    
                intent_info = self.intent_data[idx]
                
                # Filter by prompt if specified
                if prompt and intent_info['prompt'] != prompt:
                    continue
                
                # Check similarity threshold
                if similarity >= self.similarity_threshold:
                    if similarity > best_score:
                        best_score = similarity
                        best_match = intent_info
                        
            return best_match, float(min(best_score, 1.0))
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            return None, 0.0
    
    def get_prompt_fallback(self, prompt: str) -> Optional[Dict]:
        """
        Get fallback response for a prompt.
        
        Args:
            prompt: The prompt identifier
            
        Returns:
            Fallback response data or None
        """
        prompt_data = self.flow_data.get(prompt)
        if not prompt_data:
            return None
            
        fallbacks = prompt_data.get('fallback', [])
        if fallbacks:
            # Return first fallback (could be randomized)
            fallback = fallbacks[0]
            return {
                'prompt': prompt,
                'type': 'fallback',
                'intent': 'fallback',
                'action': fallback.get('action', 'wav_response'),
                'say': fallback.get('say', ''),
                'original_data': fallback
            }
        return None
    
    def get_response(self, prompt: str, text: str) -> Dict:
        """
        Get the appropriate response using FAISS semantic matching.
        
        Args:
            prompt: The current prompt identifier
            text: The user's input text
            
        Returns:
            Dictionary containing action, say, confidence, and matched_intent
        """
        logger.info(f"FAISS matching - Prompt: {prompt}, Text: {text}")
        
        # Clean user text
        user_text = text.strip().lower()
        
        # Use FAISS to find best match
        best_match, confidence = self.find_best_match(user_text, prompt)
        
        if best_match and confidence >= self.similarity_threshold:
            logger.info(f"FAISS matched: {best_match['intent']} with confidence {confidence}")
            
            return {
                "action": best_match['action'],
                "say": best_match['say'],
                "confidence": confidence,
                "matched_intent": best_match['intent']
            }
        
        # Try fallback response
        fallback = self.get_prompt_fallback(prompt)
        if fallback:
            logger.info("Using fallback response")
            return {
                "action": fallback['action'],
                "say": fallback['say'],
                "confidence": 0.5,
                "matched_intent": "fallback"
            }
        
        # Ultimate fallback
        logger.warning("No match found, using ultimate fallback")
        return {
            "action": "hangup",
            "say": "",
            "confidence": 0.0,
            "matched_intent": "no_match"
        }
    
    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt identifiers."""
        return list(self.flow_data.keys())
    
    def get_stats(self) -> Dict:
        """Get statistics about the matcher."""
        return {
            'total_intents': len(self.intent_data),
            'total_prompts': len(self.flow_data),
            'model_name': self.model_name,
            'similarity_threshold': self.similarity_threshold,
            'index_size': self.index.ntotal if self.index else 0
        }

# Global instance
_faiss_matcher = None

def get_faiss_matcher() -> FAISSIntentMatcher:
    """Get or create the global FAISS matcher instance."""
    global _faiss_matcher
    if _faiss_matcher is None:
        _faiss_matcher = FAISSIntentMatcher()
    return _faiss_matcher 