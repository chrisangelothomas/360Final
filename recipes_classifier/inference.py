"""
Inference module for classifying recipes and finding similar recipes using embeddings.
"""
import os
import ast
import warnings
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics.pairwise import cosine_similarity

from .config import OUTPUT_DIR, DATA_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "recipe_embeddings.npz")


def _get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class RecipeClassifier:
    """Class for classifying recipes and finding similar recipes."""
    
    def __init__(self):
        """Initialize the classifier by loading model, tokenizer, and embeddings."""
        self.device = _get_device()
        self.tokenizer = None
        self.model = None
        self.config = None
        self.id2label = None
        self.label2id = None
        
        # Embedding data
        self.embeddings = None
        self.recipe_texts = None
        self.recipe_categories = None
        self.recipe_labels = None
        
        self._load_model()
        self._load_embeddings()
        self.text_lookup = self._build_text_lookup()
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        
        # Load config
        self.config = AutoConfig.from_pretrained(OUTPUT_DIR)
        self.config.num_labels = 9
        
        # Get label mappings
        if hasattr(self.config, 'id2label') and self.config.id2label:
            if isinstance(list(self.config.id2label.keys())[0], str):
                self.id2label = {int(k): v for k, v in self.config.id2label.items()}
            else:
                self.id2label = self.config.id2label
            self.label2id = self.config.label2id
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            OUTPUT_DIR,
            num_labels=9,
            id2label=self.config.id2label,
            label2id=self.config.label2id
        )
        self.model.eval()
        self.model.config.output_hidden_states = True
        self.model.to(self.device)
    
    def _load_embeddings(self):
        """Load the pre-computed embeddings from disk."""
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(
                f"Embeddings file not found at {EMBEDDINGS_PATH}. "
                "Please run 'python -m recipes_classifier.embeddings' first."
            )
        
        data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.recipe_texts = data['texts']
        self.recipe_categories = data['categories']
        self.recipe_labels = data['labels']
    
    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify a recipe description into one of the 9 categories.
        
        Args:
            text: The recipe description or dish name
            
        Returns:
            tuple: (category_name, confidence_score)
        """
        # Tokenize input
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_id].item()
        
        category = self.id2label[predicted_id]
        return category, confidence
    
    def predict_proba(self, text: str) -> np.ndarray:
        """
        Get probability distribution over all classes for a given text.
        
        Args:
            text: The recipe description or dish name
            
        Returns:
            numpy array: Probability distribution over all classes
        """
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        return probs[0].cpu().numpy()  # Return probability distribution
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get the embedding vector for a given text.
        
        Args:
            text: The recipe description or dish name
            
        Returns:
            numpy array: The embedding vector
        """
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            hidden_states = outputs.hidden_states[-1]
            # Get CLS token embedding
            embedding = hidden_states[:, 0, :].cpu().numpy()
        
        return embedding[0]  # Return first (and only) embedding
    
    def _build_text_lookup(self):
        """Build a lookup dictionary mapping full text to (title, description)."""
        lookup = {}
        for filename in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                continue
            # Suppress dtype warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
                df = pd.read_csv(path, low_memory=False)
            # Build text the same way as in data.py (but without period to match existing embeddings)
            for _, row in df.iterrows():
                parts = []
                for col in ["title", "description", "ingredients", "directions"]:
                    if col not in row or pd.isna(row[col]):
                        continue
                    val = row[col]
                    if isinstance(val, float):
                        continue
                    if col == "directions" and isinstance(val, str):
                        # Try to parse list like "['step1', 'step2', ...]"
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=SyntaxWarning)
                            try:
                                parsed = ast.literal_eval(val)
                                if isinstance(parsed, list):
                                    val = " ".join(parsed)
                            except (ValueError, SyntaxError):
                                pass
                    if isinstance(val, str):
                        parts.append(val)
                if len(parts) >= 2:
                    # Build full text WITHOUT period (to match existing embeddings)
                    full_text = " ".join(parts)
                    # Store title and description separately
                    title = parts[0] if len(parts) > 0 else ""
                    description = parts[1] if len(parts) > 1 else ""
                    lookup[full_text] = (title, description)
        return lookup
    
    def find_similar_recipes(
        self,
        query_text: str,
        category_filter: str = None,
        strategy: str = "moderate_similarity",  # "moderate_similarity" or "opposite"
        skip_n: int = 5,  # For moderate_similarity: skip top N most similar
        top_k: int = 5
    ) -> list[dict]:
        """
        Find similar recipes using cosine similarity on embeddings.
        
        Args:
            query_text: The recipe description to find complementary recipes for
            category_filter: Optional category to filter results
            strategy: "moderate_similarity" (skip top N) or "opposite" (lowest similarity)
            skip_n: Number of most similar recipes to skip (for moderate_similarity)
            top_k: Number of complementary recipes to return
            
        Returns:
            list: List of dicts with 'text', 'category', 'similarity' keys
        """
        # Get embedding for query
        query_embedding = self.get_embedding(query_text)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Filter by category if specified
        if category_filter:
            mask = self.recipe_categories == category_filter
            filtered_embeddings = self.embeddings[mask]
            filtered_texts = self.recipe_texts[mask]
            filtered_categories = self.recipe_categories[mask]
        else:
            filtered_embeddings = self.embeddings
            filtered_texts = self.recipe_texts
            filtered_categories = self.recipe_categories
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]

        if strategy == "moderate_similarity":
            # Sort by similarity descending, skip top N, then take top_k
            sorted_indices = np.argsort(similarities)[::-1]  # Descending
            # Skip the most similar ones
            complementary_indices = sorted_indices[skip_n:skip_n+top_k]
        elif strategy == "opposite":
            # Sort by similarity ascending (lowest first), take top_k
            sorted_indices = np.argsort(similarities)  # Ascending
            complementary_indices = sorted_indices[:top_k]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
  
        # Build results
        results = []
        for idx in complementary_indices:
            text = filtered_texts[idx]
            # Get title and description from lookup, fallback to parsing if not found
            title, description = self.text_lookup.get(text, (text.split()[0] if text.split() else "", text))
            results.append({
                'text': text,
                'title': title,
                'description': description,
                'category': filtered_categories[idx],
                'similarity': float(similarities[idx])
            })
        
        return results


def apply_pairing_rules(dish_category: str, rules: dict) -> list[str]:
    """
    Apply pairing rules to determine which categories pair with the dish category.
    
    Args:
        dish_category: The category of the dish (e.g., "main-dish")
        rules: Dictionary mapping categories to their paired categories
        
    Returns:
        list: List of category names that pair with the dish category
    """
    # Placeholder for rule system - will be implemented when rules are provided
    return rules.get(dish_category, [])

