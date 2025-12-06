"""
Pairing rules for recipe categories.
This file can be easily updated with your custom pairing rules.
"""

# Default pairing rules - can be customized
# Format: {category: [list of categories that pair well with it]}
PAIRING_RULES = {
    "appetizers-and-snacks": ["main-dish", "drinks"],
    "desserts": ["main-dish", "drinks"],
    "side-dish": ["main-dish"],
    "main-dish": ["appetizers-and-snacks", "side-dish", "desserts", "drinks"],
    "salad": ["soups-stews-and-chili", "main-dish", "drinks"],
    "bread": ["soups-stews-and-chili", "appetizers-and-snacks"],
    "soups-stews-and-chili": ["salad", "bread"],
    "breakfast-and-brunch": ["side-dish"],
    "drinks": ["desserts"]
}


def get_pairing_rules() -> dict:
    """
    Get the pairing rules dictionary.
    
    Returns:
        dict: Mapping of categories to their paired categories
    """
    return PAIRING_RULES.copy()


def update_pairing_rules(new_rules: dict):
    """
    Update the pairing rules.
    
    Args:
        new_rules: Dictionary mapping categories to their paired categories
    """
    global PAIRING_RULES
    PAIRING_RULES.update(new_rules)

