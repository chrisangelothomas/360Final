"""
Main interface for recipe classification and recommendation system.
"""
from recipes_classifier.inference import RecipeClassifier, apply_pairing_rules
from recipes_classifier.pairing_rules import get_pairing_rules


def main():
    """Main interface for recipe recommendations."""
    print("=" * 60)
    print("Recipe Classification and Recommendation System")
    print("=" * 60)
    print()
    
    # Initialize classifier (loads model and embeddings)
    try:
        classifier = RecipeClassifier()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python -m recipes_classifier.embeddings' first to generate embeddings.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print()
    
    # Get user input
    user_input = input("What are you planning to make? ").strip()
    
    if not user_input:
        print("No input provided. Exiting.")
        return
    
    print("\n" + "-" * 60)
    print("Analyzing your dish...")
    print("-" * 60)
    
    # Classify the dish
    category, confidence = classifier.classify(user_input)
    print(f"\nClassification: {category}")
    print(f"Confidence: {confidence:.2%}")
    
    # Load pairing rules
    pairing_rules = get_pairing_rules()
    
    # Apply pairing rules
    paired_categories = apply_pairing_rules(category, pairing_rules)
    
    if not paired_categories:
        print(f"\nNo pairing rules defined for '{category}' category.")
        print("Finding similar recipes in the same category...")
        recommendations = classifier.find_similar_recipes(
            user_input,
            category_filter=category,
            top_k=2
        )
    else:
        print(f"\nRecommended pairings for '{category}': {', '.join(paired_categories)}")
        print("\nFinding similar recipes in paired categories...")
        
        # Get recommendations from all paired categories
        all_recommendations = []
        for paired_cat in paired_categories:  # Process all paired categories
            recs = classifier.find_similar_recipes(
                user_input,
                category_filter=paired_cat,
                top_k=1
            )
            all_recommendations.extend(recs)
        
        # If we don't have enough, also get from same category
        if len(all_recommendations) < len(paired_categories):
            same_cat_recs = classifier.find_similar_recipes(
                user_input,
                category_filter=category,
                top_k=1
            )
            all_recommendations.extend(same_cat_recs)
        
        recommendations = all_recommendations  # Show all recommendations
    
    # Display recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['category'].upper()}")
            
            # Get title and description
            title = rec.get('title', '')
            description = rec.get('description', rec['text'])
            
            # Display title and description on separate lines
            print(f"   {title}")
            # Truncate description if too long
            description_preview = description[:200]
            if len(description) > 200:
                description_preview += "..."
            print(f"   {description_preview}")
    else:
        print("\nNo recommendations found.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
