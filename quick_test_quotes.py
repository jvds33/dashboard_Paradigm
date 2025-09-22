"""Quick test script for quotes module - easy to modify for your own data."""

import pandas as pd
from quotes import summarize_reviews_to_text
import textwrap


def quick_test(theme: str, df: pd.DataFrame, text_column: str = 'text'):
    """
    Quick test function for the quotes module.
    
    Args:
        theme: The theme to summarize on
        df: DataFrame containing the review data  
        text_column: Name of the column containing the review text
    """
    print(f"🔍 Testing theme: {theme}")
    print("=" * 60)
    
    # Extract reviews
    reviews = df[text_column].dropna().astype(str).tolist()
    reviews = [review.strip() for review in reviews if review.strip()]
    
    print(f"📊 Found {len(reviews)} reviews")
    print()
    
    # Generate summary
    result = summarize_reviews_to_text(theme, reviews)
    
    # Format output nicely
    print("📝 SUMMARY:")
    print("-" * 60)
    paragraphs = result.split('\n\n')
    for i, paragraph in enumerate(paragraphs, 1):
        if paragraph.strip():
            wrapped = textwrap.fill(paragraph.strip(), width=70, initial_indent="   ", subsequent_indent="   ")
            print(f"   {wrapped}")
            print()
    print("=" * 60)


def load_real_data():
    """Load real data from the Excel file."""
    output_path = "/Users/jessevdsluis/Library/CloudStorage/OneDrive-Gedeeldebibliotheken-SpringX/SpringXSharepoint - Documenten/SpringX Analytics/Klanten/Paradigma/CX/KTO/Rapportages/KTO_Themes.xlsx"
    
    try:
        df = pd.read_excel(output_path)
        print(f"✅ Loaded data from Excel file")
        print(f"📊 Dataset shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading Excel: {e}")
        return None


# Example usage - modify this section for your own data
if __name__ == "__main__":
    # Load real data
    df = load_real_data()
    
    if df is not None and 'Label' in df.columns:
        # Filter for Resolu
        df_resolu = df[df["Label"] == "Resolu"]
        print(f"📊 Resolu reviews: {len(df_resolu)}")
        
        # Test with default theme
        default_theme = "Communicatie & Informatievoorziening > Operationele communicatie"
        quick_test(default_theme, df_resolu)
        
        # Test other themes
        other_themes = [
            "Communicatie & Informatievoorziening > Transparantie",
            "Samenwerking & partnership > Vertrouwen"
        ]
        
        for theme in other_themes:
            quick_test(theme, df_resolu)
            print()
    else:
        # Fallback to sample data
        print("📝 Using sample data...")
        sample_data = {
            'text': [
                "De communicatie verloopt goed, duidelijke updates over de voortgang.",
                "Operationele communicatie is helder en tijdig.",
                "Goede informatievoorziening, we weten altijd waar we aan toe zijn.",
                "Communicatie kan beter, soms moeten we zelf vragen stellen.",
                "Duidelijke communicatie over procedures en processen."
            ],
            'Label': ['Resolu', 'Resolu', 'Resolu', 'Resolu', 'Resolu']
        }
        df = pd.DataFrame(sample_data)
        df_resolu = df[df["Label"] == "Resolu"]
        
        default_theme = "Communicatie & Informatievoorziening > Operationele communicatie"
        quick_test(default_theme, df_resolu)
