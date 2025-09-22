"""Test script for the quotes module with DataFrame input and nice output formatting."""

import pandas as pd
from quotes import summarize_reviews_to_text
import textwrap


def test_quotes_with_dataframe(theme: str, df: pd.DataFrame, text_column: str = 'text'):
    """
    Test the quotes module with a DataFrame containing review text.
    
    Args:
        theme: The theme to summarize on (e.g., 'wachttijd', 'rapportagekwaliteit', 'bereikbaarheid', 'AVG')
        df: DataFrame containing the review data
        text_column: Name of the column containing the review text
    """
    print("=" * 80)
    print(f"🔍 QUOTES TEST - THEMA: {theme.upper()}")
    print("=" * 80)
    
    # Extract reviews from DataFrame
    if text_column not in df.columns:
        print(f"❌ Error: Column '{text_column}' not found in DataFrame")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Filter out empty/null reviews
    reviews = df[text_column].dropna().astype(str).tolist()
    reviews = [review.strip() for review in reviews if review.strip()]
    
    print(f"📊 Dataset Info:")
    print(f"   • Total rows in DataFrame: {len(df)}")
    print(f"   • Non-empty reviews found: {len(reviews)}")
    print(f"   • Theme: {theme}")
    print()
    
    if not reviews:
        print("❌ No valid reviews found in the dataset")
        return
    
    print("📝 Sample Reviews:")
    print("-" * 40)
    for i, review in enumerate(reviews[:3], 1):  # Show first 3 reviews
        wrapped = textwrap.fill(review, width=70, initial_indent="   ", subsequent_indent="   ")
        print(f"{i}. {wrapped}")
        print()
    
    if len(reviews) > 3:
        print(f"   ... and {len(reviews) - 3} more reviews")
        print()
    
    print("🤖 Generating Summary...")
    print("-" * 40)
    
    try:
        # Generate summary using the quotes module
        result_text = summarize_reviews_to_text(theme, reviews)
        
        print("✅ SUMMARY GENERATED:")
        print("=" * 80)
        
        # Format the output nicely
        paragraphs = result_text.split('\n\n')
        for i, paragraph in enumerate(paragraphs, 1):
            if paragraph.strip():
                print(f"📄 Paragraph {i}:")
                wrapped = textwrap.fill(paragraph.strip(), width=75, initial_indent="   ", subsequent_indent="   ")
                print(wrapped)
                print()
        
        print("=" * 80)
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        print("Please check your Azure OpenAI configuration and try again.")


def load_real_data():
    """Load real data from the Excel file."""
    output_path = "/Users/jessevdsluis/Library/CloudStorage/OneDrive-Gedeeldebibliotheken-SpringX/SpringXSharepoint - Documenten/SpringX Analytics/Klanten/Paradigma/CX/KTO/Rapportages/KTO_Themes.xlsx"
    
    try:
        # Try to load the Excel file
        df = pd.read_excel(output_path)
        print(f"✅ Successfully loaded data from: {output_path}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {output_path}")
        print("📝 Using sample data instead...")
        return create_sample_dataframe()
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        print("📝 Using sample data instead...")
        return create_sample_dataframe()


def create_sample_dataframe():
    """Create a sample DataFrame for testing."""
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'text': [
            "Voor een spoedgeval wachten we inmiddels drie weken op een afspraak met de bedrijfsarts. De triage via het portal duurde ook vijf werkdagen voordat we een reactie kregen. De casemanager is verder vriendelijk.",
            "Het dashboard is overzichtelijk, maar rapportages komen soms te laat. De interface is gebruiksvriendelijk.",
            "De casemanager is verder vriendelijk, maar hij kan zich wel beter aanpassen aan de klant. Communicatie verloopt goed.",
            "Wachttijden zijn te lang, vooral voor spoedgevallen. Het systeem werkt wel stabiel.",
            "Goede service, snelle reactie op vragen. Rapportages zijn duidelijk en overzichtelijk.",
            "Het portaal is traag en soms niet beschikbaar. Dit zorgt voor vertragingen in de workflow.",
            "Professionele begeleiding, duidelijke communicatie. Alleen de wachttijden kunnen beter.",
            "Systeem werkt goed, maar de rapportage functionaliteit kan worden verbeterd. Meer real-time data zou fijn zijn."
        ],
        'rating': [3, 4, 3, 2, 5, 2, 4, 3],
        'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20', '2024-01-21', '2024-01-22']
    }
    return pd.DataFrame(sample_data)


# ---- Test Examples ----
if __name__ == "__main__":
    print("🚀 QUOTES MODULE TEST SUITE")
    print("=" * 80)
    
    # Load real data or fallback to sample
    df = load_real_data()
    
    # Filter for Resolu
    if 'Label' in df.columns:
        df_resolu = df[df["Label"] == "Resolu"]
        print(f"📊 Resolu reviews found: {len(df_resolu)}")
    else:
        print("⚠️  No 'Label' column found, using all data")
        df_resolu = df
    
    # Test with default theme
    default_theme = "Communicatie & Informatievoorziening > Operationele communicatie"
    
    print(f"\n{'='*20} TESTING: {default_theme.upper()} {'='*20}")
    test_quotes_with_dataframe(default_theme, df_resolu, 'text')
    print("\n" + "="*80 + "\n")
    
    # Test other themes if desired
    other_themes = [
        "Communicatie & Informatievoorziening > Transparantie",
        "Samenwerking & partnership > Vertrouwen",
        "Dienstverlening > Kwaliteit van dienstverlening"
    ]
    
    for theme in other_themes:
        print(f"\n{'='*20} TESTING: {theme.upper()} {'='*20}")
        test_quotes_with_dataframe(theme, df_resolu, 'text')
        print("\n" + "="*80 + "\n")
    
    print("🎉 All tests completed!")
