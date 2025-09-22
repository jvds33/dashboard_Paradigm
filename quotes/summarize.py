"""Main function for summarizing reviews."""

from typing import List
import json
import pandas as pd

# Handle both relative imports (when used as module) and direct imports (when run directly)
try:
    from .llm_client import create_llm_client
    from .prompts import create_prompt_template
except ImportError:
    from llm_client import create_llm_client
    from prompts import create_prompt_template


def extract_quotes_from_df(df: pd.DataFrame, target_theme: str, themes_column: str = "THEMES") -> str:
    """
    Extract all quotes for a specific theme from DataFrame.
    
    Args:
        df: DataFrame containing the THEMES column
        target_theme: The theme to extract quotes for (e.g., "Medewerkers > Nakomen afspraken")
        themes_column: Name of the column containing theme-quote mappings (default: "THEMES")
        
    Returns:
        str: All quotes for the theme concatenated with newlines
    """
    quotes = []
    
    for _, row in df.iterrows():
        theme_data = row[themes_column]
        
        if pd.isna(theme_data):
            continue
            
        try:
            # Parse JSON string to dict
            if isinstance(theme_data, str):
                theme_dict = json.loads(theme_data)
            elif isinstance(theme_data, dict):
                theme_dict = theme_data
            else:
                continue
                
            # Check if target theme exists and extract quote
            if target_theme in theme_dict:
                quote = theme_dict[target_theme]
                if quote and str(quote).strip():
                    quotes.append(f'"{str(quote).strip()}"')
                    
        except (json.JSONDecodeError, Exception):
            continue
    
    return "\n".join(quotes)


def summarize_theme_from_df(
    df: pd.DataFrame,
    subthema_naam: str,
    themes_column: str = "THEMES",
    publiek: str = "de klantmanagers en het afdelingshoofd",
    aantal_woorden: int = 50,
    toon_beschrijving: str = "zakelijk, helder, energiek maar niet wervend"
) -> str:
    """
    Summarize quotes for a specific theme directly from DataFrame.
    
    Args:
        df: DataFrame containing the THEMES column
        subthema_naam: The theme to summarize on (e.g., 'Medewerkers > Nakomen afspraken')
        themes_column: Name of the column containing theme-quote mappings
        publiek: Target audience for the summary
        aantal_woorden: Maximum number of words for the summary
        toon_beschrijving: Tone description for the summary
        
    Returns:
        str: One paragraph summary that captures the dominant feeling and core reasons
    """
    # Extract quotes for the theme
    quotes_blok = extract_quotes_from_df(df, subthema_naam, themes_column)
    
    if not quotes_blok.strip():
        return f"Geen quotes gevonden voor thema '{subthema_naam}'"
    
    # Use existing summarization function
    return summarize_reviews_to_text(
        subthema_naam=subthema_naam,
        quotes_blok=quotes_blok,
        publiek=publiek,
        aantal_woorden=aantal_woorden,
        toon_beschrijving=toon_beschrijving
    )


def summarize_reviews_to_text(
    subthema_naam: str, 
    quotes_blok: str, 
    publiek: str = "de klantmanagers en het afdelingshoofd",
    aantal_woorden: int = 50,
    toon_beschrijving: str = "zakelijk, helder, energiek maar niet wervend"
) -> str:
    """
    Summarize quotes for a specific subthema.
    
    Args:
        subthema_naam: The subtheme to summarize on (e.g., 'responsiviteit', 'communicatie')
        quotes_blok: Block of quotes as a single string
        publiek: Target audience for the summary
        aantal_woorden: Maximum number of words for the summary
        toon_beschrijving: Tone description for the summary
        
    Returns:
        str: One paragraph summary that captures the dominant feeling and core reasons
    """
    # Create LLM client and prompt template
    llm = create_llm_client()
    prompt = create_prompt_template()
    
    # Create chain and invoke
    chain = prompt | llm
    response = chain.invoke({
        "subthema_naam": subthema_naam,
        "quotes_blok": quotes_blok,
        "publiek": publiek,
        "aantal_woorden": aantal_woorden,
        "toon_beschrijving": toon_beschrijving
    })
    
    # AzureChatOpenAI returns a ChatMessage-like object; extract content
    return getattr(response, "content", str(response))


# ---- Kort voorbeeld ----
if __name__ == "__main__":
    # Create sample DataFrame with THEMES column
    sample_data = {
        'sys_respondentId': [1, 2, 3, 4, 5],
        'THEMES': [
            '{"Medewerkers > Nakomen afspraken": "Ze komen altijd hun afspraken na, heel betrouwbaar", "Service > Snelheid": "Heel snel geholpen"}',
            '{"Medewerkers > Nakomen afspraken": "Soms komen ze te laat, dat is vervelend", "Communicatie": "Duidelijke communicatie"}',
            '{"Medewerkers > Nakomen afspraken": "Afspraken worden stipt nagekomen, professioneel", "Service": "Goede service"}',
            '{"Andere thema": "Irrelevant voor deze test"}',
            '{"Medewerkers > Nakomen afspraken": "Helaas vaak uitgestelde afspraken, frustrerend"}'
        ]
    }
    
    df_test = pd.DataFrame(sample_data)
    
    print("=== TEST WITH DATAFRAME ===")
    target_theme = "Dienstverlening > Kwaliteit van dienstverlening"
    
    # Test quote extraction
    quotes_extracted = extract_quotes_from_df(df_test, target_theme)
    print(f"Extracted quotes for '{target_theme}':")
    print(quotes_extracted)
    print()
    
    # Test full summarization from DataFrame
    result = summarize_theme_from_df(
        df=df_test,
        subthema_naam=target_theme,
        aantal_woorden=60
    )
    print(f"Summary for '{target_theme}':")
    print(result)
    print()
    
    # Also test with the actual DataFrame if available
    try:
        output_path = "/Users/jessevdsluis/Library/CloudStorage/OneDrive-Gedeeldebibliotheken-SpringX/SpringXSharepoint - Documenten/SpringX Analytics/2. Klanten/Paradigma/CX/KTO/Rapportages/KTO_Themes.xlsx"
        df_real = pd.read_excel(output_path)
        
        print("=== TEST WITH REAL DATA ===")
        real_quotes = extract_quotes_from_df(df_real, target_theme)
        if real_quotes.strip():
            print(f"Raw quotes_blok sent to API for '{target_theme}':")
            print("--- START QUOTES_BLOK ---")
            print(real_quotes)
            print("--- END QUOTES_BLOK ---")
            print(f"Number of quotes found: {len(real_quotes.split(chr(10)))}")
            print()
            
            real_result = summarize_theme_from_df(
                df=df_real,
                subthema_naam=target_theme,
                aantal_woorden=60
            )
            print(f"AI-generated summary for '{target_theme}':")
            print(real_result)
        else:
            print(f"No quotes found for '{target_theme}' in real data")
            # Let's see what themes are actually available
            print("\nAvailable themes in the data:")
            available_themes = set()
            for _, row in df_real.iterrows():
                theme_data = row.get('THEMES')
                if pd.notna(theme_data):
                    try:
                        if isinstance(theme_data, str):
                            theme_dict = json.loads(theme_data)
                        elif isinstance(theme_data, dict):
                            theme_dict = theme_data
                        else:
                            continue
                        available_themes.update(theme_dict.keys())
                    except:
                        continue
            
            for theme in sorted(list(available_themes))[:10]:  # Show first 10 themes
                print(f"  - {theme}")
            if len(available_themes) > 10:
                print(f"  ... and {len(available_themes) - 10} more themes")
                
    except Exception as e:
        print(f"Could not load real data: {e}")

