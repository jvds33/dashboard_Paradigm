
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from config import get_azure_openai_config, get_azure_deployment, get_azure_model

# Load environment variables
load_dotenv()



# ---------- Pydantic schema ----------
class ReviewExcerpts(BaseModel):
    """Excerpts voor één review (1-based index)."""
    review_index: int = Field(..., ge=1, description="1-based index van de review in de input")
    excerpts: List[str] = Field(
        default_factory=list,
        description="Exacte zinnen of aaneengesloten zinsdelen uit deze review die over THEMA gaan (volgorde behouden, geen duplicaten)"
    )

class ThemeExcerpts(BaseModel):
    """Alle excerpts, gegroepeerd per review."""
    reviews: List[ReviewExcerpts]


# ---------- LLM ----------
azure_config = get_azure_openai_config()
deployment = get_azure_deployment()

# Fail fast if anything is missing
for key, val in [
    ("AZURE_OPENAI_API_KEY", azure_config["api_key"]),
    ("AZURE_OPENAI_ENDPOINT", azure_config["azure_endpoint"]),
    ("AZURE_OPENAI_API_VERSION", azure_config["api_version"]),
    ("AZURE_OPENAI_DEPLOYMENT", deployment),
]:
    if not val:
        raise RuntimeError(f"Missing {key} in .env")

# Use the Azure-specific client
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_key=azure_config["api_key"],
    azure_endpoint=azure_config["azure_endpoint"],
    api_version=azure_config["api_version"],
    # temperature=1  # either omit or set to 1
)
structured_llm = llm.with_structured_output(ThemeExcerpts)


# ---------- Prompt (zonder few-shots) ----------
SYSTEM = """Je bent een extreem nauwkeurige informatie-extractor.

DOMEIN (context):
- B2B-klantfeedback (HR, leidinggevenden, directie) over arbodienst-achtige dienstverlening:
  bedrijfsarts, casemanagement, PMO/RI&E, verzuimbegeleiding, keuringen, privacy/AVG, portal/dashboards,
  kosten/transparantie, bereikbaarheid, wachttijden, rapportagekwaliteit.

TAAK:
- Voor ELKE REVIEW: selecteer ALLE fragmenten die inhoudelijk over THEMA gaan.
- Een fragment is óf (a) de volledige zin, óf (b) het kleinste aaneengesloten zinsdeel dat het thema duidelijk maakt.

REGELS:
- Geef tekst exact zoals in de input (zelfde casing, leestekens, spaties); GEEN parafrases, GEEN toevoegingen, GEEN "...".
- Elk fragment valt volledig binnen één enkele originele zin (niet over zinnen combineren).
- Vermijd losse kretologie (bv. alleen "te duur"), tenzij er geen beter passend fragment in de review is.
- Focus uitsluitend op THEMA binnen dit domein; negeer andere onderwerpen (kantoor, catering, etc.) als die niet bij THEMA horen.
- Behoud volgorde van voorkomen binnen elke review; verwijder duplicaten.
- Als een review geen match heeft: lever voor die review een lege lijst `excerpts: []`.

UITVOER:
- Exclusief geldig JSON dat voldoet aan dit schema:
  {{
    "reviews": [
      {{ "review_index": <int 1-based>, "excerpts": ["<exact fragment 1>", "<exact fragment 2>"] }}
    ]
  }}
"""

HUMAN = """THEMA: {theme}

REVIEWS:
{numbered_reviews}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", HUMAN),
])


def extract_excerpts(theme: str, reviews: List[str]) -> ThemeExcerpts:
    """
    Input:
      - theme: het thema waarop je wilt filteren (bv. 'wachttijd', 'rapportagekwaliteit', 'bereikbaarheid', 'AVG').
      - reviews: lijst van reviews (één review kan meerdere zinnen bevatten).
    Output:
      - ThemeExcerpts met per review alle exacte zinnen/subzinnen die over THEMA gaan, in volgorde, zonder duplicaten.
      - Reviews zonder match krijgen excerpts=[].
    """
    # Nummer de reviews 1-based zoals in de prompt verwacht
    numbered = []
    for i, text in enumerate(reviews, start=1):
        if text and text.strip():
            numbered.append(f"{i}) {text.strip()}")
    joined = "\n".join(numbered)

    chain = prompt | structured_llm
    return chain.invoke({"theme": theme, "numbered_reviews": joined})


# ---- Kort voorbeeld ----
if __name__ == "__main__":
    theme = "wachttijd"
    reviews = [
        "Voor een spoedgeval wachten we inmiddels drie weken op een afspraak met de bedrijfsarts. De triage via het portal duurde ook vijf werkdagen voordat we een reactie kregen. De casemanager is verder vriendelijk.",
        "Het dashboard is overzichtelijk, maar rapportages komen soms te laat."
    ]
    result = extract_excerpts(theme, reviews)
    print(result.model_dump())
