import os
import json
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field


# --- Schema for GENERATION requests (all fields required) ---
class GenerationRequest(BaseModel):
    gene: Optional[str] = Field(description="The gene mutation symbol (e.g., ABCA4). Must be one from the valid list. If invalid, this is null.")
    laterality: str = Field(description="The eye laterality. Must be 'L' or 'R'.")
    age: int = Field(description="The patient age in years. Integer between 0 and 100.")


# --- Schema for EDIT requests (only changed fields are set) ---
class EditRequest(BaseModel):
    gene: Optional[str] = Field(default=None, description="The NEW gene mutation symbol, only if the user wants to change it. null if unchanged.")
    laterality: Optional[str] = Field(default=None, description="The NEW eye laterality ('L' or 'R'), only if the user wants to change it. null if unchanged.")
    age: Optional[int] = Field(default=None, description="The NEW patient age, only if the user wants to change it. null if unchanged.")


def classify_intent(user_prompt):
    """
    Determine whether the user wants to 'generate' a new image or 'edit' the current one.
    Uses keyword matching for speed and reliability.
    Returns 'edit' or 'generate'.
    """
    prompt_lower = user_prompt.lower()
    
    edit_keywords = [
        "edit", "change", "modify", "alter", "adjust", "update",
        "age it", "age the", "make it", "make the", "switch",
        "what if", "what would", "what about", "look like",
        "convert", "transform", "turn it", "turn the",
        "increase", "decrease", "reduce", "raise", "lower",
        "older", "younger", "flip", "swap",
    ]
    
    generate_keywords = [
        "generate", "create", "show me", "new image", "produce",
        "make me a", "give me a", "start fresh", "from scratch",
    ]
    
    # Check generate keywords first (stronger signal)
    for kw in generate_keywords:
        if kw in prompt_lower:
            return "generate"
    
    # Then check edit keywords
    for kw in edit_keywords:
        if kw in prompt_lower:
            return "edit"
    
    # Default to generate if no clear signal
    return "generate"


# --- GENERATION feature extraction (original behaviour) ---
def extract_features(user_prompt, valid_genes_list):
    """Uses Llama3 to convert natural language into structured JSON for generation."""
    ollama_base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    if not ollama_base_url.startswith("http"):
        ollama_base_url = f"http://{ollama_base_url}"
    
    parser = JsonOutputParser(pydantic_object=GenerationRequest)
        
    template = """
    You are a medical data assistant. Extract clinical parameters from the user's request.
    
    VALID GENES: {valid_genes}
    VALID LATERALITY: "L" (Left) or "R" (Right). Default to "R" if not specified.
    VALID AGE: Integer between 0 and 100. Default to 55 if not specified.
    
    RULES:
    - Map disease names to genes (e.g. "Stargardt" -> "ABCA4").
    - CRITICAL: If the user asks for a gene NOT in the valid list, set "gene" to null. DO NOT GUESS.
    - If laterality is unspecified, default to "R".
    - If age is unspecified, default to 55.
    - output ONLY the JSON object, nothing else.
    
    {format_instructions}
    
    User Request: {query}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "valid_genes"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    try:
        model = ChatOllama(base_url=ollama_base_url, model="llama3", temperature=0)
        chain = prompt | model | parser
        return chain.invoke({"query": user_prompt, "valid_genes": ", ".join(valid_genes_list)})
    except Exception as e:
        return {"error": str(e), "gene": None, "laterality": "R", "age": 55}


# --- EDIT feature extraction (only changed fields) ---
def extract_edit_features(user_prompt, valid_genes_list, current_params):
    """
    Uses Llama3 to extract ONLY the parameters the user wants to change.
    Parameters not mentioned remain null (meaning keep current value).
    
    Args:
        user_prompt: The user's edit request
        valid_genes_list: List of valid gene symbols
        current_params: Dict with current 'gene', 'laterality', 'age'
    
    Returns:
        Dict with 'gene', 'laterality', 'age' — only changed fields are non-null.
    """
    ollama_base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    if not ollama_base_url.startswith("http"):
        ollama_base_url = f"http://{ollama_base_url}"
    
    parser = JsonOutputParser(pydantic_object=EditRequest)
    
    template = """
    You are a medical data assistant. The user wants to EDIT an existing medical image.
    
    The current image has these parameters:
    - Gene: {current_gene}
    - Laterality: {current_laterality} (L=Left, R=Right)
    - Age: {current_age}
    
    Extract ONLY the parameters the user wants to CHANGE. Set unchanged parameters to null.
    
    VALID GENES: {valid_genes}
    VALID LATERALITY: "L" (Left) or "R" (Right).
    VALID AGE: Integer between 0 and 100.
    
    RULES:
    - ONLY set a field if the user explicitly wants to change it.
    - Map disease names to genes (e.g. "Stargardt" -> "ABCA4").
    - CRITICAL: If the user asks for a gene NOT in the valid list, set "gene" to null.
    - If the user does NOT mention a parameter, set it to null (keep current value).
    - Output ONLY the JSON object, nothing else.
    
    {format_instructions}
    
    User Request: {query}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "valid_genes", "current_gene", "current_laterality", "current_age"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    try:
        model = ChatOllama(base_url=ollama_base_url, model="llama3", temperature=0)
        chain = prompt | model | parser
        result = chain.invoke({
            "query": user_prompt,
            "valid_genes": ", ".join(valid_genes_list),
            "current_gene": current_params.get("gene", "Unknown"),
            "current_laterality": current_params.get("laterality", "R"),
            "current_age": current_params.get("age", 55),
        })
        return result
    except Exception as e:
        return {"error": str(e), "gene": None, "laterality": None, "age": None}