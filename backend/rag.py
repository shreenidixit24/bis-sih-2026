"""
rag.py - Retrieval Augmented Generation (RAG) Engine for BIS AI Assistant

This module handles:
1. Knowledge Base Loading - loads BIS data from JSON files
2. Intent Detection - determines what type of question the user asked
3. Knowledge Base Search - finds relevant information using keyword matching
4. Response Generation - builds a structured response with sources

Architecture:
    User Question
        ↓
    Intent Detection
        ↓
    Knowledge Base Search
        ↓
    Retrieve Relevant Information
        ↓
    AI Response (via LLM if configured, else rule-based fallback)
        ↓
    Source Citation

NOTE: This is a DEMO implementation for SIH 2026 hackathon.
For production: Replace the fallback_response() with a real LLM API call.
"""

import json
import os
import re
from pathlib import Path

# Optional: load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Path to the BIS knowledge base JSON file
DATA_FILE = Path(__file__).parent.parent / 'data' / 'bis_data.json'

# If you have an OpenAI API key, set it in .env as: OPENAI_API_KEY=sk-...
# If you have a Google Gemini API key, set it in .env as: GEMINI_API_KEY=...
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# ─────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────

def load_knowledge_base():
    """Load the BIS knowledge base from the JSON file."""
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: Knowledge base file not found at {DATA_FILE}")
        return {}
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in knowledge base: {e}")
        return {}

# Load the knowledge base once when the module is imported
KNOWLEDGE_BASE = load_knowledge_base()

# ─────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────

def detect_intent(question):
    """
    Detect the intent of the user's question.
    
    Returns one of:
    - 'certification' : user wants to know about BIS certification process
    - 'standard'      : user wants to find or know about a standard
    - 'hallmarking'   : user wants to know about gold hallmarking
    - 'qco'           : user wants to know about Quality Control Orders
    - 'document'      : user wants to know about documents required
    - 'general'       : general BIS information query
    """
    q = question.lower()

    if any(kw in q for kw in ['certif', 'isi mark', 'standard mark', 'license', 'manak online', 'fmcs', 'foreign manufacturer']):
        return 'certification'
    elif any(kw in q for kw in ['hallmark', 'gold', 'jewel', 'silver', 'huid']):
        return 'hallmarking'
    elif any(kw in q for kw in ['quality control order', 'qco', 'mandatory']):
        return 'qco'
    elif any(kw in q for kw in ['document', 'papers', 'required', 'submit']):
        return 'document'
    elif any(kw in q for kw in ['standard', 'is number', 'is ', 'find standard', 'applicable standard']):
        return 'standard'
    else:
        return 'general'

# ─────────────────────────────────────────────
# KNOWLEDGE BASE SEARCH
# ─────────────────────────────────────────────

def compute_score(text, query_words):
    """Compute a relevance score for a piece of text against query words."""
    text_lower = text.lower()
    score = 0
    for word in query_words:
        if len(word) > 2:  # ignore very short words
            count = text_lower.count(word)
            score += count
    return score

def search_knowledge_base(question, intent, top_k=3):
    """
    Search the knowledge base for relevant information.
    
    Args:
        question: The user's question string
        intent: Detected intent category
        top_k: Maximum number of results to return
    
    Returns:
        List of relevant knowledge base entries (dicts)
    """
    # Tokenize the question into words
    query_words = re.findall(r'\w+', question.lower())
    # Remove common stopwords
    stopwords = {'what', 'is', 'are', 'how', 'do', 'i', 'the', 'a', 'an', 'for', 'to', 'of', 'in', 'and', 'or', 'my'}
    query_words = [w for w in query_words if w not in stopwords]

    results = []

    # Search sections based on intent priority
    sections_to_search = []
    if intent == 'certification':
        sections_to_search = ['certification', 'general', 'faqs']
    elif intent == 'standard':
        sections_to_search = ['general', 'standards', 'faqs']
    elif intent == 'hallmarking':
        sections_to_search = ['faqs', 'general', 'certification']
    elif intent == 'qco':
        sections_to_search = ['general', 'certification', 'faqs']
    elif intent == 'document':
        sections_to_search = ['certification', 'faqs', 'general']
    else:
        sections_to_search = ['general', 'faqs', 'certification', 'standards']

    scored_results = []

    for section_name in sections_to_search:
        section = KNOWLEDGE_BASE.get(section_name, [])
        if not isinstance(section, list):
            continue
        for entry in section:
            # Score based on keywords field
            keywords = entry.get('keywords', [])
            keyword_score = sum(1 for kw in keywords if any(w in kw or kw in w for w in query_words))

            # Score based on title and content
            title = entry.get('title', '')
            content = entry.get('content', '')
            text_score = compute_score(title + ' ' + content, query_words)

            total_score = (keyword_score * 3) + text_score  # keywords weighted more

            if total_score > 0:
                scored_results.append((total_score, section_name, entry))

    # Sort by score descending
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # Return top_k unique entries
    seen_ids = set()
    for score, section, entry in scored_results:
        entry_id = entry.get('id', '')
        if entry_id not in seen_ids:
            seen_ids.add(entry_id)
            results.append(entry)
        if len(results) >= top_k:
            break

    return results

# ─────────────────────────────────────────────
# RESPONSE GENERATION
# ─────────────────────────────────────────────

def build_response_from_kb(question, entries):
    """
    Build a structured response from knowledge base entries.
    This is the fallback when no LLM API is configured.
    """
    if not entries:
        return {
            'answer': "I couldn't find sufficient verified information in the current BIS knowledge base. Please verify the latest requirements from the official BIS website at www.bis.gov.in or contact BIS directly.",
            'key_points': [],
            'sources': [{'title': 'BIS Official Website', 'url': 'https://www.bis.gov.in/'}],
            'confidence': 'low'
        }

    # Use the top entry as primary answer
    primary = entries[0]
    answer = primary.get('content', '') or primary.get('answer', '')

    # Collect key points from all entries
    key_points = []
    for entry in entries:
        kp = entry.get('key_points', [])
        key_points.extend(kp)

    # Deduplicate key_points (keep order)
    seen = set()
    unique_kp = []
    for kp in key_points:
        if kp not in seen:
            seen.add(kp)
            unique_kp.append(kp)

    # Build sources list
    sources = []
    seen_sources = set()
    for entry in entries:
        src_title = entry.get('source', '')
        src_url = entry.get('source_url', 'https://www.bis.gov.in/')
        if src_title and src_title not in seen_sources:
            seen_sources.add(src_title)
            sources.append({'title': src_title, 'url': src_url})

    return {
        'answer': answer,
        'key_points': unique_kp[:8],  # limit to 8 key points
        'sources': sources,
        'confidence': 'medium'
    }

def call_llm_api(question, context_entries):
    """
    Call an external LLM API to generate an answer.
    
    HOW TO ENABLE:
    1. For OpenAI: Add OPENAI_API_KEY=sk-xxx to your .env file
                   Run: pip install openai
    2. For Gemini: Add GEMINI_API_KEY=xxx to your .env file
                   Run: pip install google-generativeai
    
    Returns None if no API key is configured, which triggers fallback.
    """
    # Build context string from knowledge base entries
    context = "\n\n".join([
        f"Title: {e.get('title', '')}\nContent: {e.get('content', '') or e.get('answer', '')}\nSource: {e.get('source', '')} ({e.get('source_url', '')})"
        for e in context_entries
    ])

    system_prompt = """You are an AI assistant for BIS (Bureau of Indian Standards) India. 
Your role is to help users understand BIS standards, certification procedures, and compliance requirements.

IMPORTANT RULES:
- Only answer based on the provided context from BIS knowledge base
- Do NOT invent IS numbers, fees, or specific requirements not in the context
- If unsure, direct users to the official BIS website: www.bis.gov.in
- Keep answers clear, concise and professional
- Always cite sources from the provided context"""

    user_prompt = f"""Context from BIS Knowledge Base:
{context}

User Question: {question}

Please provide a helpful answer based only on the above context. If the context doesn't contain enough information, say so clearly."""

    # Try OpenAI
    if OPENAI_API_KEY:
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

    # Try Gemini
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-pro')
            full_prompt = system_prompt + "\n\n" + user_prompt
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None

    return None  # No API key configured - use fallback

# ─────────────────────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────────────────────

def get_answer(question):
    """
    Main function: Takes a user question and returns a structured answer.
    
    This is the entry point called by the Flask API endpoints.
    
    Args:
        question (str): The user's question
    
    Returns:
        dict: {
            'answer': str,          # Main answer text
            'key_points': list,     # List of key point strings
            'sources': list,        # List of {title, url} dicts
            'confidence': str,      # 'high', 'medium', or 'low'
            'mode': str             # 'llm' or 'demo'
        }
    """
    if not question or not question.strip():
        return {
            'answer': 'Please enter a question.',
            'key_points': [],
            'sources': [],
            'confidence': 'low',
            'mode': 'demo'
        }

    # Step 1: Detect intent
    intent = detect_intent(question)

    # Step 2: Search knowledge base
    relevant_entries = search_knowledge_base(question, intent, top_k=3)

    # Step 3: Try LLM API if configured
    llm_answer = call_llm_api(question, relevant_entries)

    if llm_answer:
        # LLM answer available
        sources = []
        seen = set()
        for entry in relevant_entries:
            src = entry.get('source', '')
            if src and src not in seen:
                seen.add(src)
                sources.append({'title': src, 'url': entry.get('source_url', 'https://www.bis.gov.in/')})
        return {
            'answer': llm_answer,
            'key_points': relevant_entries[0].get('key_points', [])[:6] if relevant_entries else [],
            'sources': sources,
            'confidence': 'high',
            'mode': 'llm'
        }
    else:
        # Fallback: build response from knowledge base directly
        result = build_response_from_kb(question, relevant_entries)
        result['mode'] = 'demo'
        return result


def find_standards(product_name, category='', purpose=''):
    """
    Find potentially relevant Indian Standards for a given product.
    
    Args:
        product_name (str): Name of the product
        category (str): Industry/category of the product
        purpose (str): Purpose of searching
    
    Returns:
        dict with 'standards' list and 'disclaimer' string
    """
    query = f"{product_name} {category} {purpose}".lower()
    query_words = re.findall(r'\w+', query)
    stopwords = {'the', 'a', 'an', 'for', 'to', 'of', 'in', 'and', 'or', 'my', 'our'}
    query_words = [w for w in query_words if w not in stopwords and len(w) > 2]

    standards = KNOWLEDGE_BASE.get('standards', [])
    scored = []

    for std in standards:
        score = 0
        # Score against keywords
        for kw in std.get('keywords', []):
            for word in query_words:
                if word in kw or kw in word:
                    score += 3
        # Score against applicable_products
        for prod in std.get('applicable_products', []):
            score += compute_score(prod, query_words)
        # Score against title and description
        score += compute_score(std.get('title', '') + ' ' + std.get('description', ''), query_words)

        if score > 0:
            scored.append((score, std))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_standards = [s for _, s in scored[:5]]  # return top 5

    return {
        'standards': top_standards,
        'disclaimer': 'These are potentially relevant standards identified from the demo knowledge base. The actual applicable standard must be verified from the official BIS website (www.bis.gov.in) or by consulting BIS directly.',
        'query': {'product': product_name, 'category': category, 'purpose': purpose}
    }


def generate_compliance_checklist(product_name, business_type):
    """
    Generate a basic compliance checklist for a product.
    
    Args:
        product_name (str): Name of the product
        business_type (str): 'manufacturer', 'importer', 'seller', or 'other'
    
    Returns:
        dict with 'checklist' list and 'note' string
    """
    base_checklist = [
        {"id": 1, "item": f"Identify the applicable Indian Standard (IS) for {product_name}", "category": "Standards", "completed": False},
        {"id": 2, "item": "Check whether BIS certification is compulsory (via Quality Control Order / QCO)", "category": "Certification", "completed": False},
        {"id": 3, "item": "Review the full text of the applicable Indian Standard", "category": "Standards", "completed": False},
        {"id": 4, "item": "Understand the product requirements specified in the standard", "category": "Product", "completed": False},
        {"id": 5, "item": "Identify the testing requirements and recognized test laboratories", "category": "Testing", "completed": False},
        {"id": 6, "item": "Prepare all required business and product documentation", "category": "Documentation", "completed": False},
        {"id": 7, "item": "Verify the latest BIS requirements from www.bis.gov.in or Manak Online portal", "category": "Verification", "completed": False},
    ]

    # Add role-specific items
    if business_type.lower() == 'manufacturer':
        base_checklist.extend([
            {"id": 8, "item": "Register on the BIS Manak Online portal (www.manakonline.in)", "category": "Application", "completed": False},
            {"id": 9, "item": "Submit application for BIS Standard Mark License", "category": "Application", "completed": False},
            {"id": 10, "item": "Get product samples tested at a BIS recognized / NABL accredited laboratory", "category": "Testing", "completed": False},
            {"id": 11, "item": "Prepare for possible BIS factory inspection", "category": "Inspection", "completed": False},
        ])
    elif business_type.lower() == 'importer':
        base_checklist.extend([
            {"id": 8, "item": "Check if the foreign manufacturer has BIS certification (FMCS)", "category": "Certification", "completed": False},
            {"id": 9, "item": "Verify BIS compliance of imported products before sale", "category": "Compliance", "completed": False},
            {"id": 10, "item": "Ensure imported products bear valid BIS Standard Mark (if QCO applies)", "category": "Marking", "completed": False},
        ])
    elif business_type.lower() == 'seller':
        base_checklist.extend([
            {"id": 8, "item": "Verify that products sourced from manufacturers/importers have valid BIS certification", "category": "Verification", "completed": False},
            {"id": 9, "item": "Do not sell products under QCO without valid BIS Standard Mark", "category": "Compliance", "completed": False},
        ])

    return {
        'checklist': base_checklist,
        'product': product_name,
        'business_type': business_type,
        'note': 'DEMO DATA: This checklist is for informational purposes only. Always verify current requirements from the official BIS website (www.bis.gov.in) or by contacting BIS directly.'
    }
