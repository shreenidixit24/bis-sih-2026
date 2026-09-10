"""
database.py - Knowledge Base Database Interface

This module provides database-like interface functions for
accessing the BIS knowledge base. In this demo version,
it reads from JSON files. In production, this could be
replaced with a real vector database like ChromaDB, Pinecone,
or FAISS for better semantic search.

NOTE: For SIH 2026 hackathon demo.
"""

import json
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / 'data' / 'bis_data.json'


def load_all_data():
    """Load the entire knowledge base."""
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}


def get_all_standards():
    """Return all standards from the knowledge base."""
    data = load_all_data()
    return data.get('standards', [])


def get_standard_by_category(category):
    """Get standards filtered by category."""
    standards = get_all_standards()
    return [
        s for s in standards
        if category.lower() in s.get('category', '').lower()
    ]


def get_certification_info():
    """Return all certification-related knowledge entries."""
    data = load_all_data()
    return data.get('certification', [])


def get_general_info():
    """Return all general BIS information entries."""
    data = load_all_data()
    return data.get('general', [])


def get_faqs():
    """Return all FAQ entries."""
    data = load_all_data()
    return data.get('faqs', [])


def get_sample_questions():
    """Return sample questions for the demo UI."""
    data = load_all_data()
    return data.get('sample_questions', [])


def search_standards(query):
    """
    Simple keyword-based search across all standards.
    
    Args:
        query (str): Search query string
    
    Returns:
        List of matching standard entries
    """
    standards = get_all_standards()
    query_lower = query.lower()
    results = []
    for std in standards:
        searchable = (
            std.get('title', '') + ' ' +
            std.get('description', '') + ' ' +
            std.get('category', '') + ' ' +
            ' '.join(std.get('keywords', [])) + ' ' +
            ' '.join(std.get('applicable_products', []))
        ).lower()
        if any(word in searchable for word in query_lower.split() if len(word) > 2):
            results.append(std)
    return results
