"""
chatbot.py - Chatbot Request Handler

This module processes chatbot requests and formats responses
for the Flask API. It acts as a bridge between the API layer
and the RAG engine.

NOTE: For SIH 2026 hackathon demo.
"""

from backend.rag import get_answer


def process_chat_message(question):
    """
    Process a user's chat message and return a formatted response.
    
    Args:
        question (str): The user's question
    
    Returns:
        dict: Formatted response ready for JSON API output
    """
    if not question or not question.strip():
        return {
            'success': False,
            'error': 'Please enter a question.',
            'answer': '',
            'key_points': [],
            'sources': []
        }

    # Get answer from RAG engine
    result = get_answer(question)

    return {
        'success': True,
        'question': question,
        'answer': result.get('answer', ''),
        'key_points': result.get('key_points', []),
        'sources': result.get('sources', []),
        'confidence': result.get('confidence', 'medium'),
        'mode': result.get('mode', 'demo')
    }
