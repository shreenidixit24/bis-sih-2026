"""
app.py - Main Flask Application for BIS AI Assistant

Run this file to start the web server:
    python app.py

Then open your browser at: http://localhost:5000

NOTE: This is a demo application for SIH 2026 hackathon.
Do not use in production without proper security review.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from pathlib import Path

# Import our backend modules
from backend.chatbot import process_chat_message
from backend.rag import find_standards, generate_compliance_checklist
from backend.database import get_sample_questions, get_all_standards

# ─────────────────────────────────────────────
# FLASK APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow frontend to call API from different origins

# Maximum upload size: 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Folder to save uploaded documents
UPLOAD_FOLDER = Path(__file__).parent / 'documents'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Allowed file extensions for document upload
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
# PAGE ROUTES (HTML pages)
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Home page."""
    sample_questions = get_sample_questions()
    return render_template('index.html', sample_questions=sample_questions)


@app.route('/assistant')
def assistant():
    """AI Assistant chatbot page."""
    sample_questions = get_sample_questions()
    return render_template('assistant.html', sample_questions=sample_questions)


@app.route('/standards')
def standards():
    """Find Standards page."""
    all_standards = get_all_standards()
    # Get unique categories for the dropdown
    categories = sorted(set(s.get('category', 'General') for s in all_standards))
    return render_template('standards.html', categories=categories)


@app.route('/certification')
def certification():
    """Certification Guide page."""
    return render_template('certification.html')


@app.route('/documents')
def documents():
    """Document Assistant page."""
    return render_template('documents.html')


@app.route('/compliance')
def compliance():
    """Compliance Checklist page."""
    return render_template('compliance.html')


# ─────────────────────────────────────────────
# API ROUTES (JSON endpoints)
# ─────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    Chatbot API endpoint.
    
    Request body (JSON):
        { "question": "What is BIS certification?" }
    
    Response (JSON):
        {
            "success": true,
            "answer": "...",
            "key_points": [...],
            "sources": [{"title": "...", "url": "..."}],
            "confidence": "medium",
            "mode": "demo"
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Invalid request. Please send JSON data.'}), 400

    question = data.get('question', '').strip()
    if not question:
        return jsonify({'success': False, 'error': 'Please enter a question.'}), 400

    result = process_chat_message(question)
    return jsonify(result)


@app.route('/api/find-standard', methods=['POST'])
def api_find_standard():
    """
    Find Standard API endpoint.
    
    Request body (JSON):
        {
            "product": "Pressure Cooker",
            "category": "Kitchen Appliances",
            "purpose": "Manufacturing"
        }
    
    Response (JSON):
        {
            "success": true,
            "standards": [...],
            "disclaimer": "..."
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Invalid request.'}), 400

    product = data.get('product', '').strip()
    category = data.get('category', '').strip()
    purpose = data.get('purpose', '').strip()

    if not product:
        return jsonify({'success': False, 'error': 'Please enter a product name.'}), 400

    result = find_standards(product, category, purpose)
    return jsonify({'success': True, **result})


@app.route('/api/certification', methods=['POST'])
def api_certification():
    """
    Certification information API endpoint.
    
    Request body (JSON):
        { "question": "What are the steps for BIS certification?" }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Invalid request.'}), 400

    question = data.get('question', '').strip()
    if not question:
        # Return default certification steps
        question = "How do I get BIS certification?"

    result = process_chat_message(question)
    return jsonify(result)


@app.route('/api/document', methods=['POST'])
def api_document():
    """
    Document upload and analysis API endpoint.
    Supports PDF upload and question answering about the document.
    
    For demo: uses mock data to show document analysis functionality.
    For production: integrate with a real PDF parser + vector search.
    """
    # Check if file is in request
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected.'}), 400
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Please upload a valid PDF document.'}), 400

        # Save the file (demo - in production, extract text and index it)
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # In demo mode, return mock document analysis
        return jsonify({
            'success': True,
            'filename': filename,
            'pages': 'N/A (Demo Mode)',
            'status': 'Document uploaded. Demo mode: using sample BIS document data for Q&A.',
            'mode': 'demo',
            'note': 'In production mode, the system will extract and index the actual document content using RAG.'
        })

    # Handle document Q&A
    data = request.get_json()
    if data:
        question = data.get('question', '').strip()
        if question:
            # In demo, answer from our knowledge base
            result = process_chat_message(question)
            # Add demo document context
            result['relevant_section'] = 'Clause 4.1 - General Requirements (Demo)'
            result['page_number'] = 'Page 3 (Demo)'
            result['source_document'] = data.get('filename', 'Uploaded Document')
            return jsonify(result)

    return jsonify({'success': False, 'error': 'Invalid request.'}), 400


@app.route('/api/compliance', methods=['POST'])
def api_compliance():
    """
    Compliance Checklist API endpoint.
    
    Request body (JSON):
        {
            "product": "Pressure Cooker",
            "business_type": "manufacturer"
        }
    
    Response (JSON):
        {
            "success": true,
            "checklist": [...],
            "product": "...",
            "business_type": "...",
            "note": "..."
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Invalid request.'}), 400

    product = data.get('product', '').strip()
    business_type = data.get('business_type', 'manufacturer').strip()

    if not product:
        return jsonify({'success': False, 'error': 'Please enter a product name.'}), 400

    result = generate_compliance_checklist(product, business_type)
    return jsonify({'success': True, **result})


# ─────────────────────────────────────────────
# STATIC FILES (for uploaded documents)
# ─────────────────────────────────────────────

@app.route('/documents/<filename>')
def uploaded_file(filename):
    """Serve uploaded documents."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 10 MB.'}), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error. Please try again.'}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("="*60)
    print("  BIS AI Assistant - SIH 2026 Demo")
    print("="*60)
    print("  Starting server...")
    print("  Open your browser at: http://localhost:5000")
    print("="*60)
    # debug=True auto-reloads on code changes (development only)
    app.run(debug=True, host='0.0.0.0', port=5000)
