from flask import Flask, request, render_template, redirect, url_for
import sqlite3
import os
from werkzeug.utils import secure_filename
import ollama
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import time
from typing import Optional, List, Dict, Any
from flask import Response
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///study.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Database setup
db = SQLAlchemy(app)

class Question(db.Model):
    """
    Represents a question and its AI-generated answer.
    
    Attributes:
        id: Primary key
        question_text: The question text
        answer_text: The generated answer
        date_asked: When the question was asked
        material_type: Type of material (flashcard, quiz, etc)
        source: Source of the question (topic, notes, etc)
    """
    __tablename__ = 'questions'
    
    id = db.Column(db.Integer, primary_key=True)
    question_text = db.Column(db.String(500), nullable=False)
    answer_text = db.Column(db.Text, nullable=False)  # Changed to Text for longer answers
    date_asked = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    material_type = db.Column(db.String(50), nullable=False)
    source = db.Column(db.String(50), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=True)
    
    def __repr__(self) -> str:
        return f'<Question {self.id}: {self.question_text[:50]}...>'

class Flashcard(db.Model):
    """
    Represents a flashcard with term and definition.
    
    Attributes:
        id: Primary key
        term: The term to learn
        definition: The definition/explanation
        created_at: Creation timestamp
        mastery_level: How well the user knows this (0-5)
        source: Source of the flashcard (topic, notes, etc)
    """
    __tablename__ = 'flashcards'
    
    id = db.Column(db.Integer, primary_key=True)
    term = db.Column(db.String(200), nullable=False)
    definition = db.Column(db.Text, nullable=False)  # Changed to Text
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    mastery_level = db.Column(db.Integer, default=0)
    source = db.Column(db.String(50), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=True)
    
    def __repr__(self) -> str:
        return f'<Flashcard {self.id}: {self.term[:20]}...>'

class Subject(db.Model):
    """
    Represents a subject/topic for organizing study materials.
    """
    __tablename__ = 'subjects'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    notes = db.relationship('Note', backref='subject', lazy=True)
    flashcards = db.relationship('Flashcard', backref='subject', lazy=True)
    questions = db.relationship('Question', backref='subject', lazy=True)
    
    def __repr__(self) -> str:
        return f'<Subject {self.id}: {self.name}>'

class Note(db.Model):
    """
    Represents a study note.
    """
    __tablename__ = 'notes'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=True)
    
    def __repr__(self) -> str:
        return f'<Note {self.id}: {self.filename}>'

# Helper functions with type hints and error handling
def allowed_file(filename: str) -> bool:
    """Check if the filename has an allowed extension."""
    return (
        '.' in filename and 
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )

def save_uploaded_file(file) -> Optional[str]:
    """
    Save an uploaded file to the uploads folder.
    
    Returns:
        str: File path if successful, None otherwise
    """
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return file_path
    except Exception as e:
        app.logger.error(f"Error saving file: {str(e)}")
    return None

# Improved AI generation function
def generate_ai_response(
    prompt: str, 
    system_prompt: Optional[str] = None,
    model: str = "mistral:latest",
    max_retries: int = 3
) -> str:
    """
    Generate AI response with retry logic and error handling.
    
    Args:
        prompt: The user's prompt/question
        system_prompt: Optional system context
        model: The model to use
        max_retries: Number of retry attempts
    
    Returns:
        str: Generated response or error message
    """
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})
    
    for attempt in range(max_retries):
        try:
            response = ollama.chat(model=model, messages=messages)
            return response['message']['content']
        except Exception as e:
            if attempt == max_retries - 1:
                app.logger.error(f"AI generation failed after {max_retries} attempts: {str(e)}")
                return f"Sorry, I couldn't generate a response. Error: {str(e)}"
            time.sleep(1 * (attempt + 1))  # Exponential backoff
    
    return "Sorry, I couldn't generate a response at this time."

# Logging configuration
if not app.debug:
    if not os.path.exists('instance'):
        os.makedirs('instance')
    file_handler = RotatingFileHandler('instance/flask.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('TeacherGPT startup')

# Routes with improved error handling
@app.route('/')
def index():
    try:
        stats = {
            'questions': Question.query.count(),
            'flashcards': Flashcard.query.count(),
            'subjects': Subject.query.count()
        }
        recent_questions = Question.query.order_by(Question.date_asked.desc()).limit(5).all()
        return render_template(
            'index.html',
            stats=stats,
            recent_questions=recent_questions
        )
    except Exception as e:
        app.logger.error(f"Index route error: {str(e)}", exc_info=True)
        raise

@app.route('/upload', methods=['GET', 'POST'])
def upload() -> Response:
    """Handle file uploads for study notes."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['file']
        subject_id = request.form.get('subject_id')
        
        if not file or file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        file_path = save_uploaded_file(file)
        if not file_path:
            flash('Invalid file type')
            return redirect(request.url)
            
        # Process and save to database
        try:
            note = Note(
                filename=file.filename,
                file_path=file_path,
                subject_id=subject_id
            )
            db.session.add(note)
            db.session.commit()
            flash('Note uploaded successfully!')
            return redirect(url_for('notes'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error saving note: {str(e)}")
            flash('Error saving note')
            return redirect(request.url)
            
    subjects = Subject.query.all()
    return render_template('upload.html', subjects=subjects)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        conn = sqlite3.connect('notes.db')
        c = conn.cursor()
        c.execute("SELECT * FROM notes WHERE subject LIKE ?", ('%' + query + '%',))
        results = c.fetchall()
        conn.close()
        return render_template('search_results.html', results=results)
    return render_template('search_results.html')

@app.route('/flashcards/input', methods=['GET', 'POST'])
def flashcards():
    if request.method == 'POST':
        term = request.form['term']
        definition = request.form['definition']
        conn = sqlite3.connect('flashcards.db')
        c = conn.cursor()
        c.execute("INSERT INTO flashcards (term, definition) VALUES (?, ?)", (term, definition))
        conn.commit()
        conn.close()
        return redirect(url_for('flashcards'))
    return render_template('flashcards.html')

@app.route('/flashcards/view')
def view_flashcards():
    flashcards = Flashcard.query.order_by(Flashcard.created_at.desc()).all()
    return render_template('flashcards.html', flashcards=flashcards)

@app.route('/quiz/view')
def quiz():
    flashcards = Flashcard.query.order_by(Flashcard.created_at.desc()).limit(10).all()
    return render_template('quiz.html', flashcards=flashcards)

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    correct = 0
    total = Flashcard.query.count()
    
    for flashcard in Flashcard.query.all():
        user_answer = request.form.get(f'answer_{flashcard.id}')
        if user_answer and user_answer.lower() == flashcard.definition.lower():
            correct += 1
    
    return render_template('quiz_results.html', 
                         correct=correct, 
                         total=total,
                         percentage=int((correct/total)*100))

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if request.method == 'POST':
        question = request.form['question']
        
        # Construct prompt for Ollama model
        prompt = f"Answer the following question based on the uploaded textbooks and notes: {question}"

        # Query Ollama model with the custom prompt
        response = generate_ai_response(prompt)
        return render_template('answer.html', answer=response)
    return render_template('ask.html')

@app.route('/general_question', methods=['GET', 'POST'])
def general_question():
    if request.method == 'POST':
        question = request.form['question']
        material_type = request.form.get('material_type', 'flashcard')
        source = request.form.get('source', 'topic')
        
        start_time = time.time()
        
        # Generate prompt based on source
        if source == 'notes':
            # Get relevant notes from database
            notes = Note.query.filter(Note.filename.contains(question)).all()
            if not notes:
                return render_template('general_question.html',
                                    error="No notes found containing that topic",
                                    material_types=['flashcard', 'quiz', 'qa', 'summary'])
            
            # Create prompt using note content
            prompt = f"Create {material_type} about {question} using these notes:\n"
            for note in notes:
                try:
                    with open(note.file_path, 'r') as f:
                        prompt += f"\nNote: {note.filename}\n{f.read()}\n"
                except:
                    continue
        else:
            # Standard topic-based prompt
            prompt = question
        
        # Query Ollama model with the question
        response = generate_ai_response(prompt, model="mistral:latest")
        content = response
        
        generation_time = round(time.time() - start_time, 2)
        
        # Save to database
        new_question = Question(
            question_text=question,
            answer_text=content,
            material_type=material_type,
            source=source
        )
        db.session.add(new_question)
        
        # If flashcard, parse and save separately
        if material_type == 'flashcard':
            parts = content.split('\n\n')
            if len(parts) >= 2:
                new_flashcard = Flashcard(
                    term=parts[0].replace('Term: ', '').strip(),
                    definition=parts[1].replace('Definition: ', '').strip(),
                    source=source
                )
                db.session.add(new_flashcard)
        
        db.session.commit()
        
        return render_template('study_material.html', 
                             material={
                                 'question': question,
                                 'content': content,
                                 'type': material_type,
                                 'generation_time': generation_time,
                                 'source': source
                             },
                             material_types=['flashcard', 'quiz', 'qa', 'summary'])
    
    return render_template('general_question.html', 
                         material_types=['flashcard', 'quiz', 'qa', 'summary'])

@app.route('/study_plan', methods=['GET', 'POST'])
def study_plan():
    if request.method == 'POST':
        # User inputs study hours, deadlines, and preferences
        study_hours = request.form['study_hours']
        deadlines = request.form['deadlines']
        
        # Generate study plan prompt
        prompt = f"Create a personalized study plan for the user with {study_hours} study hours per day, " \
                 f"considering the following deadlines: {deadlines}. Include break times and exam preparation."

        # Query Ollama model for a study plan
        response = generate_ai_response(prompt)
        return render_template('study_plan.html', plan=response)
    return render_template('study_plan_form.html')

# Error handling
@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    error_msg = f"An internal error occurred"
    if app.debug:
        error_msg += f": {str(error)}\n{error.__class__.__name__}"
    app.logger.error(f"500 Error: {str(error)}", exc_info=True)
    return render_template('error.html', error=error_msg), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(403)
def forbidden_error(error):
    return render_template('error.html', error="Access forbidden"), 403

# Initialize database
def init_db():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
