import sqlite3
import hashlib
import os
from datetime import datetime
from typing import Optional

DB_PATH = "history.db"

def init_db():
    """Initializes the SQLite database and creates the audit table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generated_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            job_title TEXT,
            company TEXT,
            fit_score INTEGER,
            verdict TEXT,
            pdf_path TEXT,
            tex_path TEXT,
            tex_hash TEXT,
            language TEXT,
            doc_type TEXT,
            manually_edited BOOLEAN DEFAULT 0,
            verification_warnings TEXT
        )
    ''')
    conn.commit()
    conn.close()

def hash_file(filepath: str) -> str:
    """Returns the SHA-256 hash of a file for tamper-evident provenance tracking."""
    hasher = hashlib.sha256()
    if not os.path.exists(filepath):
        return ""
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def log_document(
    job_title: str,
    company: str,
    fit_score: int,
    verdict: str,
    pdf_path: str,
    tex_path: str,
    language: str,
    doc_type: str,
    warnings: Optional[str] = None
) -> int:
    """Logs a generated document to the local database and returns the row ID."""
    tex_hash = hash_file(tex_path)
    timestamp = datetime.now().isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO generated_documents 
        (timestamp, job_title, company, fit_score, verdict, pdf_path, tex_path, tex_hash, language, doc_type, verification_warnings)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, job_title, company, fit_score, verdict, pdf_path, tex_path, tex_hash, language, doc_type, warnings))
    
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return row_id

def mark_manually_edited(doc_id: int):
    """Flags a generated document as manually edited if the user alters the output."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE generated_documents SET manually_edited = 1 WHERE id = ?', (doc_id,))
    conn.commit()
    conn.close()