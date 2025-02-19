import json
import sqlite3
from pathlib import Path

def init_db():
    conn = sqlite3.connect('memes.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS memes (
            timestamp TEXT PRIMARY KEY,
            image_url TEXT,
            raw_api_response TEXT,
            description TEXT,
            caption TEXT,
            messages_sent TEXT,
            model_choice TEXT,
            completion_data TEXT,
            captioned_filename TEXT
        )
    ''')
    conn.commit()
    return conn

def migrate_data():
    # Initialize database
    conn = init_db()
    c = conn.cursor()
    
    # Load existing JSON data
    try:
        with open('image_analyses.json', 'r', encoding='utf-8') as f:
            analyses = json.load(f)
    except FileNotFoundError:
        print("No existing analyses found.")
        return
    
    # Migrate each analysis
    for analysis in analyses:
        try:
            # Extract data with defaults
            timestamp = analysis.get('timestamp', '')
            image_url = analysis.get('image_url', '')
            raw_api_response = analysis.get('raw_api_response', '')
            description = analysis.get('description', '')
            caption = analysis.get('caption', '')
            messages_sent = json.dumps(analysis.get('messages_sent', []))
            model_choice = analysis.get('model_choice', 'openai')
            completion_data = json.dumps(analysis.get('completion_data', {}))
            
            # Look for captioned image in saves directory
            saves_dir = Path('saves')
            possible_filename = f"meme_{timestamp.replace(':', '_')}.jpg"
            if (saves_dir / possible_filename).exists():
                captioned_filename = possible_filename
            else:
                # Try to find any matching file
                matching_files = list(saves_dir.glob(f"meme_*_{caption[:30]}*.jpg"))
                captioned_filename = matching_files[0].name if matching_files else None
            
            # Insert into database
            c.execute('''
                INSERT OR REPLACE INTO memes 
                (timestamp, image_url, raw_api_response, description, caption, 
                 messages_sent, model_choice, completion_data, captioned_filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, image_url, raw_api_response, description, caption,
                messages_sent, model_choice, completion_data, captioned_filename
            ))
            
            print(f"Migrated analysis from {timestamp}")
            
        except Exception as e:
            print(f"Error migrating analysis {timestamp}: {str(e)}")
            continue
    
    conn.commit()
    conn.close()
    print("Migration complete!")

if __name__ == "__main__":
    migrate_data() 