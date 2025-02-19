from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify, send_from_directory
from openai import OpenAI, AzureOpenAI
import os
import json
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import textwrap
import uuid
import shutil
from pathlib import Path
import praw
from functools import lru_cache
import sqlite3
import secrets
import string

app = Flask(__name__, static_folder='static')

# Add this after app initialization
SAVES_DIR = Path("saves")
SAVES_DIR.mkdir(exist_ok=True)

# Add after SAVES_DIR initialization
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Create a simple placeholder image if it doesn't exist
if not (STATIC_DIR / "placeholder.png").exists():
    img = Image.new('RGB', (200, 200), color='#f0f0f0')
    img.save(STATIC_DIR / "placeholder.png")

# Add after app initialization
CACHE_TIMEOUT = timedelta(hours=1)
meme_cache = {}

def get_cached_image(key, creator_func):
    """
    Get image from cache or create it if not exists/expired
    key: unique identifier for the image
    creator_func: function to create the image if not in cache
    """
    now = datetime.now()
    if key in meme_cache:
        timestamp, image = meme_cache[key]
        if now - timestamp < CACHE_TIMEOUT:
            return image
            
    # Create new image
    image = creator_func()
    meme_cache[key] = (now, image)
    return image

@lru_cache(maxsize=100)
def load_analyses_cached():
    """Cached version of load_analyses"""
    return load_analyses()

def invalidate_cache():
    """Invalidate the analyses cache when new data is saved"""
    load_analyses_cached.cache_clear()

def init_db():
    conn = sqlite3.connect('memes.db')
    c = conn.cursor()
    
    # Create memes table
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
            captioned_filename TEXT,
            upvotes INTEGER DEFAULT 0,
            downvotes INTEGER DEFAULT 0
        )
    ''')
    
    # Create shortened URLs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS short_urls (
            short_id TEXT PRIMARY KEY,
            timestamp TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (timestamp) REFERENCES memes(timestamp)
        )
    ''')
    conn.commit()
    return conn

def get_db():
    conn = sqlite3.connect('memes.db')
    conn.row_factory = sqlite3.Row
    return conn

def save_analysis(image_url, api_response, description, caption, messages, completion, model_choice, captioned_filename):
    conn = get_db()
    try:
        data = {
            "model": completion.model,
            "created": completion.created,
            "response_ms": completion.response_ms if hasattr(completion, 'response_ms') else None
        }
        
        conn.execute('''
            INSERT INTO memes 
            (timestamp, image_url, raw_api_response, description, caption, 
             messages_sent, model_choice, completion_data, captioned_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            image_url,
            api_response,
            description,
            caption,
            json.dumps(messages),
            model_choice,
            json.dumps(data),
            captioned_filename
        ))
        conn.commit()
    finally:
        conn.close()

def parse_response(response):
    # Split the response into description and caption
    try:
        description = response.split("DESCRIPTION:")[1].split("CAPTION:")[0].strip()
        caption = response.split("CAPTION:")[1].strip()
        return description, caption
    except:
        return response, ""

def create_captioned_image(image_url, caption):
    # Download the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    # Calculate dimensions
    margin = 100
    caption_height = 150
    new_height = img.height + caption_height
    
    # Create new image with white background
    new_img = Image.new('RGB', (img.width, new_height), 'white')
    new_img.paste(img, (0, 0))
    
    # Add caption
    draw = ImageDraw.Draw(new_img)
    
    # Try to load Arial font
    try:
        base_font_size = 40
        font = ImageFont.truetype("arial.ttf", base_font_size)
    except:
        font = ImageFont.load_default()
        
    # Calculate optimal font size
    max_width = img.width - (2 * margin)
    font_size = base_font_size
    
    while font_size > 10:  # Minimum readable font size
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
            
        # Wrap text and calculate total height
        wrapped_text = textwrap.fill(caption, width=int(max_width / (font_size * 0.6)))
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Check if text fits within constraints
        if text_width <= max_width and text_height <= caption_height - 20:
            break
            
        font_size -= 2
    
    # Final text wrapping with optimal font size
    wrapped_text = textwrap.fill(caption, width=int(max_width / (font_size * 0.6)))
    
    # Calculate text position for center alignment
    text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (img.width - text_width) // 2
    text_y = img.height + ((caption_height - text_height) // 2)
    
    # Draw text
    draw.text((text_x, text_y), wrapped_text, font=font, fill='black')
    
    # Modified save logic
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    permanent_filename = f"meme_{uuid.uuid4()}.jpg"
    
    # Save temporary file first
    new_img.save(temp_filename)
    
    # Move to permanent location
    shutil.move(temp_filename, SAVES_DIR / permanent_filename)
    
    return permanent_filename

def analyze_image(image_url, model_choice="openai"):
    # Check for API keys in environment
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if model_choice == "xai" and not XAI_API_KEY:
        return "Error: XAI_API_KEY not found in environment variables. Please set it first.", None
    elif model_choice == "openai" and not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not found in environment variables. Please set it first.", None
    
    try:
        if model_choice == "xai":
            client = OpenAI(
                api_key=XAI_API_KEY,
                base_url="https://api.x.ai/v1",
            )
            model = "grok-2-vision-latest"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": "First describe what's in this image, then create a humorous Far Side-style caption for it. Format the response as: DESCRIPTION: (description) CAPTION: (funny caption)",
                        },
                    ],
                },
            ]
        else:  # OpenAI
            client = OpenAI()  # Uses OPENAI_API_KEY from environment
            model = "gpt-4o-mini"  # Updated model name
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "First describe what's in this image, then create a humorous Far Side-style caption for it. Format the response as: DESCRIPTION: (description) CAPTION: (funny caption)"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500
        )
        
        response = completion.choices[0].message.content
        
        # Extract description and caption with better error handling
        try:
            if "DESCRIPTION:" not in response:
                description = "No description provided by the model"
            else:
                description = response.split("CAPTION:")[0].replace("DESCRIPTION:", "").strip()
            
            if "CAPTION:" not in response:
                # Try to find anything in quotes as a caption
                import re
                quotes = re.findall(r'"([^"]*)"', response)
                if quotes:
                    caption = quotes[0]
                else:
                    # Take the last sentence if no quotes found
                    sentences = response.split('.')
                    caption = sentences[-1].strip()
            else:
                caption = response.split("CAPTION:")[1].strip()
            
            # Clean up caption if needed
            caption = caption.strip('"')  # Remove extra quotes
            if not caption.startswith('"'):
                caption = f'"{caption}"'  # Ensure caption is in quotes
                
        except Exception as e:
            return f"Error: Failed to parse model response - {str(e)}", None
        
        # Create the captioned image
        captioned_image = create_captioned_image(image_url, caption)
        
        # Save the analysis
        save_analysis(
            image_url=image_url,
            api_response=response,
            description=description,
            caption=caption,
            messages=messages,
            completion=completion,
            model_choice=model_choice,
            captioned_filename=captioned_image
        )
        
        return response, captioned_image
    except Exception as e:
        return f"Error: {str(e)}", None

def get_gallery_images():
    try:
        # Imgur gallery tags that tend to have good images
        tags = ['pics', 'photography', 'interesting', 'nature']
        images = []
        
        for tag in tags:
            try:
                url = f"https://api.imgur.com/3/gallery/t/{tag}/top/day"
                headers = {'Authorization': 'Client-ID 6db4f0f2e8d9567'}  # This is a public client ID
                
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('data', {}).get('items', [])
                    
                    for item in items:
                        if not item.get('is_album') and not item.get('nsfw', True):
                            image_url = item.get('link', '')
                            if image_url.endswith(('.jpg', '.jpeg', '.png')):
                                images.append({
                                    'url': image_url,
                                    'title': item.get('title', 'Untitled'),
                                    'subreddit': tag  # Using tag as category
                                })
            except Exception as e:
                print(f"Error fetching from {tag}: {str(e)}")
                continue
                
        return images[:20]  # Return max 20 images
        
    except Exception as e:
        print(f"Error in get_gallery_images: {str(e)}")
        return []

def get_recent_memes(limit=12):
    conn = get_db()
    try:
        memes = conn.execute('''
            SELECT timestamp, caption, captioned_filename, image_url 
            FROM memes 
            WHERE captioned_filename IS NOT NULL
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,)).fetchall()
        
        return [{
            'image': row['captioned_filename'],
            'caption': row['caption'],
            'timestamp': row['timestamp'],
            'url': url_for('view_analysis', timestamp=row['timestamp'])
        } for row in memes if row['captioned_filename']]
    finally:
        conn.close()

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    error = None
    captioned_image = None
    gallery_images = []
    
    try:
        gallery_images = get_gallery_images()
    except Exception as e:
        error = f"Error loading gallery images: {str(e)}"
    
    if request.method == 'POST':
        image_url = request.form.get('image_url')
        model_choice = request.form.get('model_choice', 'xai')
        if image_url:
            result, captioned_image = analyze_image(image_url, model_choice)
            if result.startswith("Error:"):
                error = result
                result = None
    
    # Get recent memes
    recent_memes = get_recent_memes()
    
    return render_template('index.html', 
                         result=result, 
                         error=error, 
                         captioned_image=captioned_image,
                         reddit_images=gallery_images,
                         recent_memes=recent_memes)

@app.route('/saves/<path:filename>')
def serve_saved_image(filename):
    if not filename:
        return "No filename provided", 404
    return send_from_directory('saves', filename, as_attachment=False)

def get_analysis(timestamp):
    conn = get_db()
    try:
        row = conn.execute('SELECT * FROM memes WHERE timestamp = ?', 
                          (timestamp,)).fetchone()
        if row:
            data = dict(row)
            data['messages_sent'] = json.loads(data['messages_sent'])
            data['completion_data'] = json.loads(data['completion_data'])
            return data
        return None
    finally:
        conn.close()

@app.route('/history')
def history():
    conn = get_db()
    try:
        analyses = conn.execute('''
            SELECT *, upvotes, downvotes FROM memes 
            ORDER BY timestamp DESC
        ''').fetchall()
        
        grouped_analyses = {}
        for analysis in analyses:
            date = analysis['timestamp'].split('T')[0]
            model = analysis['model_choice']
            
            if date not in grouped_analyses:
                grouped_analyses[date] = {'xai': [], 'openai': []}
            
            analysis_dict = dict(analysis)
            analysis_dict['completion_data'] = json.loads(analysis['completion_data'])
            analysis_dict['messages_sent'] = json.loads(analysis['messages_sent'])
            analysis_dict['captioned_image'] = analysis['captioned_filename']
            
            grouped_analyses[date][model].append(analysis_dict)
            
        return render_template('history.html', grouped_analyses=grouped_analyses)
    finally:
        conn.close()

@app.route('/view/<timestamp>', methods=['GET', 'POST'])
def view_analysis(timestamp):
    if request.method == 'POST':
        image_url = request.form.get('image_url')
        if image_url:
            result, captioned_image = analyze_image(image_url)
            if result.startswith("Error:"):
                error = result
                return render_template('view.html', error=error)
            return redirect(url_for('view_analysis', timestamp=timestamp))
    
    analysis = get_analysis(timestamp)
    if analysis:
        # Recreate the captioned image
        captioned_image = create_captioned_image(analysis['image_url'], analysis['caption'])
        return render_template('view.html', analysis=analysis, captioned_image=captioned_image)
    return "Analysis not found", 404

# Modify the cleanup function
def cleanup_temp_files():
    for file in os.listdir():
        if file.startswith('temp_') and file.endswith('.jpg'):
            try:
                os.remove(file)
            except:
                pass

# Clean up temporary files when the application stops
import atexit
atexit.register(cleanup_temp_files)

# Call init_db() when app starts
init_db()

@app.route('/stats')
def stats():
    conn = get_db()
    try:
        entries = conn.execute('SELECT * FROM memes ORDER BY timestamp DESC').fetchall()
        
        entries_processed = []
        for entry in entries:
            entry_dict = dict(entry)
            entry_dict['completion_data'] = json.loads(entry['completion_data'])
            entries_processed.append(entry_dict)
        
        stats = {
            'total_memes': len(entries),
            'grok_count': len([e for e in entries if e['model_choice'] == 'xai']),
            'gpt4_count': len([e for e in entries if e['model_choice'] == 'openai'])
        }
        
        return render_template('admin.html', entries=entries_processed, stats=stats)
    finally:
        conn.close()

@app.route('/admin/delete/<timestamp>', methods=['POST'])
def delete_entry(timestamp):
    conn = get_db()
    try:
        # Get the entry first to delete associated image
        entry = conn.execute('SELECT captioned_filename FROM memes WHERE timestamp = ?', 
                           (timestamp,)).fetchone()
        
        if entry and entry['captioned_filename']:
            image_path = os.path.join('saves', entry['captioned_filename'])
            if os.path.exists(image_path):
                os.remove(image_path)
        
        conn.execute('DELETE FROM memes WHERE timestamp = ?', (timestamp,))
        conn.commit()
        return jsonify({'success': True})
    finally:
        conn.close()

@app.route('/vote/<timestamp>/<vote_type>', methods=['POST'])
def vote(timestamp, vote_type):
    if vote_type not in ['up', 'down']:
        return 'Invalid vote type', 400
        
    conn = get_db()
    try:
        if vote_type == 'up':
            conn.execute('UPDATE memes SET upvotes = upvotes + 1 WHERE timestamp = ?', 
                        (timestamp,))
        else:
            conn.execute('UPDATE memes SET downvotes = downvotes + 1 WHERE timestamp = ?', 
                        (timestamp,))
        conn.commit()
        
        # Get updated counts
        result = conn.execute('''
            SELECT upvotes, downvotes 
            FROM memes 
            WHERE timestamp = ?
        ''', (timestamp,)).fetchone()
        
        return jsonify({
            'upvotes': result['upvotes'],
            'downvotes': result['downvotes']
        })
    finally:
        conn.close()

@app.route('/compare')
def compare():
    conn = get_db()
    try:
        # Get model performance stats
        stats = conn.execute('''
            SELECT 
                model_choice,
                AVG(CAST(json_extract(completion_data, '$.response_ms') AS INTEGER)) as avg_response,
                AVG(LENGTH(caption)) as avg_length,
                COUNT(*) as total_generations,
                AVG(upvotes) as avg_upvotes
            FROM memes 
            GROUP BY model_choice
        ''').fetchall()
        
        model_stats = {row['model_choice']: dict(row) for row in stats}
        
        # Get pairs for side-by-side comparison
        pairs = conn.execute('''
            SELECT 
                a1.image_url,
                a1.caption as grok_caption,
                a1.captioned_filename as grok_image,
                a1.completion_data as grok_completion_data,
                a2.caption as gpt_caption,
                a2.captioned_filename as gpt_image,
                a2.completion_data as gpt_completion_data
            FROM memes a1
            JOIN memes a2 ON a1.image_url = a2.image_url 
            AND a1.model_choice = 'xai' 
            AND a2.model_choice = 'openai'
            ORDER BY a1.timestamp DESC
        ''').fetchall()
        
        comparison_pairs = []
        for pair in pairs:
            comparison_pairs.append({
                'image_url': pair['image_url'],
                'grok': {
                    'caption': pair['grok_caption'],
                    'captioned_image': pair['grok_image'],
                    'completion_data': json.loads(pair['grok_completion_data'])
                },
                'gpt': {
                    'caption': pair['gpt_caption'],
                    'captioned_image': pair['gpt_image'],
                    'completion_data': json.loads(pair['gpt_completion_data'])
                }
            })
            
        return render_template('compare.html', 
                             pairs=comparison_pairs,
                             model_stats=model_stats)
    finally:
        conn.close()

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def generate_short_id():
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(7))

@app.route('/s/<short_id>')
def share_view(short_id):
    conn = get_db()
    try:
        result = conn.execute('''
            SELECT m.* FROM memes m
            JOIN short_urls s ON m.timestamp = s.timestamp
            WHERE s.short_id = ?
        ''', (short_id,)).fetchone()
        
        if not result:
            return "Meme not found", 404
            
        analysis = dict(result)
        analysis['completion_data'] = json.loads(analysis['completion_data'])
        return render_template('view.html', 
                             analysis=analysis,
                             captioned_image=analysis['captioned_filename'])
    finally:
        conn.close()

@app.route('/share/<timestamp>')
def create_share_url(timestamp):
    conn = get_db()
    try:
        # Check if short URL already exists
        existing = conn.execute('''
            SELECT short_id FROM short_urls WHERE timestamp = ?
        ''', (timestamp,)).fetchone()
        
        if existing:
            short_id = existing[0]
        else:
            # Create new short URL
            short_id = generate_short_id()
            conn.execute('''
                INSERT INTO short_urls (short_id, timestamp)
                VALUES (?, ?)
            ''', (short_id, timestamp))
            conn.commit()
            
        share_url = url_for('share_view', short_id=short_id, _external=True)
        return {'share_url': share_url}
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True) 