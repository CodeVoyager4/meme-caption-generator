from flask import Flask, render_template, request, send_file, redirect, url_for
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

app = Flask(__name__)

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
    conn.close()

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

@app.route('/saves/<filename>')
def serve_saved_image(filename):
    return send_file(SAVES_DIR / filename)

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
            SELECT * FROM memes 
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

if __name__ == '__main__':
    app.run(debug=True) 