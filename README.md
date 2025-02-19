# Funalize - AI Meme Analysis and Captioning

A Flask web application that analyzes memes using AI vision models and generates humorous Far Side-style captions. The application supports multiple AI models and maintains a history of analyzed memes.

## Features

- Upload and analyze images using AI vision models
- Generate humorous Far Side-style captions
- Support for multiple AI models (OpenAI GPT-4 Vision and X.AI/Grok)
- Image caption rendering with customizable font and layout
- Persistent storage of meme analyses in SQLite database
- Browse history of analyzed memes
- Admin interface for managing meme entries
- Responsive web interface
- Caching system for improved performance

## Prerequisites

- Python 3.11 or higher
- OpenAI API key (for GPT-4 Vision)
- X.AI API key (optional, for Grok model)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/funalize.git
cd funalize
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_api_key

# Optional: For X.AI
export XAI_API_KEY=your_xai_api_key
```

5. Initialize the database:
```bash
python
>>> from app import init_db
>>> init_db()
>>> exit()
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload an image to analyze:
   - Click the upload button or drag and drop an image
   - Select your preferred AI model (OpenAI or X.AI)
   - Wait for the analysis and caption generation

4. View the results:
   - See the AI's description of the image
   - Read the generated Far Side-style caption
   - Download the captioned image
   - Browse your analysis history

## Project Structure

- `app.py` - Main Flask application and routing logic
- `templates/` - HTML templates
  - `base.html` - Base template with common layout
  - `index.html` - Home page with upload form
  - `history.html` - Analysis history page
  - `view.html` - Individual analysis view
  - `admin.html` - Admin interface
- `static/` - Static assets (CSS, JS, images)
- `saves/` - Directory for saved captioned images
- `memes.db` - SQLite database for storing analyses

## API Models

The application supports two AI vision models:

1. **OpenAI GPT-4 Vision**
   - Default model
   - Requires OPENAI_API_KEY
   - More accurate descriptions and captions

2. **X.AI Grok Vision**
   - Alternative model
   - Requires XAI_API_KEY
   - Different style of analysis and captioning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Far Side comics by Gary Larson for inspiration
- OpenAI and X.AI for their vision models
- Flask and its community for the excellent web framework
