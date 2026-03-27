# The Personal Intelligence System

## Quick Start

### Setup

```bash
# Create Virtual Environment 
python -m venv env

# Activate Enviroment for windows
env\Scripts\activate

# Activate Enviroment for mac
source env/bin/activate


# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### Testing
1. Open browser: `http://localhost:5000`
2. You'll see: ✓ Frontend is running
3. Click "Test API" button
4. You'll see: ✓ API is called + Backend is running message

### What's Tested
- ✓ Backend is running (Flask server on port 5000)
- ✓ Frontend is running (HTML served from `/`)
- ✓ API is called (endpoint `/api/test` returns JSON)