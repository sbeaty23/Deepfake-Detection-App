## Prerequisites

Python 3.10 or newer
Node.js and npm

# Backend Setup

Install the backend dependencies in a virtual environment:

> cd backend
and
> pip install -r requirements.txt

Start the API:

> fastapi dev main.py

The API should run at http://127.0.0.1:8000.

Interactive API documentation is available at http://127.0.0.1:8000/docs.

# Running the Application

Keep both the frontend and backend running in separate terminals.

The React frontend sends uploaded images to the FastAPI backend for validation. The current backend validates image uploads; AI-generated image detection still requires a detection model to be integrated.

