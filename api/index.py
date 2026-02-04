"""Vercel serverless function entry point."""

from server import app

# Export the FastAPI app for Vercel
app = app
