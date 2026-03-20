"""
setup.py
Run this ONCE before starting the bot to embed all knowledge base documents.
Usage: python setup.py
"""

from rag.embedder import embed_documents

if __name__ == "__main__":
    print("=" * 50)
    print("AVIVO Bot — Knowledge Base Setup")
    print("=" * 50)
    embed_documents(force=False)
    print("\n✅ Setup complete. You can now run: python app.py")
