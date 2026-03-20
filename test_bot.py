"""
test_bot.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for the AVIVO Hybrid Bot.
Run before launching the bot to verify all components are working correctly.

Usage:
    python test_bot.py             # Run all tests
    python test_bot.py --rag       # RAG pipeline only
    python test_bot.py --vision    # Vision pipeline only
    python test_bot.py --cache     # Cache utility only
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time
import argparse
import traceback
from pathlib import Path

# ── ANSI colours ──────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ PASS{RESET} — {msg}")
def fail(msg): print(f"  {RED}❌ FAIL{RESET} — {msg}")
def info(msg): print(f"  {BLUE}ℹ️  {RESET}{msg}")
def warn(msg): print(f"  {YELLOW}⚠️  {RESET}{msg}")
def header(title): print(f"\n{BOLD}{BLUE}{'─'*50}\n  {title}\n{'─'*50}{RESET}")


# ──────────────────────────────────────────────────
# TEST: Ollama connectivity
# ──────────────────────────────────────────────────
def test_ollama_connection():
    header("1. Ollama Connectivity")
    import requests

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "mistral")

    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        info(f"Ollama is running at {host}")
        info(f"Available models: {models or 'none pulled yet'}")

        if not any(model in m for m in models):
            warn(f"Model '{model}' not found. Run: ollama pull {model}")
            return False
        else:
            ok(f"Model '{model}' is available")
            return True

    except Exception as e:
        fail(f"Cannot connect to Ollama at {host}: {e}")
        info("Fix: Install Ollama from https://ollama.com and run `ollama serve`")
        return False


# ──────────────────────────────────────────────────
# TEST: Embedding model loads
# ──────────────────────────────────────────────────
def test_embedding_model():
    header("2. Embedding Model (sentence-transformers)")
    try:
        t0 = time.time()
        from rag.embedder import get_model
        model = get_model()
        elapsed = time.time() - t0

        test_sentence = "What is the refund policy?"
        embedding = model.encode(test_sentence, normalize_embeddings=True)

        ok(f"Model loaded in {elapsed:.1f}s")
        ok(f"Embedding shape: {embedding.shape}  (expected: (384,))")

        if embedding.shape == (384,):
            ok("Embedding dimensions correct")
            return True
        else:
            fail(f"Unexpected shape: {embedding.shape}")
            return False

    except Exception as e:
        fail(f"Embedding model error: {e}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────
# TEST: Document embedding pipeline
# ──────────────────────────────────────────────────
def test_embedding_pipeline():
    header("3. Document Embedding Pipeline")
    try:
        from rag.embedder import embed_documents, get_all_chunks, DB_PATH

        docs_dir = Path(__file__).parent / "rag" / "docs"
        doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))

        if not doc_files:
            fail(f"No documents found in {docs_dir}")
            return False

        info(f"Found {len(doc_files)} document(s): {[f.name for f in doc_files]}")

        embed_documents(force=False)
        ok("embed_documents() ran without error")

        chunks = get_all_chunks()
        if not chunks:
            fail("No chunks found in DB after embedding")
            return False

        ok(f"Total chunks in DB: {len(chunks)}")
        info(f"Sample chunk ({chunks[0]['doc_name']}): \"{chunks[0]['chunk'][:80]}...\"")
        return True

    except Exception as e:
        fail(f"Embedding pipeline error: {e}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────
# TEST: Full RAG pipeline (retrieval + LLM)
# ──────────────────────────────────────────────────
def test_rag_pipeline():
    header("4. Full RAG Pipeline (Retrieval + Ollama)")

    test_queries = [
        "What is the refund policy?",
        "How much does the Professional plan cost?",
        "What is cosine similarity?",
    ]

    all_passed = True
    try:
        from rag.retriever import retrieve_top_k, answer_query

        # Test retrieval only first (no LLM needed)
        info("Testing retrieval (cosine similarity)...")
        for query in test_queries[:1]:
            chunks = retrieve_top_k(query, k=2)
            if chunks:
                ok(f"Retrieved {len(chunks)} chunks for: \"{query}\"")
                info(f"  Top chunk score: {chunks[0]['score']:.3f}  |  doc: {chunks[0]['doc_name']}")
            else:
                fail(f"No chunks retrieved for: \"{query}\"")
                all_passed = False

        # Test full RAG (needs Ollama)
        info("Testing full RAG answer (requires Ollama)...")
        result = answer_query(test_queries[0])

        if result.get("answer"):
            ok(f"LLM answered successfully")
            info(f"  Answer preview: \"{result['answer'][:120]}...\"")
            info(f"  Source: {result['source']}")
        else:
            fail("LLM returned empty answer")
            all_passed = False

    except RuntimeError as e:
        warn(f"Ollama not available — skipping LLM test: {e}")
        info("Retrieval test still passed. Bot will work once Ollama is running.")
    except Exception as e:
        fail(f"RAG pipeline error: {e}")
        traceback.print_exc()
        all_passed = False

    return all_passed


# ──────────────────────────────────────────────────
# TEST: Vision pipeline
# ──────────────────────────────────────────────────
def test_vision_pipeline():
    header("5. Vision Pipeline (BLIP Image Captioning)")
    try:
        # Create a simple test image (red square) using Pillow
        from PIL import Image, ImageDraw
        import tempfile

        info("Creating test image (red square)...")
        img = Image.new("RGB", (224, 224), color=(200, 50, 50))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 174, 174], fill=(255, 200, 0))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            test_image_path = f.name

        info(f"Test image saved to: {test_image_path}")

        t0 = time.time()
        from vision.captioner import describe_image
        result = describe_image(test_image_path)
        elapsed = time.time() - t0

        ok(f"describe_image() completed in {elapsed:.1f}s")
        ok(f"Caption: \"{result['caption']}\"")
        ok(f"Tags: {result['tags']}")
        ok(f"Model used: {result['model']}")

        # Clean up
        os.unlink(test_image_path)
        return True

    except Exception as e:
        fail(f"Vision pipeline error: {e}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────
# TEST: Cache utility
# ──────────────────────────────────────────────────
def test_cache():
    header("6. Cache Utility (SQLite)")
    try:
        from utils.cache import get_cached, set_cache, clear_cache

        test_query = "__test_query_avivo_bot__"
        test_result = {"answer": "test answer", "source": "test.md"}

        # Should be empty
        cached = get_cached(test_query)
        assert cached is None, "Cache should be empty for new query"
        ok("Cache miss for new query (correct)")

        # Set and retrieve
        set_cache(test_query, test_result)
        cached = get_cached(test_query)
        assert cached is not None, "Cache should return result after set"
        assert cached["answer"] == test_result["answer"]
        ok("Cache hit after set (correct)")

        # Case-insensitive (normalized)
        cached_upper = get_cached(test_query.upper())
        assert cached_upper is not None, "Cache should be case-insensitive"
        ok("Cache is case-insensitive (correct)")

        # Clean up test entry
        clear_cache()
        after_clear = get_cached(test_query)
        assert after_clear is None, "Cache should be empty after clear"
        ok("Cache cleared successfully")

        return True

    except Exception as e:
        fail(f"Cache error: {e}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────
# TEST: History utility
# ──────────────────────────────────────────────────
def test_history():
    header("7. History Utility (Per-User Deque)")
    try:
        from utils.history import add_to_history, get_history, clear_history

        user_id = "test_user_999"

        # Empty to start
        assert get_history(user_id) == []
        ok("Empty history for new user (correct)")

        # Add 4 entries — should keep only last 3
        for i in range(4):
            add_to_history(user_id, "ask", f"query {i}", f"answer {i}")

        history = get_history(user_id)
        assert len(history) == 3, f"Expected 3, got {len(history)}"
        ok("History capped at 3 entries (correct)")

        # Most recent entry should be query 3
        assert history[-1]["query"] == "query 3"
        ok("Most recent entry is correct")

        clear_history(user_id)
        assert get_history(user_id) == []
        ok("History cleared successfully")

        return True

    except Exception as e:
        fail(f"History error: {e}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AVIVO Bot Integration Tests")
    parser.add_argument("--rag",    action="store_true", help="RAG pipeline tests only")
    parser.add_argument("--vision", action="store_true", help="Vision pipeline tests only")
    parser.add_argument("--cache",  action="store_true", help="Cache/history tests only")
    args = parser.parse_args()

    run_all = not (args.rag or args.vision or args.cache)

    print(f"\n{BOLD}{'═'*50}")
    print("  AVIVO Hybrid Bot — Integration Tests")
    print(f"{'═'*50}{RESET}")

    results = {}

    if run_all or args.rag:
        results["Ollama"]     = test_ollama_connection()
        results["Embeddings"] = test_embedding_model()
        results["EmbedPipe"]  = test_embedding_pipeline()
        results["RAG"]        = test_rag_pipeline()

    if run_all or args.vision:
        results["Vision"]     = test_vision_pipeline()

    if run_all or args.cache:
        results["Cache"]      = test_cache()
        results["History"]    = test_history()

    # ── Summary ───────────────────────────────────
    print(f"\n{BOLD}{'─'*50}\n  RESULTS SUMMARY\n{'─'*50}{RESET}")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)

    for name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {name:<15} {status}")

    print(f"\n  {BOLD}{passed}/{total} tests passed{RESET}")

    if passed == total:
        print(f"\n{GREEN}{BOLD}  🚀 All systems go! Run: python app.py{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{YELLOW}{BOLD}  ⚠️  Fix the failing tests before running the bot.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
