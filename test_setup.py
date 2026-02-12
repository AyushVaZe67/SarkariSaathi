#!/usr/bin/env python3
"""
SarkariSaathi Setup Verification Script
Run this to check if everything is configured correctly
"""

import sys
import os


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_python_version():
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher required")
        return False
    print("✅ Python version OK")
    return True


def check_dependencies():
    print_header("Checking Dependencies")
    required = [
        'fastapi',
        'uvicorn',
        'sentence_transformers',
        'faiss',
        'groq',
        'dotenv',
        'numpy',
        'pydantic'
    ]

    missing = []
    for package in required:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            elif package == 'faiss':
                __import__('faiss')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies installed")
    return True


def check_env_file():
    print_header("Checking Environment Variables")
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("Create .env file with your GROQ_API_KEY")
        return False

    print("✅ .env file exists")

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env")
        return False

    if not api_key.startswith('gsk_'):
        print("⚠️  WARNING: API key doesn't start with 'gsk_'")
        print("   Make sure it's a valid Groq API key")

    print(f"✅ GROQ_API_KEY found (starts with: {api_key[:10]}...)")
    return True


def check_data_file():
    print_header("Checking Data File")
    data_path = "data/girls_education_maharashtra.json"

    if not os.path.exists('data'):
        print("❌ data/ directory not found")
        return False

    print("✅ data/ directory exists")

    if not os.path.exists(data_path):
        print(f"❌ {data_path} not found")
        return False

    print(f"✅ {data_path} exists")

    try:
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ JSON file valid ({len(data)} schemes loaded)")
        return True
    except Exception as e:
        print(f"❌ Error reading JSON: {str(e)}")
        return False


def check_static_files():
    print_header("Checking Static Files")

    if not os.path.exists('static'):
        print("❌ static/ directory not found")
        return False

    print("✅ static/ directory exists")

    if not os.path.exists('static/index.html'):
        print("❌ static/index.html not found")
        return False

    print("✅ static/index.html exists")
    return True


def test_groq_connection():
    print_header("Testing Groq API Connection")

    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("❌ Cannot test - no API key")
            return False

        from groq import Groq
        client = Groq(api_key=api_key)

        print("Sending test message to Groq...")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )

        answer = response.choices[0].message.content
        print(f"✅ Groq API responded: {answer}")
        return True

    except Exception as e:
        print(f"❌ Groq API test failed: {str(e)}")
        return False


def test_embeddings():
    print_header("Testing Embeddings Model")

    try:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model (this may take a moment)...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        test_text = ["Test sentence"]
        embeddings = model.encode(test_text)

        print(f"✅ Embeddings model loaded (dimension: {len(embeddings[0])})")
        return True

    except Exception as e:
        print(f"❌ Embeddings test failed: {str(e)}")
        return False


def test_faiss():
    print_header("Testing FAISS")

    try:
        import faiss
        import numpy as np

        # Create a simple test index
        dimension = 384
        index = faiss.IndexFlatL2(dimension)

        # Add some test vectors
        vectors = np.random.random((10, dimension)).astype('float32')
        index.add(vectors)

        print(f"✅ FAISS working (test index created with {index.ntotal} vectors)")
        return True

    except Exception as e:
        print(f"❌ FAISS test failed: {str(e)}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          SarkariSaathi Setup Verification                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("Data File", check_data_file),
        ("Static Files", check_static_files),
        ("Groq API", test_groq_connection),
        ("Embeddings", test_embeddings),
        ("FAISS", test_faiss),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {str(e)}")
            results.append((name, False))

    # Summary
    print_header("Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\n{passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! You're ready to run the app!")
        print("\nRun: python app.py")
        print("Then open: http://127.0.0.1:8000")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("Refer to TROUBLESHOOTING.md for help.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)