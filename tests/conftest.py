import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.pop("GEMINI_API_KEY", None)          # tests always use the offline FakeGemini
