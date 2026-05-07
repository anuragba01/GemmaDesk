import sys
import os
sys.path.append("./src")
from engines.document import DocumentEngine

# Dummy gemma engine
class DummyGemma:
    def answer(self, prompt):
        return "EASY"

with open("dummy.txt", "w") as f:
    f.write("Hello world")

engine = DocumentEngine("./chroma_db", "nomic-ai/nomic-embed-text-v1.5", gemma_engine=DummyGemma())
engine.ingest_text("dummy.txt")
