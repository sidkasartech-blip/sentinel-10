"""
src/observability/tracer.py

Langfuse tracing — wraps every agent hop with a trace span.
This gives you a full timeline of every query in the dashboard:
  - Which agent ran
  - How long it took
  - How many tokens it used
  - What the input/output was
"""

import os
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

# Initialise once at module level
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)


def get_tracer():
    return langfuse