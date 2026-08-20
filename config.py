import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j settings
NEO4J_URI      = "link"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = "gpt-4"

# IFC file path
IFC_FILE_PATH  = r"local folder"

# Tessellation settings
DEFLECTION_TOLERANCE = 0.001

# Spatial computation settings
VERTEX_TOLERANCE = 0.001
