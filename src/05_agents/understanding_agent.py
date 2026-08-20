# -*- coding: utf-8 -*-
import json
import os
from openai import OpenAI
class UnderstandingAgent:
    def __init__(self, api_key, model="gpt-4",
                 base_url=None):
        """
        Initialize Understanding Agent
        with LLM API key.

        base_url is optional (default None,
        which uses OpenAI's native endpoint).
        Passing a different base_url routes
        requests to another OpenAI-compatible
        provider (e.g. Groq, Mistral, or
        Gemini's OpenAI-compatible endpoint)
        using the same client - see
        llm_providers.py for the provider
        registry.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url)
        self.model    = model
        self.base_url = base_url
    def build_system_prompt(self):
        """
        Build system prompt for GPT-4
        Includes:
        - Role definition
        - IFC entity types (bSDD based)
        - Spatial predicates (DE-9IM based)
        - Query types
        - Identifier priority rules
        - Few-shot examples
        - Output format

        NOTE: This unified system prompt is
        used identically across the House,
        Duplex, and Institute test models.
        It includes the IFC entity types
        required across all three models.
        """
        return """
You are a BIM spatial query parser.
Your job is to convert natural language
queries about building elements into
structured JSON commands.
Output ONLY valid JSON. No explanation.
No extra text. No markdown.
=== IFC ENTITY TYPES (bSDD standard) ===
- IfcWallStandardCase: vertical building
  element, internal or external wall
- IfcWall: general vertical wall element
- IfcSlab: horizontal surface element,
  floor or ceiling
- IfcBeam: horizontal or nearly horizontal
  structural member
- IfcColumn: vertical structural member
- IfcFooting: foundation element below
  grade, supports walls or columns,
  bearing footing or strip footing
- IfcDoor: opening element hosted in wall
- IfcWindow: transparent opening in wall
- IfcSpace: enclosed area representing
  a room or zone
- IfcMember: structural member such as
  roof rafter or purlin
- IfcRailing: guardrail or handrail
- IfcStair: stair element
- IfcProduct: any physical building element
  (use when type is unknown)

=== SPATIAL PREDICATES (DE-9IM based) ===
- TOUCHES: geometries have at least one
  boundary point in common but no interior
  points in common
  (adjacent, meets, adjoins, connected at
  boundary, borders)
- OVERLAPS: geometries share some but not
  all points in common and the intersection
  has the same dimension as the geometries
  themselves — neither contains the other
  (clashes, intersects, conflicts with,
  partially penetrates)
- CONTAINS: geometry B lies completely in
  the interior of geometry A — no boundary
  contact between A and B
  (inside, within, enclosed by,
  fully inside, interior of)
- COVERS: at least one point of B lies in
  A and no point of B lies in the exterior
  of A — B is completely within or on the
  boundary of A
  (completely bounds, fully overlaps
  boundary of, encloses boundary of)
- EQUALS: geometries are topologically
  equal — same interior and same boundary
  (same as, duplicate, identical,
  coincident, copy of)
- DISJOINT: geometries have no point in
  common — no interior or boundary contact
  (separate, no contact, not related
  geometrically, far from, independent of)
NOTE: Words like "supports", "sits on",
"rests on" describe structural meaning
not geometry. Do not map them to COVERS
unless geometry confirms it.

=== QUERY TYPES ===
Type 1 - Relationship Query:
  Ask about relationship between TWO
  specific elements
  Example: "What is the relationship
            between Wall A and Slab B?"
Type 2 - Element Query:
  Ask which elements have a relationship
  with ONE specific element
  Example: "Which walls touch Column A?"
=== IDENTIFIER PRIORITY ===
1. GlobalId (25-char alphanumeric string)
2. Name (element name string)
3. Type only (IfcWall, IfcSlab etc.)
=== STOREY FILTER (optional) ===
If user mentions floor or storey:
- "ground floor" → "Erdgeschoss"
- "upper floor"  → "Dachgeschoss"
- "first floor"  → "Erdgeschoss"
Add storey_filter to output if mentioned.
Use null if not mentioned.
=== OUTPUT FORMAT ===
For Relationship Query:
{
  "query_type": "relationship",
  "entity_a": {
    "identifier": "name",
    "value": "element_name_here",
    "type": "IfcWallStandardCase"
  },
  "entity_b": {
    "identifier": "name",
    "value": "element_name_here",
    "type": "IfcSlab"
  }
}

For Element Query:
{
  "query_type": "element",
  "predicate": "TOUCHES",
  "target_type": "IfcWallStandardCase",
  "source": {
    "identifier": "name",
    "value": "element_name_here",
    "type": "IfcSlab"
  },
  "storey_filter": null
}
For unknown or unclear query:
{
  "query_type": "unknown",
  "message": "Please specify element by
              name, GlobalId, or type"
}
=== FEW-SHOT EXAMPLES ===
Example 1 - Relationship Query:
Input: "What is the spatial relationship
        between Wand-Int-ERDG-4
        and Wand-Int-ERDG-2?"
Output:
{
  "query_type": "relationship",
  "entity_a": {
    "identifier": "name",
    "value": "Wand-Int-ERDG-4",
    "type": "IfcWallStandardCase"
  },
  "entity_b": {
    "identifier": "name",
    "value": "Wand-Int-ERDG-2",
    "type": "IfcWallStandardCase"
  }
}

Example 2 - Element Query by name:
Input: "Which walls touch Wand-Int-ERDG-2?"
Output:
{
  "query_type": "element",
  "predicate": "TOUCHES",
  "target_type": "IfcWallStandardCase",
  "source": {
    "identifier": "name",
    "value": "Wand-Int-ERDG-2",
    "type": "IfcWallStandardCase"
  },
  "storey_filter": null
}
Example 3 - Element Query different types:
Input: "Which slabs cover Wand-Int-ERDG-4?"
Output:
{
  "query_type": "element",
  "predicate": "COVERS",
  "target_type": "IfcSlab",
  "source": {
    "identifier": "name",
    "value": "Wand-Int-ERDG-4",
    "type": "IfcWallStandardCase"
  },
  "storey_filter": null
}
Example 4 - Element Query with storey:
Input: "Which walls on the ground floor
        touch Bodenplatte?"
Output:
{
  "query_type": "element",
  "predicate": "TOUCHES",
  "target_type": "IfcWallStandardCase",
  "source": {
    "identifier": "name",
    "value": "Bodenplatte",
    "type": "IfcSlab"
  },
  "storey_filter": "Erdgeschoss"
}

Example 5 - Query with GlobalId:
Input: "What touches 2XPyKWY018sA1ygZKgQPtU?"
Output:
{
  "query_type": "element",
  "predicate": "TOUCHES",
  "target_type": "IfcProduct",
  "source": {
    "identifier": "GlobalId",
    "value": "2XPyKWY018sA1ygZKgQPtU",
    "type": "IfcProduct"
  },
  "storey_filter": null
}
Example 6 - Relationship Query disjoint:
Input: "Is Bodenplatte disjoint
        from Wand-Ext-OG-1?"
Output:
{
  "query_type": "relationship",
  "entity_a": {
    "identifier": "name",
    "value": "Bodenplatte",
    "type": "IfcSlab"
  },
  "entity_b": {
    "identifier": "name",
    "value": "Wand-Ext-OG-1",
    "type": "IfcWallStandardCase"
  }
}

Example 7 - Element Query overlap:
Input: "Which elements overlap
        with Wendeltreppe?"
Output:
{
  "query_type": "element",
  "predicate": "OVERLAPS",
  "target_type": "IfcProduct",
  "source": {
    "identifier": "name",
    "value": "Wendeltreppe",
    "type": "IfcStair"
  },
  "storey_filter": null
}
Example 8 - Type only query:
Input: "Which columns are inside
        the living space?"
Output:
{
  "query_type": "element",
  "predicate": "CONTAINS",
  "target_type": "IfcColumn",
  "source": {
    "identifier": "name",
    "value": "living space",
    "type": "IfcSpace"
  },
  "storey_filter": null
}
"""
    def parse(self, nl_query):
        """
        Parse natural language query
        into structured command using GPT-4
        Returns dict with structured command
        """
        try:
            response = self.client.chat.completions.create(
                model    = self.model,
                messages = [
                    {
                        "role"   : "system",
                        "content": self.build_system_prompt()
                    },
                    {
                        "role"   : "user",
                        "content": nl_query
                    }
                ],
                temperature = 0.0,
                max_tokens  = 500
            )
            # Extract response text
            response_text = response.choices[0]\
                .message.content.strip()
            # Parse JSON
            command = json.loads(response_text)
            return command
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {response_text}")
            return {
                "query_type": "error",
                "message"   : "Failed to parse query"
            }
        except Exception as e:
            print(f"Error calling GPT-4: {e}")
            return {
                "query_type": "error",
                "message"   : str(e)
            }

# Quick test
if __name__ == "__main__":
    import sys
    sys.path.append(
        r"C:\USB data\articles ideas\Brep Graph"
        r"\System development\BRepGraph_Article1"
        r"\BRepGraph_Article1")
    import config
    agent = UnderstandingAgent(
        api_key = config.OPENAI_API_KEY,
        model   = config.OPENAI_MODEL)
    # Test queries
    test_queries = [
        # Relationship queries
        "What is the spatial relationship "
        "between Wand-Int-ERDG-4 "
        "and Wand-Int-ERDG-2?",
        "Is Bodenplatte disjoint "
        "from Wand-Ext-OG-1?",
        # Element queries
        "Which walls touch Wand-Int-ERDG-2?",
        "Which slabs cover Wand-Int-ERDG-4?",
        "Which elements overlap "
        "with Wendeltreppe?",
        # GlobalId query
        "What touches "
        "2XPyKWY018sA1ygZKgQPtU?",
        # Storey filter
        "Which walls on the ground floor "
        "touch Bodenplatte?",
        # Type only
        "Which columns are inside "
        "the living space?"
    ]

    print("=== Understanding Agent Tests ===\n")
    for i, query in enumerate(test_queries):
        print(f"Test {i+1}: {query}")
        result = agent.parse(query)
        print(f"Result:\n"
              f"{json.dumps(result, indent=2)}")
        print("-" * 40)
    input("\nPress Enter to exit...")
