# -*- coding: utf-8 -*-
import json
from openai import OpenAI
class ResponseAgent:
    def __init__(self, api_key,
                 model="gpt-4",
                 base_url=None):
        """
        Initialize Response Agent
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
        - Rules
        - Output format
        - Few-shot examples
        """
        return """
You are a BIM spatial query assistant.
Your job is to convert structured
computation results into clear and
concise natural language answers.
Use ONLY the data provided.
Do not add, translate, or interpret
any values. Do not hallucinate.
=== RULES ===
1. Use element names exactly as provided
2. Use storey names exactly as provided
3. Use IFC type names exactly as provided
4. Use relationship type exactly as provided
5. If total_matches is 0 -> clearly state
   no elements found
6. If total_matches > 0 -> list all matches
7. Keep answer to 3-5 lines maximum
8. Do not explain what relationships mean
9. Do not add information not in the data
=== OUTPUT FORMAT ===

For relationship query:
"[element_a] ([type_a], [storey_a])
[RELATIONSHIP]
[element_b] ([type_b], [storey_b])"
For element query with matches:
"[total_matches] [target_type] found
that [PREDICATE] [source] ([storey]):
-> [match_1_name] ([match_1_type],
                   [match_1_storey])
-> [match_2_name] ([match_2_type],
                   [match_2_storey])"
For no matches:
"No [target_type] found that [PREDICATE]
[source] in this building model."
=== FEW-SHOT EXAMPLES ===
Example 1 - Relationship TOUCHES:
Data:
{
  "query_type": "relationship",
  "element_a": "Wand-Int-ERDG-4",
  "type_a": "IfcWallStandardCase",
  "storey_a": "Erdgeschoss",
  "element_b": "Wand-Int-ERDG-2",
  "type_b": "IfcWallStandardCase",
  "storey_b": "Erdgeschoss",
  "relationship": "TOUCHES"
}
Answer:
"Wand-Int-ERDG-4 (IfcWallStandardCase,
Erdgeschoss) TOUCHES Wand-Int-ERDG-2
(IfcWallStandardCase, Erdgeschoss)."

Example 2 - Relationship DISJOINT:
Data:
{
  "query_type": "relationship",
  "element_a": "Bodenplatte",
  "type_a": "IfcSlab",
  "storey_a": "Erdgeschoss",
  "element_b": "Wand-Ext-OG-1",
  "type_b": "IfcWallStandardCase",
  "storey_b": "Dachgeschoss",
  "relationship": "DISJOINT"
}
Answer:
"Bodenplatte (IfcSlab, Erdgeschoss)
is DISJOINT from Wand-Ext-OG-1
(IfcWallStandardCase, Dachgeschoss).
They share no boundary or interior
points."
Example 3 - Element query one match:
Data:
{
  "query_type": "element",
  "source": "Wand-Int-ERDG-2",
  "source_type": "IfcWallStandardCase",
  "source_storey": "Erdgeschoss",
  "predicate": "TOUCHES",
  "target_type": "IfcWallStandardCase",
  "matches": [
    {
      "name": "Wand-Int-ERDG-4",
      "type": "IfcWallStandardCase",
      "storey": "Erdgeschoss"
    }
  ],
  "total_matches": 1
}
Answer:
"1 IfcWallStandardCase found that
TOUCHES Wand-Int-ERDG-2
(IfcWallStandardCase, Erdgeschoss):
-> Wand-Int-ERDG-4
   (IfcWallStandardCase, Erdgeschoss)"

Example 4 - Element query multiple matches:
Data:
{
  "query_type": "element",
  "source": "Wand-Ext-ERDG-1",
  "source_type": "IfcWallStandardCase",
  "source_storey": "Erdgeschoss",
  "predicate": "TOUCHES",
  "target_type": "IfcWallStandardCase",
  "matches": [
    {
      "name": "Wand-Ext-ERDG-4",
      "type": "IfcWallStandardCase",
      "storey": "Erdgeschoss"
    },
    {
      "name": "Wand-Ext-ERDG-2",
      "type": "IfcWallStandardCase",
      "storey": "Erdgeschoss"
    },
    {
      "name": "Wand-Ext-OG-1",
      "type": "IfcWallStandardCase",
      "storey": "Dachgeschoss"
    }
  ],
  "total_matches": 3
}
Answer:
"3 IfcWallStandardCase elements found
that TOUCH Wand-Ext-ERDG-1
(IfcWallStandardCase, Erdgeschoss):
-> Wand-Ext-ERDG-4
   (IfcWallStandardCase, Erdgeschoss)
-> Wand-Ext-ERDG-2
   (IfcWallStandardCase, Erdgeschoss)
-> Wand-Ext-OG-1
   (IfcWallStandardCase, Dachgeschoss)"

Example 5 - No matches:
Data:
{
  "query_type": "element",
  "source": "Wendeltreppe",
  "source_type": "IfcStair",
  "source_storey": "Erdgeschoss",
  "predicate": "OVERLAPS",
  "target_type": "IfcProduct",
  "matches": [],
  "total_matches": 0
}
Answer:
"No IfcProduct found that OVERLAPS
with Wendeltreppe (IfcStair,
Erdgeschoss) in this building model."
Example 6 - COVERS:
Data:
{
  "query_type": "element",
  "source": "Wand-Int-ERDG-4",
  "source_type": "IfcWallStandardCase",
  "source_storey": "Erdgeschoss",
  "predicate": "COVERS",
  "target_type": "IfcSlab",
  "matches": [
    {
      "name": "Bodenplatte",
      "type": "IfcSlab",
      "storey": "Erdgeschoss"
    }
  ],
  "total_matches": 1
}
Answer:
"1 IfcSlab found that is COVERED by
Wand-Int-ERDG-4
(IfcWallStandardCase, Erdgeschoss):
-> Bodenplatte (IfcSlab, Erdgeschoss)"
"""

    def prepare_data(self,
                     computation_result):
        """
        Prepare data dict from
        computation result for GPT-4
        Extracts relevant fields only
        Returns clean dict for prompt
        """
        query_type = computation_result.get(
            'query_type', 'unknown')
        # Relationship query
        if query_type == 'relationship':
            entity_a = computation_result[
                'entity_a']
            entity_b = computation_result[
                'entity_b']
            return {
                'query_type'  : 'relationship',
                'element_a'   : entity_a['name'],
                'type_a'      : entity_a['type'],
                'storey_a'    : entity_a['storey'],
                'element_b'   : entity_b['name'],
                'type_b'      : entity_b['type'],
                'storey_b'    : entity_b['storey'],
                'relationship': computation_result[
                    'relationship']
            }
        # Element query
        elif query_type == 'element':
            source  = computation_result['source']
            matches = computation_result.get(
                'matches', [])
            # Extract match elements cleanly
            match_list = []
            for m in matches:
                if isinstance(m, dict):
                    element = m.get(
                        'element', m)
                    match_list.append({
                        'name'  : element.get(
                            'name', ''),
                        'type'  : element.get(
                            'type', ''),
                        'storey': element.get(
                            'storey', '')
                    })
            return {
                'query_type'   : 'element',
                'source'       : source['name'],
                'source_type'  : source['type'],
                'source_storey': source['storey'],
                'predicate'    : computation_result\
                    .get('predicate', ''),
                'target_type'  : match_list[0][
                    'type']
                    if match_list
                    else 'IfcProduct',
                'matches'      : match_list,
                'total_matches': len(match_list)
            }
        return {'query_type': 'unknown'}
    def format(self, computation_result):
        """
        Main formatting method
        Takes computation result dict
        Returns natural language answer
        """
        # Handle error results
        if computation_result.get(
                'status') == 'error':
            return (
                f"Sorry, I could not process "
                f"this query. "
                f"{computation_result.get('message', '')}"
            )

        # Prepare clean data for GPT-4
        data = self.prepare_data(
            computation_result)
        # Build user prompt with data
        user_prompt = (
            f"Convert this computation result "
            f"into a natural language answer:\n\n"
            f"{json.dumps(data, indent=2)}"
        )
        try:
            response = self.client.chat\
                .completions.create(
                model    = self.model,
                messages = [
                    {
                        "role"   : "system",
                        "content": self.build_system_prompt()
                    },
                    {
                        "role"   : "user",
                        "content": user_prompt
                    }
                ],
                temperature = 0.0,
                max_tokens  = 300
            )
            answer = response.choices[0]\
                .message.content.strip()
            return answer

        except Exception as e:
            print(f"GPT-4 error: {e}")
            return self.fallback_format(
                computation_result)
    def fallback_format(self,
                        computation_result):
        """
        Fallback formatter if GPT-4 fails
        Uses Python templates
        No API call needed
        """
        query_type = computation_result.get(
            'query_type', 'unknown')
        if query_type == 'relationship':
            a   = computation_result['entity_a']
            b   = computation_result['entity_b']
            rel = computation_result['relationship']
            return (
                f"{a['name']} "
                f"({a['type']}, {a['storey']}) "
                f"{rel} "
                f"{b['name']} "
                f"({b['type']}, {b['storey']})"
            )

        elif query_type == 'element':
            source    = computation_result['source']
            predicate = computation_result['predicate']
            matches   = computation_result['matches']
            if not matches:
                return (
                    f"No elements found that "
                    f"{predicate} "
                    f"{source['name']} "
                    f"in this building model."
                )
            lines = [
                f"{len(matches)} element(s) "
                f"found that {predicate} "
                f"{source['name']}:"
            ]
            for m in matches:
                e = m['element']
                lines.append(
                    f"  -> {e['name']} "
                    f"({e['type']}, {e['storey']})")
            return "\n".join(lines)
        return "Could not format result."
# Quick test
if __name__ == "__main__":
    import sys
    sys.path.append(
        r"C:\USB data\articles ideas\Brep Graph"
        r"\System development\BRepGraph_Article1"
        r"\BRepGraph_Article1")
    sys.path.append(
        r"C:\USB data\articles ideas\Brep Graph"
        r"\System development\BRepGraph_Article1"
        r"\BRepGraph_Article1\src\05_agents")
    import config
    import time
    from retrieval_agent    import RetrievalAgent
    from computation_agent  import ComputationAgent
    # Initialize agents
    retrieval   = RetrievalAgent(
        uri      = config.NEO4J_URI,
        user     = config.NEO4J_USER,
        password = config.NEO4J_PASSWORD)
    computation = ComputationAgent(
        uri              = config.NEO4J_URI,
        user             = config.NEO4J_USER,
        password         = config.NEO4J_PASSWORD,
        vertex_tolerance = config.VERTEX_TOLERANCE)
    response = ResponseAgent(
        api_key = config.OPENAI_API_KEY,
        model   = config.OPENAI_MODEL)
    retrieval.connect()
    computation.connect()
    print("=== Response Agent Tests ===\n")
    # Test 1 - Relationship query
    print("Test 1: Relationship TOUCHES")
    print("-" * 40)
    command_1 = {
        "query_type": "relationship",
        "entity_a"  : {
            "identifier": "name",
            "value"     : "Wand-Int-ERDG-4",
            "type"      : "IfcWallStandardCase"
        },
        "entity_b"  : {
            "identifier": "name",
            "value"     : "Wand-Int-ERDG-2",
            "type"      : "IfcWallStandardCase"
        }
    }
    r1 = retrieval.retrieve(command_1)
    c1 = computation.compute(r1)
    a1 = response.format(c1)
    print(f"Answer:\n{a1}\n")
    time.sleep(3)
    # Test 2 - Element query TOUCHES
    print("Test 2: Element TOUCHES")
    print("-" * 40)
    command_2 = {
        "query_type"   : "element",
        "predicate"    : "TOUCHES",
        "target_type"  : "IfcWallStandardCase",
        "source"       : {
            "identifier": "name",
            "value"     : "Wand-Int-ERDG-2",
            "type"      : "IfcWallStandardCase"
        },
        "storey_filter": None
    }
    r2 = retrieval.retrieve(command_2)
    c2 = computation.compute(r2)
    a2 = response.format(c2)
    print(f"Answer:\n{a2}\n")
    time.sleep(3)
    # Test 3 - Element query COVERS
    print("Test 3: Element COVERS")
    print("-" * 40)
    command_3 = {
        "query_type"   : "element",
        "predicate"    : "COVERS",
        "target_type"  : "IfcSlab",
        "source"       : {
            "identifier": "name",
            "value"     : "Wand-Int-ERDG-4",
            "type"      : "IfcWallStandardCase"
        },
        "storey_filter": None
    }
    r3 = retrieval.retrieve(command_3)
    c3 = computation.compute(r3)
    a3 = response.format(c3)
    print(f"Answer:\n{a3}\n")
    time.sleep(3)
    # Test 4 - No matches
    print("Test 4: No matches - OVERLAPS")
    print("-" * 40)
    command_4 = {
        "query_type"   : "element",
        "predicate"    : "OVERLAPS",
        "target_type"  : "IfcProduct",
        "source"       : {
            "identifier": "name",
            "value"     : "Wendeltreppe",
            "type"      : "IfcStair"
        },
        "storey_filter": None
    }
    r4 = retrieval.retrieve(command_4)
    c4 = computation.compute(r4)
    a4 = response.format(c4)
    print(f"Answer:\n{a4}\n")
    retrieval.close()
    computation.close()
    input("\nPress Enter to exit...")
