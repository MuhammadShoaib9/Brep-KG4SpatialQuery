
# -*- coding: utf-8 -*-
import sys
import time
from neo4j import GraphDatabase

sys.path.append(
    r"C:\USB data\articles ideas\Brep Graph"
    r"\System development\BRepGraph_Article1"
    r"\BRepGraph_Article1\src\04_spatial_engine")

from spatial_engine import SpatialEngine


class ComputationAgent:

    def __init__(self, uri, user, password,
                 vertex_tolerance=0.001):
        """
        Initialize Computation Agent
        Links to SpatialEngine for
        geometric computation
        """
        self.engine = SpatialEngine(
            uri              = uri,
            user             = user,
            password         = password,
            vertex_tolerance = vertex_tolerance)

    def connect(self):
        """
        Connect spatial engine to Neo4j
        """
        return self.engine.connect()

    def close(self):
        """
        Close connection
        """
        self.engine.close()

    def compute_relationship(self,
                             global_id_a,
                             global_id_b):
        """
        Compute spatial relationship
        between two elements
        Uses SpatialEngine which:
        - checks cache first
        - computes if not cached
        - stores result in graph
        Returns relationship type string
        """
        return self.engine.classify_relationship(
            global_id_a, global_id_b)

    def compute(self, retrieval_result):
        """
        Main computation method
        Takes retrieval result dict
        Returns computation result dict

        Handles two query types:
        1. relationship: compute A vs B
        2. element: compute source vs
           all candidates
        """
        status     = retrieval_result.get(
            'status', 'error')
        query_type = retrieval_result.get(
            'query_type', 'unknown')

        # ----------------------------------------
        # Already cached - return directly
        # ----------------------------------------
        if status == 'cached':
            print(f"  Result from cache: "
                  f"{retrieval_result['relationship']}")
            return {
                'status'      : 'success',
                'query_type'  : 'relationship',
                'entity_a'    : retrieval_result[
                    'entity_a'],
                'entity_b'    : retrieval_result[
                    'entity_b'],
                'relationship': retrieval_result[
                    'relationship'],
                'from_cache'  : True
            }

        # ----------------------------------------
        # Error from retrieval
        # ----------------------------------------
        if status == 'error':
            return {
                'status' : 'error',
                'message': retrieval_result.get(
                    'message', 'Retrieval error')
            }

        # ----------------------------------------
        # Relationship Query
        # ----------------------------------------
        if query_type == 'relationship':

            entity_a = retrieval_result['entity_a']
            entity_b = retrieval_result['entity_b']

            print(f"\n  Computing relationship:")
            print(f"  A: {entity_a['name']}")
            print(f"  B: {entity_b['name']}")

            start = time.time()
            relationship = self.compute_relationship(
                entity_a['GlobalId'],
                entity_b['GlobalId'])
            elapsed = time.time() - start

            print(f"  Result: {relationship}")
            print(f"  Time: {elapsed:.3f}s")

            return {
                'status'      : 'success',
                'query_type'  : 'relationship',
                'entity_a'    : entity_a,
                'entity_b'    : entity_b,
                'relationship': relationship,
                'time'        : elapsed,
                'from_cache'  : False
            }

        # ----------------------------------------
        # Element Query
        # ----------------------------------------
        elif query_type == 'element':

            source     = retrieval_result['source']
            candidates = retrieval_result[
                'candidates']
            predicate  = retrieval_result[
                'predicate']

            print(f"\n  Computing element query:")
            print(f"  Source: {source['name']}")
            print(f"  Predicate: {predicate}")
            print(f"  Checking {len(candidates)} "
                  f"candidates...")

            matches    = []
            total_time = 0

            for candidate in candidates:

                # Skip if same element
                if candidate['GlobalId'] == \
                        source['GlobalId']:
                    continue

                start = time.time()
                relationship = \
                    self.compute_relationship(
                        source['GlobalId'],
                        candidate['GlobalId'])
                elapsed = time.time() - start
                total_time += elapsed

                if relationship == predicate:
                    matches.append({
                        'element'     : candidate,
                        'relationship': relationship,
                        'time'        : elapsed
                    })
                    print(f"  MATCH: "
                          f"{candidate['name']} "
                          f"[{relationship}] "
                          f"({elapsed:.3f}s)")
                else:
                    print(f"  Skip: "
                          f"{candidate['name']} "
                          f"[{relationship}]"
                          f"({elapsed:.3f}s)")

            print(f"\n  Total matches: "
                  f"{len(matches)}")
            print(f"  Total time: "
                  f"{total_time:.3f}s")

            return {
                'status'    : 'success',
                'query_type': 'element',
                'source'    : source,
                'predicate' : predicate,
                'matches'   : matches,
                'total_time': total_time
            }

        # ----------------------------------------
        # Unknown query type
        # ----------------------------------------
        else:
            return {
                'status' : 'error',
                'message': 'Unknown query type'
            }


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
    from retrieval_agent import RetrievalAgent

    # Initialize agents
    retrieval = RetrievalAgent(
        uri      = config.NEO4J_URI,
        user     = config.NEO4J_USER,
        password = config.NEO4J_PASSWORD)

    computation = ComputationAgent(
        uri              = config.NEO4J_URI,
        user             = config.NEO4J_USER,
        password         = config.NEO4J_PASSWORD,
        vertex_tolerance = config.VERTEX_TOLERANCE)

    retrieval.connect()
    computation.connect()

    print("=== Computation Agent Tests ===\n")

    # Test 1 - Relationship query
    print("Test 1: Relationship query")
    print("-" * 40)
    command_1 = {
        "query_type": "relationship",
        "entity_a": {
            "identifier": "name",
            "value"     : "Wand-Int-ERDG-4",
            "type"      : "IfcWallStandardCase"
        },
        "entity_b": {
            "identifier": "name",
            "value"     : "Wand-Int-ERDG-2",
            "type"      : "IfcWallStandardCase"
        }
    }
    retrieval_1 = retrieval.retrieve(command_1)
    result_1    = computation.compute(retrieval_1)
    print(f"\n  Final Result:")
    print(f"  {result_1['entity_a']['name']} "
          f"-[{result_1['relationship']}]-> "
          f"{result_1['entity_b']['name']}")

    # Test 2 - Element query TOUCHES
    print("\nTest 2: Element query - TOUCHES")
    print("-" * 40)
    command_2 = {
        "query_type"  : "element",
        "predicate"   : "TOUCHES",
        "target_type" : "IfcWallStandardCase",
        "source": {
            "identifier": "name",
            "value"     : "Wand-Int-ERDG-2",
            "type"      : "IfcWallStandardCase"
        },
        "storey_filter": None
    }
    retrieval_2 = retrieval.retrieve(command_2)
    result_2    = computation.compute(retrieval_2)
    print(f"\n  Final Result:")
    print(f"  Walls that TOUCH "
          f"{result_2['source']['name']}:")
    for m in result_2['matches']:
        print(f"  → {m['element']['name']}")

    # Test 3 - Element query COVERS
    print("\nTest 3: Element query - COVERS")
    print("-" * 40)
    command_3 = {
        "query_type"  : "element",
        "predicate"   : "COVERS",
        "target_type" : "IfcSlab",
        "source": {
            "identifier": "name",
            "value"     : "Wand-Int-ERDG-4",
            "type"      : "IfcWallStandardCase"
        },
        "storey_filter": None
    }
    retrieval_3 = retrieval.retrieve(command_3)
    result_3    = computation.compute(retrieval_3)
    print(f"\n  Final Result:")
    print(f"  Slabs that COVER "
          f"{result_3['source']['name']}:")
    for m in result_3['matches']:
        print(f"  → {m['element']['name']}")

    # Test 4 - Cached query
    print("\nTest 4: Cached query (same as Test 1)")
    print("-" * 40)
    retrieval_4 = retrieval.retrieve(command_1)
    result_4    = computation.compute(retrieval_4)
    print(f"\n  Final Result:")
    print(f"  {result_4['entity_a']['name']} "
          f"-[{result_4['relationship']}]-> "
          f"{result_4['entity_b']['name']}")
    print(f"  From cache: {result_4['from_cache']}")

    retrieval.close()
    computation.close()

    input("\nPress Enter to exit...")
