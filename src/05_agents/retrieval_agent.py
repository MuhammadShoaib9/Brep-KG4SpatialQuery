# -*- coding: utf-8 -*-
import json
from neo4j import GraphDatabase


class RetrievalAgent:

    def __init__(self, uri, user, password):
        """
        Initialize Retrieval Agent
        with Neo4j connection
        """
        self.uri      = uri
        self.user     = user
        self.password = password
        self.driver   = None

    def connect(self):
        """
        Connect to Neo4j database
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print("Retrieval Agent connected "
                  "to Neo4j")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def close(self):
        """
        Close Neo4j connection
        """
        if self.driver:
            self.driver.close()

    # ------------------------------------------
    # Element Finding Methods
    # ------------------------------------------

    def find_by_global_id(self, global_id):
        """
        Find element by GlobalId
        Most precise search
        Returns element dict or None
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:IfcProduct 
                      {GlobalId: $GlobalId})
                RETURN e.GlobalId as GlobalId,
                       e.name     as name,
                       e.type     as type,
                       e.storey   as storey
            """, GlobalId=global_id)
            record = result.single()
            if record:
                return {
                    'GlobalId': record['GlobalId'],
                    'name'    : record['name'],
                    'type'    : record['type'],
                    'storey'  : record['storey']
                }
            return None

    def find_by_name(self, name,
                     storey_filter=None):
        """
        Find element by name
        Optional storey filter
        Returns element dict or None
        """
        with self.driver.session() as session:
            if storey_filter:
                result = session.run("""
                    MATCH (e:IfcProduct 
                          {name: $name})
                    WHERE e.storey = $storey
                    RETURN e.GlobalId as GlobalId,
                           e.name     as name,
                           e.type     as type,
                           e.storey   as storey
                    LIMIT 1
                """,
                    name   = name,
                    storey = storey_filter)
            else:
                result = session.run("""
                    MATCH (e:IfcProduct 
                          {name: $name})
                    RETURN e.GlobalId as GlobalId,
                           e.name     as name,
                           e.type     as type,
                           e.storey   as storey
                    LIMIT 1
                """, name=name)

            record = result.single()
            if record:
                return {
                    'GlobalId': record['GlobalId'],
                    'name'    : record['name'],
                    'type'    : record['type'],
                    'storey'  : record['storey']
                }
            return None

    def find_by_type(self, ifc_type,
                     storey_filter=None):
        """
        Find all elements of given type
        Optional storey filter
        Returns list of element dicts
        """
        with self.driver.session() as session:
            if storey_filter:
                result = session.run("""
                    MATCH (e:IfcProduct 
                          {type: $type})
                    WHERE e.storey = $storey
                    RETURN e.GlobalId as GlobalId,
                           e.name     as name,
                           e.type     as type,
                           e.storey   as storey
                """,
                    type   = ifc_type,
                    storey = storey_filter)
            else:
                result = session.run("""
                    MATCH (e:IfcProduct 
                          {type: $type})
                    RETURN e.GlobalId as GlobalId,
                           e.name     as name,
                           e.type     as type,
                           e.storey   as storey
                """, type=ifc_type)

            elements = []
            for record in result:
                elements.append({
                    'GlobalId': record['GlobalId'],
                    'name'    : record['name'],
                    'type'    : record['type'],
                    'storey'  : record['storey']
                })
            return elements

    def find_element(self, source_dict,
                     storey_filter=None):
        """
        Find element using priority order:
        1. GlobalId (most precise)
        2. Name
        3. Type only (broadest)
        Returns element dict or None
        """
        identifier = source_dict.get(
            'identifier', 'name')
        value      = source_dict.get('value', '')
        ifc_type   = source_dict.get('type', '')

        # Priority 1 - GlobalId
        if identifier == 'GlobalId':
            element = self.find_by_global_id(value)
            if element:
                return element
            print(f"  Element not found "
                  f"by GlobalId: {value}")
            return None

        # Priority 2 - Name
        if identifier == 'name' and value:
            element = self.find_by_name(
                value, storey_filter)
            if element:
                return element
            print(f"  Element not found "
                  f"by name: {value}")

        # Priority 3 - Type only
        if ifc_type and ifc_type != 'IfcProduct':
            elements = self.find_by_type(
                ifc_type, storey_filter)
            if elements:
                print(f"  Found {len(elements)} "
                      f"elements of type "
                      f"{ifc_type}")
                return elements
            print(f"  No elements found "
                  f"of type: {ifc_type}")

        return None

    # ------------------------------------------
    # Cache Check
    # ------------------------------------------

    def check_cache(self, global_id_a,
                    global_id_b):
        """
        Check if spatial relationship
        already computed and cached
        Returns relationship type or None
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:IfcProduct 
                      {GlobalId: $id_a})
                      -[r]->(b:IfcProduct 
                      {GlobalId: $id_b})
                WHERE type(r) IN [
                    'TOUCHES', 'OVERLAPS',
                    'CONTAINS', 'COVERS',
                    'EQUALS', 'DISJOINT'
                ]
                RETURN type(r) as rel_type
            """,
                id_a=global_id_a,
                id_b=global_id_b)
            record = result.single()
            if record:
                return record['rel_type']
            return None

    # ------------------------------------------
    # Candidate Retrieval for Element Query
    # ------------------------------------------

    def get_candidates(self, target_type,
                       storey_filter=None):
        """
        Get all candidate elements
        of target type for element query
        Optional storey filter
        Returns list of element dicts
        """
        with self.driver.session() as session:
            if storey_filter:
                result = session.run("""
                    MATCH (e:IfcProduct 
                          {type: $type})
                    WHERE e.storey = $storey
                    RETURN e.GlobalId as GlobalId,
                           e.name     as name,
                           e.type     as type,
                           e.storey   as storey
                """,
                    type   = target_type,
                    storey = storey_filter)
            else:
                result = session.run("""
                    MATCH (e:IfcProduct 
                          {type: $type})
                    RETURN e.GlobalId as GlobalId,
                           e.name     as name,
                           e.type     as type,
                           e.storey   as storey
                """, type=target_type)

            candidates = []
            for record in result:
                candidates.append({
                    'GlobalId': record['GlobalId'],
                    'name'    : record['name'],
                    'type'    : record['type'],
                    'storey'  : record['storey']
                })
            return candidates

    def get_all_elements(self,
                         storey_filter=None):
        """
        Get all elements when target
        type is IfcProduct (unknown type)
        Returns list of element dicts
        """
        with self.driver.session() as session:
            if storey_filter:
                result = session.run("""
                    MATCH (e:IfcProduct)
                    WHERE e.storey = $storey
                    RETURN e.GlobalId as GlobalId,
                           e.name     as name,
                           e.type     as type,
                           e.storey   as storey
                """, storey=storey_filter)
            else:
                result = session.run("""
                    MATCH (e:IfcProduct)
                    RETURN e.GlobalId as GlobalId,
                           e.name     as name,
                           e.type     as type,
                           e.storey   as storey
                """)

            elements = []
            for record in result:
                elements.append({
                    'GlobalId': record['GlobalId'],
                    'name'    : record['name'],
                    'type'    : record['type'],
                    'storey'  : record['storey']
                })
            return elements

    # ------------------------------------------
    # Main Retrieve Method
    # ------------------------------------------

    def retrieve(self, command):
        """
        Main retrieval method
        Takes structured command from
        Understanding Agent
        Returns retrieval result dict
        """
        query_type = command.get(
            'query_type', 'unknown')

        # ----------------------------------------
        # Relationship Query
        # ----------------------------------------
        if query_type == 'relationship':

            print("  Retrieving elements "
                  "for relationship query...")

            # Find entity A
            entity_a = self.find_element(
                command.get('entity_a', {}))

            if not entity_a:
                return {
                    'status' : 'error',
                    'message': f"Element A not found: "
                               f"{command.get('entity_a', {}).get('value', '')}"
                }

            # Find entity B
            entity_b = self.find_element(
                command.get('entity_b', {}))

            if not entity_b:
                return {
                    'status' : 'error',
                    'message': f"Element B not found: "
                               f"{command.get('entity_b', {}).get('value', '')}"
                }

            # Check cache
            cached = self.check_cache(
                entity_a['GlobalId'],
                entity_b['GlobalId'])

            if cached:
                return {
                    'status'      : 'cached',
                    'query_type'  : 'relationship',
                    'entity_a'    : entity_a,
                    'entity_b'    : entity_b,
                    'relationship': cached
                }

            return {
                'status'    : 'compute',
                'query_type': 'relationship',
                'entity_a'  : entity_a,
                'entity_b'  : entity_b
            }

        # ----------------------------------------
        # Element Query
        # ----------------------------------------
        elif query_type == 'element':

            print("  Retrieving elements "
                  "for element query...")

            storey_filter = command.get(
                'storey_filter', None)
            predicate     = command.get(
                'predicate', '')
            target_type   = command.get(
                'target_type', 'IfcProduct')

            # Find source element
            source = self.find_element(
                command.get('source', {}),
                storey_filter)

            if not source:
                return {
                    'status' : 'error',
                    'message': f"Source element not found: "
                               f"{command.get('source', {}).get('value', '')}"
                }

            # Handle list return from type search
            if isinstance(source, list):
                source = source[0]

            # Get candidate elements
            if target_type == 'IfcProduct':
                candidates = self.get_all_elements(
                    storey_filter)
            else:
                candidates = self.get_candidates(
                    target_type, storey_filter)

            # Remove source from candidates
            candidates = [
                c for c in candidates
                if c['GlobalId'] !=
                source['GlobalId']
            ]

            print(f"  Source: {source['name']}")
            print(f"  Candidates: "
                  f"{len(candidates)} elements")
            print(f"  Predicate: {predicate}")

            return {
                'status'    : 'compute',
                'query_type': 'element',
                'predicate' : predicate,
                'source'    : source,
                'candidates': candidates
            }

        # ----------------------------------------
        # Unknown Query
        # ----------------------------------------
        else:
            return {
                'status' : 'error',
                'message': command.get(
                    'message',
                    'Unknown query type')
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
    from understanding_agent import \
        UnderstandingAgent

    # Initialize agents
    understanding = UnderstandingAgent(
        api_key = config.OPENAI_API_KEY,
        model   = config.OPENAI_MODEL)

    retrieval = RetrievalAgent(
        uri      = config.NEO4J_URI,
        user     = config.NEO4J_USER,
        password = config.NEO4J_PASSWORD)

    retrieval.connect()

    # Test queries
    test_queries = [
        # Relationship query
        "What is the spatial relationship "
        "between Wand-Int-ERDG-4 "
        "and Wand-Int-ERDG-2?",

        # Element query
        "Which walls touch Wand-Int-ERDG-2?",

        # Element query different types
        "Which slabs cover Wand-Int-ERDG-4?",

        # Storey filter
        "Which walls on the ground floor "
        "touch Bodenplatte?"
    ]

    print("=== Retrieval Agent Tests ===\n")

    import time

    for i, query in enumerate(test_queries):
        print(f"\nTest {i+1}: {query}")
        print("-" * 40)

        # Step 1 - Understand query
        command = understanding.parse(query)
        print(f"Command: "
              f"{json.dumps(command, indent=2)}")

        # Step 2 - Retrieve elements
        result = retrieval.retrieve(command)
        print(f"Retrieval result:")
        print(f"  Status: {result['status']}")

        if result['status'] == 'error':
            print(f"  Error: {result['message']}")

        elif result['status'] == 'cached':
            print(f"  Cached relationship: "
                  f"{result['relationship']}")
            print(f"  A: {result['entity_a']['name']}")
            print(f"  B: {result['entity_b']['name']}")

        elif result['status'] == 'compute':
            if result['query_type'] == \
                    'relationship':
                print(f"  A: "
                      f"{result['entity_a']['name']}")
                print(f"  B: "
                      f"{result['entity_b']['name']}")
                print(f"  Ready for computation")

            elif result['query_type'] == 'element':
                print(f"  Source: "
                      f"{result['source']['name']}")
                print(f"  Candidates: "
                      f"{len(result['candidates'])}")
                print(f"  Predicate: "
                      f"{result['predicate']}")
                print(f"  First 3 candidates:")
                for c in result['candidates'][:3]:
                    print(f"    - {c['name']} "
                          f"({c['type']})")

        time.sleep(2)

    retrieval.close()
    input("\nPress Enter to exit...")
