# -*- coding: utf-8 -*-
import sys
import time
from neo4j import GraphDatabase

sys.path.append(
    r"C:\USB data\articles ideas\Brep Graph"
    r"\System development\BRepGraph_Article1"
    r"\BRepGraph_Article1\src\05_agents")

from understanding_agent import UnderstandingAgent
from retrieval_agent     import RetrievalAgent
from computation_agent   import ComputationAgent
from response_agent      import ResponseAgent


class ManagerAgent:

    def __init__(self,
                 openai_api_key,
                 openai_model,
                 neo4j_uri,
                 neo4j_user,
                 neo4j_password,
                 vertex_tolerance=0.001):
        """
        Initialize Manager Agent
        Coordinates all other agents
        Single entry point for user queries
        """
        self.openai_api_key   = openai_api_key
        self.openai_model     = openai_model
        self.neo4j_uri        = neo4j_uri
        self.neo4j_user       = neo4j_user
        self.neo4j_password   = neo4j_password
        self.vertex_tolerance = vertex_tolerance

        # Initialize all agents
        self.understanding = UnderstandingAgent(
            api_key = openai_api_key,
            model   = openai_model)

        self.retrieval = RetrievalAgent(
            uri      = neo4j_uri,
            user     = neo4j_user,
            password = neo4j_password)

        self.computation = ComputationAgent(
            uri              = neo4j_uri,
            user             = neo4j_user,
            password         = neo4j_password,
            vertex_tolerance = vertex_tolerance)

        self.response = ResponseAgent(
            api_key = openai_api_key,
            model   = openai_model)

        # Query history
        self.history = []

    def connect(self):
        """
        Connect all agents to Neo4j
        Returns True if successful
        """
        print("\nConnecting agents...")
        r1 = self.retrieval.connect()
        r2 = self.computation.connect()
        if r1 and r2:
            print("All agents connected ✅")
            return True
        print("Connection failed ❌")
        return False

    def close(self):
        """
        Close all agent connections
        """
        self.retrieval.close()
        self.computation.close()
        print("\nAll connections closed.")

    def run_query(self, nl_query):
        """
        Run full pipeline for one NL query
        Returns final NL answer

        Pipeline:
        1. Understanding Agent
           → parse NL query
        2. Retrieval Agent
           → find elements in graph
        3. Computation Agent
           → compute relationship
        4. Response Agent
           → format NL answer
        """
        total_start = time.time()

        print(f"\n{'='*50}")
        print(f"Query: {nl_query}")
        print(f"{'='*50}")

        # ----------------------------------------
        # Step 1 - Understanding Agent
        # ----------------------------------------
        print("\n[Step 1] Understanding query...")
        step_start = time.time()

        command = self.understanding.parse(
            nl_query)

        print(f"  Time: "
              f"{time.time()-step_start:.3f}s")

        # Handle understanding error
        if command.get('query_type') == 'error':
            return (
                "Sorry, I could not understand "
                "your query. Please rephrase it.\n"
                f"Error: {command.get('message', '')}"
            )

        # Handle unknown query
        if command.get('query_type') == 'unknown':
            return (
                "I could not identify the building "
                "elements in your query. Please "
                "specify elements by name, GlobalId,"
                " or IFC type."
            )

        print(f"  Query type: "
              f"{command.get('query_type')}")

        # ----------------------------------------
        # Step 2 - Retrieval Agent
        # ----------------------------------------
        print("\n[Step 2] Retrieving elements...")
        step_start = time.time()

        retrieval_result = self.retrieval.retrieve(
            command)

        print(f"  Time: "
              f"{time.time()-step_start:.3f}s")

        # Handle retrieval error
        if retrieval_result.get('status') == \
                'error':
            return (
                f"Element not found in the model. "
                f"{retrieval_result.get('message', '')}"
                f"\nPlease check the element name "
                f"or identifier."
            )

        # ----------------------------------------
        # Step 3 - Computation Agent
        # ----------------------------------------
        print("\n[Step 3] Computing relationship...")
        step_start = time.time()

        computation_result = \
            self.computation.compute(
                retrieval_result)

        print(f"  Time: "
              f"{time.time()-step_start:.3f}s")

        # Handle computation error
        if computation_result.get('status') == \
                'error':
            return (
                "Could not compute the spatial "
                "relationship. Please try again.\n"
                f"Error: "
                f"{computation_result.get('message', '')}"
            )

        # ----------------------------------------
        # Step 4 - Response Agent
        # ----------------------------------------
        print("\n[Step 4] Formatting answer...")
        step_start = time.time()

        answer = self.response.format(
            computation_result)

        print(f"  Time: "
              f"{time.time()-step_start:.3f}s")

        # Total time
        total_time = time.time() - total_start
        print(f"\nTotal time: {total_time:.3f}s")

        # Store in history
        self.history.append({
            'query' : nl_query,
            'answer': answer,
            'time'  : total_time
        })

        return answer

    def print_welcome(self):
        """
        Print welcome message
        """
        print("\n" + "="*50)
        print("  BRep Graph Spatial Query System")
        print("  Article 1 - B-Rep Knowledge Graph")
        print("="*50)
        print("\nSystem ready. Ask spatial queries")
        print("about building elements.")
        print("\nExamples:")
        print("  - What is the relationship between"
              " Wall A and Slab B?")
        print("  - Which walls touch Bodenplatte?")
        print("  - Which slabs cover "
              "Wand-Int-ERDG-4?")
        print("\nType 'exit' or 'quit' to stop.")
        print("Type 'history' to see past queries.")
        print("-"*50)

    def print_history(self):
        """
        Print query history
        """
        if not self.history:
            print("\nNo queries yet.")
            return

        print(f"\n--- Query History "
              f"({len(self.history)} queries) ---")
        for i, item in enumerate(self.history):
            print(f"\n{i+1}. Query: {item['query']}")
            print(f"   Answer: {item['answer']}")
            print(f"   Time: {item['time']:.3f}s")
        print("-"*40)

    def start(self):
        """
        Start interactive query loop
        Accepts user queries until
        'exit' or 'quit' is typed
        """
        self.print_welcome()

        while True:
            try:
                # Get user input
                print("\n")
                nl_query = input(
                    "Your query: ").strip()

                # Check exit commands
                if nl_query.lower() in [
                        'exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break

                # Check history command
                if nl_query.lower() == 'history':
                    self.print_history()
                    continue

                # Skip empty input
                if not nl_query:
                    print("Please enter a query.")
                    continue

                # Run full pipeline
                answer = self.run_query(nl_query)

                # Print final answer
                print(f"\n{'='*50}")
                print(f"ANSWER:")
                print(f"{answer}")
                print(f"{'='*50}")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")

        self.close()


# Quick test
if __name__ == "__main__":

    import sys
    sys.path.append(
        r"C:\USB data\articles ideas\Brep Graph"
        r"\System development\BRepGraph_Article1"
        r"\BRepGraph_Article1")

    import config

    # Initialize Manager Agent
    manager = ManagerAgent(
        openai_api_key   = config.OPENAI_API_KEY,
        openai_model     = config.OPENAI_MODEL,
        neo4j_uri        = config.NEO4J_URI,
        neo4j_user       = config.NEO4J_USER,
        neo4j_password   = config.NEO4J_PASSWORD,
        vertex_tolerance = config.VERTEX_TOLERANCE)

    # Connect all agents
    if manager.connect():

        # Run test queries first
        print("\n=== Automated Tests ===")

        test_queries = [
            "What is the spatial relationship "
            "between Wand-Int-ERDG-4 "
            "and Wand-Int-ERDG-2?",

            "Which walls touch Bodenplatte?",

            "Which slabs cover Wand-Int-ERDG-4?"
        ]

        for query in test_queries:
            answer = manager.run_query(query)
            print(f"\nFINAL ANSWER:\n{answer}")
            time.sleep(3)

        # Start interactive loop
        print("\n=== Interactive Mode ===")
        manager.start()
