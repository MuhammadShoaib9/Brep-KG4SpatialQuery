# -*- coding: utf-8 -*-
import json
import os
import time
from neo4j import GraphDatabase


class GraphBuilder:

    def __init__(self, uri, user, password):
        """
        Initialize graph builder
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
            print("Connected to Neo4j successfully")
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
            print("Neo4j connection closed")

    def clear_graph(self):
        """
        Clear all existing nodes
        and relationships
        """
        with self.driver.session() as session:
            session.run(
                "MATCH (n) DETACH DELETE n")
            print("Graph cleared")

    def create_constraints(self):
        """
        Create uniqueness constraints
        for faster queries
        """
        with self.driver.session() as session:

            session.run("""
                CREATE CONSTRAINT element_globalid
                IF NOT EXISTS
                FOR (e:IfcProduct)
                REQUIRE e.GlobalId IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT vertex_vid
                IF NOT EXISTS
                FOR (v:Vertex)
                REQUIRE v.vid IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT face_fid
                IF NOT EXISTS
                FOR (f:Face)
                REQUIRE f.fid IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT edge_eid
                IF NOT EXISTS
                FOR (e:Edge)
                REQUIRE e.eid IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT storey_name
                IF NOT EXISTS
                FOR (s:IfcBuildingStorey)
                REQUIRE s.name IS UNIQUE
            """)

            print("Constraints created")

    def get_ifc_hierarchy(self, element_type):
        """
        Return IFC class hierarchy labels
        for a given element type
        """
        hierarchy = {
            'IfcWallStandardCase': [
                'IfcWallStandardCase',
                'IfcWall',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcWall': [
                'IfcWall',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcColumn': [
                'IfcColumn',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcBeam': [
                'IfcBeam',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcSlab': [
                'IfcSlab',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcDoor': [
                'IfcDoor',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcWindow': [
                'IfcWindow',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcStair': [
                'IfcStair',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcMember': [
                'IfcMember',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcRailing': [
                'IfcRailing',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcSpace': [
                'IfcSpace',
                'IfcSpatialStructureElement',
                'IfcSpatialElement',
                'IfcProduct'
            ],
            'IfcFooting': [
                'IfcFooting',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcPile': [
                'IfcPile',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ],
            'IfcPlate': [
                'IfcPlate',
                'IfcBuildingElement',
                'IfcElement',
                'IfcProduct'
            ]
        }
        return hierarchy.get(
            element_type, ['IfcProduct'])

    def create_storey_node(self,
                           session,
                           storey_name):
        """
        Create storey node if not exists
        """
        session.run("""
            MERGE (s:IfcBuildingStorey:
                   IfcSpatialStructureElement {
                name: $name
            })
        """, name=storey_name)

    def create_element_node(self,
                            session,
                            element):
        """
        Create element node with
        IFC hierarchy labels
        """
        labels    = self.get_ifc_hierarchy(
            element['type'])
        label_str = ':'.join(labels)

        session.run(f"""
            MERGE (e:{label_str} {{
                GlobalId: $GlobalId
            }})
            SET e.ifc_id  = $ifc_id,
                e.name    = $name,
                e.type    = $type,
                e.storey  = $storey,
                e.xmin    = $xmin,
                e.xmax    = $xmax,
                e.ymin    = $ymin,
                e.ymax    = $ymax,
                e.zmin    = $zmin,
                e.zmax    = $zmax
        """,
            GlobalId = element['GlobalId'],
            ifc_id   = element.get('ifc_id', ''),
            name     = element['name'],
            type     = element['type'],
            storey   = element['storey'],
            xmin     = element['bbox']['xmin'],
            xmax     = element['bbox']['xmax'],
            ymin     = element['bbox']['ymin'],
            ymax     = element['bbox']['ymax'],
            zmin     = element['bbox']['zmin'],
            zmax     = element['bbox']['zmax']
        )

        session.run("""
            MATCH (e:IfcProduct {
                GlobalId: $GlobalId})
            MATCH (s:IfcBuildingStorey {
                name: $storey})
            MERGE (e)-[:CONTAINED_IN]->(s)
        """,
            GlobalId = element['GlobalId'],
            storey   = element['storey']
        )

    def create_vertex_nodes_batch(self,
                                  session,
                                  element):
        """
        Create ALL vertex nodes for one element
        in a single batch database call
        """
        global_id = element['GlobalId']
        vertices  = element['vertices']

        vertex_list = []
        for idx, vertex in enumerate(vertices):
            vertex_list.append({
                'vid'  : f"{global_id}_V_{idx}",
                'x'    : float(vertex[0]),
                'y'    : float(vertex[1]),
                'z'    : float(vertex[2]),
                'owner': global_id
            })

        session.run("""
            UNWIND $vertices AS v
            MERGE (vn:Vertex {vid: v.vid})
            SET vn.x     = v.x,
                vn.y     = v.y,
                vn.z     = v.z,
                vn.owner = v.owner
            WITH vn, v.owner AS gid
            MATCH (e:IfcProduct {GlobalId: gid})
            MERGE (e)-[:HAS_VERTEX]->(vn)
        """, vertices=vertex_list)

    def create_face_nodes_batch(self,
                                session,
                                element):
        """
        Create ALL face nodes for one element
        in a single batch database call
        Includes centroid for COVER check
        """
        global_id    = element['GlobalId']
        faces        = element['faces']
        face_normals = element['face_normals']
        face_areas   = element['face_areas']
        vertices     = element['vertices']

        face_list = []
        for idx, face in enumerate(faces):

            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]

            # Centroid = average of 3 vertices
            cx = (v1[0]+v2[0]+v3[0]) / 3
            cy = (v1[1]+v2[1]+v3[1]) / 3
            cz = (v1[2]+v2[2]+v3[2]) / 3

            normal = face_normals[idx]
            area   = face_areas[idx]

            face_list.append({
                'fid'     : f"{global_id}_F_{idx}",
                'normal_x': float(normal[0]),
                'normal_y': float(normal[1]),
                'normal_z': float(normal[2]),
                'area'    : float(area),
                'v1_idx'  : int(face[0]),
                'v2_idx'  : int(face[1]),
                'v3_idx'  : int(face[2]),
                'cx'      : float(cx),
                'cy'      : float(cy),
                'cz'      : float(cz),
                'owner'   : global_id
            })

        session.run("""
            UNWIND $faces AS f
            MERGE (fn:Face {fid: f.fid})
            SET fn.normal_x   = f.normal_x,
                fn.normal_y   = f.normal_y,
                fn.normal_z   = f.normal_z,
                fn.area       = f.area,
                fn.v1_idx     = f.v1_idx,
                fn.v2_idx     = f.v2_idx,
                fn.v3_idx     = f.v3_idx,
                fn.centroid_x = f.cx,
                fn.centroid_y = f.cy,
                fn.centroid_z = f.cz,
                fn.owner      = f.owner
            WITH fn, f.owner AS gid
            MATCH (e:IfcProduct {GlobalId: gid})
            MERGE (e)-[:HAS_FACE]->(fn)
        """, faces=face_list)

    def create_edge_nodes_batch(self,
                                session,
                                element):
        """
        Create ALL edge nodes for one element
        in a single batch database call
        """
        global_id    = element['GlobalId']
        edges        = element['edges']
        edge_lengths = element['edge_lengths']

        edge_list = []
        for idx, edge in enumerate(edges):
            edge_list.append({
                'eid'   : f"{global_id}_E_{idx}",
                'length': float(edge_lengths[idx]),
                'v1_idx': int(edge[0]),
                'v2_idx': int(edge[1]),
                'owner' : global_id
            })

        session.run("""
            UNWIND $edges AS eg
            MERGE (en:Edge {eid: eg.eid})
            SET en.length = eg.length,
                en.v1_idx = eg.v1_idx,
                en.v2_idx = eg.v2_idx,
                en.owner  = eg.owner
            WITH en, eg.owner AS gid
            MATCH (e:IfcProduct {GlobalId: gid})
            MERGE (e)-[:HAS_EDGE]->(en)
        """, edges=edge_list)

    def validate_graph(self):
        """
        Validate graph was built correctly
        """
        print("\n--- Graph Validation ---")
        with self.driver.session() as session:

            result = session.run("""
                MATCH (e:IfcProduct)
                RETURN count(e) as count
            """)
            print(f"Element nodes  : "
                  f"{result.single()['count']}")

            result = session.run("""
                MATCH (v:Vertex)
                RETURN count(v) as count
            """)
            print(f"Vertex nodes   : "
                  f"{result.single()['count']}")

            result = session.run("""
                MATCH (f:Face)
                RETURN count(f) as count
            """)
            print(f"Face nodes     : "
                  f"{result.single()['count']}")

            result = session.run("""
                MATCH (eg:Edge)
                RETURN count(eg) as count
            """)
            print(f"Edge nodes     : "
                  f"{result.single()['count']}")

            result = session.run("""
                MATCH (s:IfcBuildingStorey)
                RETURN count(s) as count
            """)
            print(f"Storey nodes   : "
                  f"{result.single()['count']}")

            result = session.run("""
                MATCH ()-[r]->()
                RETURN count(r) as count
            """)
            print(f"Relationships  : "
                  f"{result.single()['count']}")

            print("\nElement breakdown:")
            result = session.run("""
                MATCH (e:IfcProduct)
                RETURN e.type as type,
                       count(e) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"  {record['type']}: "
                      f"{record['count']}")

        print("--- Validation Complete ---")

    def build(self, brep_data):
        """
        Build complete B-Rep graph
        from brep_data using batch insertion
        """
        total_start = time.time()

        print(f"\nBuilding graph for "
              f"{len(brep_data)} elements...")

        # Step 1 - Clear existing graph
        t = time.time()
        self.clear_graph()
        print(f"  Time: {time.time()-t:.2f}s")

        # Step 2 - Create constraints
        t = time.time()
        self.create_constraints()
        print(f"  Time: {time.time()-t:.2f}s")

        # Step 3 - Create storey nodes
        print("\nCreating storey nodes...")
        t = time.time()
        storeys = set(
            e['storey'] for e in brep_data)
        with self.driver.session() as session:
            for storey in storeys:
                self.create_storey_node(
                    session, storey)
                print(f"  Storey: {storey}")
        print(f"  Time: {time.time()-t:.2f}s")

        # Step 4 - Create element nodes
        print("\nCreating element nodes...")
        t = time.time()
        with self.driver.session() as session:
            for i, element in enumerate(
                    brep_data):
                self.create_element_node(
                    session, element)
                print(f"  [{i+1}/{len(brep_data)}]"
                      f" {element['type']} - "
                      f"{element['name']}")
        print(f"  Time: {time.time()-t:.2f}s")

        # Step 5 - Create vertex nodes (batch)
        print("\nCreating vertex nodes (batch)...")
        t = time.time()
        with self.driver.session() as session:
            for i, element in enumerate(
                    brep_data):
                self.create_vertex_nodes_batch(
                    session, element)
                print(f"  [{i+1}/{len(brep_data)}]"
                      f" {element['name']} - "
                      f"{len(element['vertices'])}"
                      f" vertices")
        print(f"  Time: {time.time()-t:.2f}s")

        # Step 6 - Create face nodes (batch)
        print("\nCreating face nodes (batch)...")
        t = time.time()
        with self.driver.session() as session:
            for i, element in enumerate(
                    brep_data):
                self.create_face_nodes_batch(
                    session, element)
                print(f"  [{i+1}/{len(brep_data)}]"
                      f" {element['name']} - "
                      f"{len(element['faces'])}"
                      f" faces")
        print(f"  Time: {time.time()-t:.2f}s")

        # Step 7 - Create edge nodes (batch)
        print("\nCreating edge nodes (batch)...")
        t = time.time()
        with self.driver.session() as session:
            for i, element in enumerate(
                    brep_data):
                self.create_edge_nodes_batch(
                    session, element)
                print(f"  [{i+1}/{len(brep_data)}]"
                      f" {element['name']} - "
                      f"{len(element['edges'])}"
                      f" edges")
        print(f"  Time: {time.time()-t:.2f}s")

        print("\nGraph building complete!")
        total_end = time.time()
        print(f"Total time: "
              f"{total_end-total_start:.2f}s")

        self.validate_graph()


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
        r"\BRepGraph_Article1\src\02_tessellator")

    import config
    from tessellator import Tessellator

    JSON_PATH = (
        r"C:\USB data\articles ideas\Brep Graph"
        r"\System development\BRepGraph_Article1"
        r"\BRepGraph_Article1\data\results"
        r"\brep_data.json")

    print("Loading B-Rep data from JSON...")
    tessellator = Tessellator(
        ifc_file  = None,
        tolerance = 0.001)
    brep_data = tessellator.load_from_json(
        JSON_PATH)

    if brep_data:
        builder = GraphBuilder(
            uri      = config.NEO4J_URI,
            user     = config.NEO4J_USER,
            password = config.NEO4J_PASSWORD)

        if builder.connect():
            builder.build(brep_data)
            builder.close()

    input("\nPress Enter to exit...")
