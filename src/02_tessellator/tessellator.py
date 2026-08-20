import ifcopenshell
import ifcopenshell.geom
import numpy as np
import json
import os


class Tessellator:

    def __init__(self, ifc_file, tolerance=0.001):
        """
        Initialize tessellator
        ifc_file: loaded ifcopenshell file object
        tolerance: tessellation precision
        """
        self.ifc_file  = ifc_file
        self.tolerance = tolerance

        # Setup ifcopenshell geometry settings
        if ifc_file is not None:
            self.settings = ifcopenshell.geom.settings()
            self.settings.set(
                self.settings.USE_WORLD_COORDS, True)
            self.settings.set(
                self.settings.WELD_VERTICES, True)

    def compute_face_normal(self, v1, v2, v3):
        """
        Compute normal vector of a triangular face
        using cross product
        """
        edge1  = v2 - v1
        edge2  = v3 - v1
        normal = np.cross(edge1, edge2)

        # Normalize
        length = np.linalg.norm(normal)
        if length > 0:
            normal = normal / length

        return normal

    def compute_face_area(self, v1, v2, v3):
        """
        Compute area of a triangular face
        """
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross = np.cross(edge1, edge2)
        area  = 0.5 * np.linalg.norm(cross)
        return area

    def compute_bounding_box(self, vertices):
        """
        Compute bounding box from vertices
        """
        vertices_array = np.array(vertices)

        return {
            'xmin': float(np.min(vertices_array[:, 0])),
            'xmax': float(np.max(vertices_array[:, 0])),
            'ymin': float(np.min(vertices_array[:, 1])),
            'ymax': float(np.max(vertices_array[:, 1])),
            'zmin': float(np.min(vertices_array[:, 2])),
            'zmax': float(np.max(vertices_array[:, 2]))
        }

    def tessellate_element(self, element):
        """
        Tessellate a single IFC element
        Returns B-Rep data dictionary
        """
        try:
            # Create shape from IFC element
            shape = ifcopenshell.geom.create_shape(
                self.settings, element)

            # Extract raw geometry
            verts = shape.geometry.verts
            faces = shape.geometry.faces

            # Convert vertices to list of (x,y,z) tuples
            # verts is flat list [x1,y1,z1,x2,y2,z2...]
            vertices = []
            for i in range(0, len(verts), 3):
                vertices.append((
                    float(verts[i]),
                    float(verts[i+1]),
                    float(verts[i+2])
                ))

            # Convert faces to list of (v1,v2,v3) tuples
            # faces is flat list [f1,f2,f3,f4,f5,f6...]
            face_list = []
            for i in range(0, len(faces), 3):
                face_list.append((
                    faces[i],
                    faces[i+1],
                    faces[i+2]
                ))

            # Compute face normals and areas
            face_normals = []
            face_areas   = []

            for f in face_list:
                v1 = np.array(vertices[f[0]])
                v2 = np.array(vertices[f[1]])
                v3 = np.array(vertices[f[2]])

                normal = self.compute_face_normal(
                    v1, v2, v3)
                area   = self.compute_face_area(
                    v1, v2, v3)

                face_normals.append(normal.tolist())
                face_areas.append(float(area))

            # Extract edges from faces
            edges = set()
            for f in face_list:
                edges.add((min(f[0], f[1]),
                           max(f[0], f[1])))
                edges.add((min(f[1], f[2]),
                           max(f[1], f[2])))
                edges.add((min(f[0], f[2]),
                           max(f[0], f[2])))
            edges = list(edges)

            # Compute edge lengths
            edge_lengths = []
            for e in edges:
                v1     = np.array(vertices[e[0]])
                v2     = np.array(vertices[e[1]])
                length = float(np.linalg.norm(v2 - v1))
                edge_lengths.append(length)

            # Compute bounding box
            bbox = self.compute_bounding_box(vertices)

            return {
                'vertices'     : vertices,
                'faces'        : face_list,
                'edges'        : edges,
                'face_normals' : face_normals,
                'face_areas'   : face_areas,
                'edge_lengths' : edge_lengths,
                'bbox'         : bbox
            }

        except Exception as e:
            print(f"  Error tessellating element: {e}")
            return None

    def tessellate(self, elements):
        """
        Tessellate all elements
        elements: list from IFCParser
        Returns list of B-Rep data dictionaries
        """
        brep_data = []

        print(f"\nTessellating {len(elements)} elements...")

        for i, element_info in enumerate(elements):

            # Get IFC element object by GlobalId
            element = self.ifc_file.by_guid(
                element_info['GlobalId'])

            if element is None:
                print(f"  Element not found: "
                      f"{element_info['GlobalId']}")
                continue

            # Tessellate
            brep = self.tessellate_element(element)

            if brep is not None:
                # Combine element info with B-Rep data
                combined = {
                    'GlobalId'     : element_info['GlobalId'],
                    'type'         : element_info['type'],
                    'name'         : element_info['name'],
                    'storey'       : element_info['storey'],
                    'vertices'     : brep['vertices'],
                    'faces'        : brep['faces'],
                    'edges'        : brep['edges'],
                    'face_normals' : brep['face_normals'],
                    'face_areas'   : brep['face_areas'],
                    'edge_lengths' : brep['edge_lengths'],
                    'bbox'         : brep['bbox']
                }
                brep_data.append(combined)

                print(f"  [{i+1}/{len(elements)}] "
                      f"{element_info['type']} - "
                      f"{element_info['name']} - "
                      f"Vertices: {len(brep['vertices'])} "
                      f"Faces: {len(brep['faces'])}")

        print(f"\nTessellation complete.")
        print(f"Successfully tessellated: "
              f"{len(brep_data)}/{len(elements)}")

        return brep_data

    def save_to_json(self, brep_data, output_path):
        """
        Save tessellated B-Rep data to JSON file
        """
        try:
            # Convert to JSON serializable format
            json_data = []
            for element in brep_data:
                json_element = {
                    'GlobalId'     : element['GlobalId'],
                    'type'         : element['type'],
                    'name'         : element['name'],
                    'storey'       : element['storey'],
                    'vertices'     : element['vertices'],
                    'faces'        : [list(f) for f
                                      in element['faces']],
                    'edges'        : [list(e) for e
                                      in element['edges']],
                    'face_normals' : element['face_normals'],
                    'face_areas'   : element['face_areas'],
                    'edge_lengths' : element['edge_lengths'],
                    'bbox'         : element['bbox']
                }
                json_data.append(json_element)

            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            print(f"\nB-Rep data saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False

    def load_from_json(self, input_path):
        """
        Load tessellated B-Rep data from JSON file
        """
        try:
            with open(input_path, 'r') as f:
                json_data = json.load(f)

            print(f"B-Rep data loaded from: {input_path}")
            print(f"Elements loaded: {len(json_data)}")
            return json_data

        except Exception as e:
            print(f"Error loading JSON: {e}")
            return None


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
        r"\BRepGraph_Article1\src\01_ifc_parser")

    import config
    from ifc_parser import IFCParser

    # JSON output path
    JSON_PATH = (
        r"C:\USB data\articles ideas\Brep Graph"
        r"\System development\BRepGraph_Article1"
        r"\BRepGraph_Article1\data\results"
        r"\brep_data.json")

    # Check if JSON already exists
    if os.path.exists(JSON_PATH):
        print("JSON file found - loading from file...")
        tessellator = Tessellator(
            ifc_file  = None,
            tolerance = 0.001)
        brep_data = tessellator.load_from_json(JSON_PATH)

    else:
        print("No JSON found - tessellating from IFC...")

        # Stage 1 - Parse
        print("\nStage 1 - Parsing IFC...")
        parser = IFCParser(config.IFC_FILE_PATH)
        parser.load()
        elements = parser.extract_elements()

        # Stage 2 - Tessellate
        print("\nStage 2 - Tessellating...")
        tessellator = Tessellator(
            ifc_file  = parser.ifc_file,
            tolerance = 0.001)
        brep_data = tessellator.tessellate(elements)

        # Save to JSON
        tessellator.save_to_json(brep_data, JSON_PATH)

    # Print summary of first element
    if brep_data:
        first = brep_data[0]
        print(f"\nFirst element B-Rep summary:")
        print(f"  GlobalId  : {first['GlobalId']}")
        print(f"  Type      : {first['type']}")
        print(f"  Vertices  : {len(first['vertices'])}")
        print(f"  Faces     : {len(first['faces'])}")
        print(f"  Edges     : {len(first['edges'])}")
        print(f"  BBox      : "
              f"x[{first['bbox']['xmin']:.3f} -> "
              f"{first['bbox']['xmax']:.3f}] "
              f"y[{first['bbox']['ymin']:.3f} -> "
              f"{first['bbox']['ymax']:.3f}] "
              f"z[{first['bbox']['zmin']:.3f} -> "
              f"{first['bbox']['zmax']:.3f}]")

    input("\nPress Enter to exit...")
