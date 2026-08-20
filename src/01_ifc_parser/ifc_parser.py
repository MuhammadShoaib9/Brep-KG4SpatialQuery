import ifcopenshell

class IFCParser:
    
    def __init__(self, ifc_path):
        """
        Initialize parser with IFC file path
        """
        self.ifc_path = ifc_path
        self.ifc_file = None
        
    def load(self):
        """
        Load IFC file
        """
        try:
            self.ifc_file = ifcopenshell.open(self.ifc_path)
            print(f"IFC file loaded: {self.ifc_path}")
            return True
        except Exception as e:
            print(f"Error loading IFC file: {e}")
            return False
    
    def get_storey(self, element):
        """
        Find which storey an element belongs to
        """
        # Check all IfcRelContainedInSpatialStructure relationships
        for rel in self.ifc_file.by_type(
                "IfcRelContainedInSpatialStructure"):
            if element in rel.RelatedElements:
                structure = rel.RelatingStructure
                if structure.is_a("IfcBuildingStorey"):
                    return structure.Name
        return "Unknown"
    
    def extract_elements(self):
        """
        Extract all IfcProduct instances
        with GlobalId, type, name, storey
        """
        if self.ifc_file is None:
            print("IFC file not loaded. Call load() first.")
            return []
        
        elements = []
        
        # Get only physical building elements
        PHYSICAL_TYPES = [
            "IfcWallStandardCase",
            "IfcColumn", "IfcBeam", "IfcSlab",
            "IfcDoor", "IfcWindow", "IfcStair",
            "IfcRoof", "IfcSpace", "IfcMember",
            "IfcPlate", "IfcFooting", "IfcPile", "IfcRailing"
        ]

        products = []
        for element_type in PHYSICAL_TYPES:
            products.extend(
                self.ifc_file.by_type(element_type))
        
        print(f"Total IfcProduct instances found: {len(products)}")
        
        for product in products:
            
            # Skip elements without geometry
            if not product.Representation:
                continue
            
            # Extract basic information
            element_data = {
                'GlobalId' : product.GlobalId,
                'type'     : product.is_a(),
                'name'     : product.Name 
                             if product.Name 
                             else "Unnamed",
                'storey'   : self.get_storey(product)
            }
            
            elements.append(element_data)
        
        print(f"Elements with geometry: {len(elements)}")
        return elements


# Quick test
if __name__ == "__main__":
    
    import sys
    sys.path.append(r"C:\USB data\articles ideas"
                    r"\Brep Graph\System development"
                    r"\BRepGraph_Article1\BRepGraph_Article1")
    
    import config
    
    parser = IFCParser(config.IFC_FILE_PATH)
    
    if parser.load():
        elements = parser.extract_elements()
        
        # Print first 5 elements
        print("\nFirst 5 elements:")
        for e in elements[:5]:
            print(f"  GlobalId : {e['GlobalId']}")
            print(f"  Type     : {e['type']}")
            print(f"  Name     : {e['name']}")
            print(f"  Storey   : {e['storey']}")
            print()
        # Print summary by type
        print("\nElement Summary:")
        type_count = {}
        for e in elements:
            t = e['type']
            type_count[t] = type_count.get(t, 0) + 1

        for t, count in type_count.items():
            print(f"  {t}: {count}")

        print(f"\nTotal: {len(elements)}")
