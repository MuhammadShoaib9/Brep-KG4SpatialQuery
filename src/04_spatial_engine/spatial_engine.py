# -*- coding: utf-8 -*-
import numpy as np
import time
from neo4j import GraphDatabase


# Small dimensionless epsilon for
# barycentric / algebraic stability.
# Kept separate from vertex_tolerance,
# which is a geometric DISTANCE and must
# not be reused for dimensionless
# barycentric coordinate comparisons.
BARYCENTRIC_EPSILON = 1e-6
ALGEBRAIC_EPSILON    = 1e-10


class SpatialEngine:

    def __init__(self, uri, user, password,
                 vertex_tolerance=0.001):
        """
        Initialize spatial engine
        with Neo4j connection
        """
        self.uri              = uri
        self.user             = user
        self.password         = password
        self.vertex_tolerance = vertex_tolerance
        self.driver           = None

    def connect(self):
        """
        Connect to Neo4j database
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print("Spatial Engine connected to Neo4j")
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
    # Data Retrieval from Graph
    # ------------------------------------------

    def get_element_bbox(self, global_id):
        """
        Retrieve bounding box of element
        from graph node properties
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:IfcProduct 
                      {GlobalId: $GlobalId})
                RETURN e.xmin as xmin,
                       e.xmax as xmax,
                       e.ymin as ymin,
                       e.ymax as ymax,
                       e.zmin as zmin,
                       e.zmax as zmax
            """, GlobalId=global_id)
            record = result.single()
            if record:
                return {
                    'xmin': record['xmin'],
                    'xmax': record['xmax'],
                    'ymin': record['ymin'],
                    'ymax': record['ymax'],
                    'zmin': record['zmin'],
                    'zmax': record['zmax']
                }
            return None

    @staticmethod
    def _numeric_vid_key(vid_string):
        """
        Extract the numeric tessellation
        index from a vertex id string of
        the form "{GlobalId}_V_{idx}".

        This is REQUIRED because Neo4j's
        default ORDER BY on a string
        property sorts lexicographically
        ("V_10" < "V_2"), which silently
        misaligns the returned vertex list
        against the integer v1_idx/v2_idx/
        v3_idx indices stored on Face and
        Edge nodes, corrupting triangle
        and edge reconstruction for any
        element with 10 or more vertices.

        Returns the integer suffix, or a
        very large fallback value if the
        expected pattern is not found (so
        malformed ids sort last rather
        than silently succeeding).
        """
        try:
            suffix = vid_string.rsplit(
                '_V_', 1)[-1]
            return int(suffix)
        except (ValueError, AttributeError):
            return 10**9

    def get_element_vertices(self, global_id):
        """
        Retrieve all vertices of element
        from graph, ordered NUMERICALLY by
        their tessellation index (not by
        Neo4j's default lexicographic
        string ordering on vid).

        Correct ordering is essential
        because Face and Edge nodes
        reference vertices by integer
        index (v1_idx, v2_idx, v3_idx),
        and this list is indexed
        positionally to reconstruct
        triangles and edges elsewhere
        in this engine.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:IfcProduct 
                      {GlobalId: $GlobalId})
                      -[:HAS_VERTEX]->(v:Vertex)
                RETURN v.x as x,
                       v.y as y,
                       v.z as z,
                       v.vid as vid
            """, GlobalId=global_id)

            raw = [
                (record['vid'], np.array([
                    record['x'],
                    record['y'],
                    record['z']
                ]))
                for record in result
            ]

        raw.sort(
            key=lambda item:
                SpatialEngine._numeric_vid_key(
                    item[0]))

        return [coord for _, coord in raw]

    def get_element_faces(self, global_id):
        """
        Retrieve all faces of element
        with centroid for distance calculation
        Centroid used in COVER check
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:IfcProduct 
                      {GlobalId: $GlobalId})
                      -[:HAS_FACE]->(f:Face)
                RETURN f.normal_x   as nx,
                       f.normal_y   as ny,
                       f.normal_z   as nz,
                       f.area       as area,
                       f.v1_idx     as v1,
                       f.v2_idx     as v2,
                       f.v3_idx     as v3,
                       f.centroid_x as cx,
                       f.centroid_y as cy,
                       f.centroid_z as cz
            """, GlobalId=global_id)
            faces = []
            for record in result:
                faces.append({
                    'normal'  : np.array([
                        record['nx'],
                        record['ny'],
                        record['nz']
                    ]),
                    'area'    : record['area'],
                    'v1'      : record['v1'],
                    'v2'      : record['v2'],
                    'v3'      : record['v3'],
                    'centroid': np.array([
                        record['cx'],
                        record['cy'],
                        record['cz']
                    ])
                })
            return faces

    def get_element_edges(self, global_id):
        """
        Retrieve all edges of element
        as (v1_idx, v2_idx) index pairs
        Used for boundary/TOUCH evaluation
        and edge-face intersection tests
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:IfcProduct 
                      {GlobalId: $GlobalId})
                      -[:HAS_EDGE]->(eg:Edge)
                RETURN eg.v1_idx as v1,
                       eg.v2_idx as v2,
                       eg.length as length
            """, GlobalId=global_id)
            edges = []
            for record in result:
                edges.append({
                    'v1'    : record['v1'],
                    'v2'    : record['v2'],
                    'length': record['length']
                })
            return edges

    def get_element_geometry(self, global_id):
        """
        Fetch bbox, vertices, faces, and
        edges for an element in a single
        bundled call, plus PRECOMPUTED
        per-edge and per-face bounding
        boxes for fast pairwise rejection.

        Intended to be called ONCE per
        element per classify_relationship
        invocation and reused across all
        six predicate checks, rather than
        each check independently
        re-querying Neo4j.
        Returns a dict with keys:
        bbox, vertices, faces, edges,
        edge_bboxes, face_bboxes.
        """
        vertices = self.get_element_vertices(
            global_id)
        faces    = self.get_element_faces(
            global_id)
        edges    = self.get_element_edges(
            global_id)
        bbox     = self.get_element_bbox(
            global_id)

        edge_bboxes = []
        for edge in edges:
            p0 = vertices[edge['v1']]
            p1 = vertices[edge['v2']]
            edge_bboxes.append({
                'xmin': min(p0[0], p1[0]),
                'xmax': max(p0[0], p1[0]),
                'ymin': min(p0[1], p1[1]),
                'ymax': max(p0[1], p1[1]),
                'zmin': min(p0[2], p1[2]),
                'zmax': max(p0[2], p1[2]),
            })

        face_bboxes = []
        for face in faces:
            v1 = vertices[face['v1']]
            v2 = vertices[face['v2']]
            v3 = vertices[face['v3']]
            face_bboxes.append({
                'xmin': min(v1[0], v2[0], v3[0]),
                'xmax': max(v1[0], v2[0], v3[0]),
                'ymin': min(v1[1], v2[1], v3[1]),
                'ymax': max(v1[1], v2[1], v3[1]),
                'zmin': min(v1[2], v2[2], v3[2]),
                'zmax': max(v1[2], v2[2], v3[2]),
            })

        return {
            'bbox'       : bbox,
            'vertices'   : vertices,
            'faces'      : faces,
            'edges'      : edges,
            'edge_bboxes': edge_bboxes,
            'face_bboxes': face_bboxes,
        }

    @staticmethod
    def _bbox_overlap(bbox_a, bbox_b, tol):
        """
        Fast axis-aligned bounding box
        overlap test used to reject
        clearly-irrelevant edge/edge,
        face/face, or edge/face pairs
        before running an expensive exact
        geometric test.
        Returns True if the two boxes
        overlap (within tolerance).
        """
        return not (
            bbox_a['xmax'] < bbox_b['xmin'] - tol
            or
            bbox_b['xmax'] < bbox_a['xmin'] - tol
            or
            bbox_a['ymax'] < bbox_b['ymin'] - tol
            or
            bbox_b['ymax'] < bbox_a['ymin'] - tol
            or
            bbox_a['zmax'] < bbox_b['zmin'] - tol
            or
            bbox_b['zmax'] < bbox_a['zmin'] - tol
        )

    # ------------------------------------------
    # Cache Management
    # ------------------------------------------

    def check_cache(self, global_id_a,
                    global_id_b):
        """
        Check if relationship already
        exists in graph
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

    def store_relationship(self,
                           global_id_a,
                           global_id_b,
                           rel_type):
        """
        Store spatial relationship as
        edge in graph for future queries
        """
        with self.driver.session() as session:
            session.run(f"""
                MATCH (a:IfcProduct 
                      {{GlobalId: $id_a}})
                MATCH (b:IfcProduct 
                      {{GlobalId: $id_b}})
                MERGE (a)-[:{rel_type}]->(b)
            """,
                id_a=global_id_a,
                id_b=global_id_b)

    # ------------------------------------------
    # Geometric Primitives
    # ------------------------------------------

    def point_in_polyhedron(self, point,
                            vertices, faces,
                            face_bboxes=None):
        """
        Ray casting algorithm to check if
        point is STRICTLY inside a closed
        polyhedron (interior test).
        Casts ray in +X direction.
        Counts face intersections.
        Odd count = inside.

        If face_bboxes (precomputed via
        get_element_geometry) is provided,
        each face is first cheaply
        rejected if the point's Y or Z
        coordinate falls outside that
        face's Y/Z bounding range - since
        the ray travels only in +X, it
        cannot possibly intersect a face
        whose Y/Z extent does not bracket
        the point. This is a significant
        speedup for elements with hundreds
        of faces: this function was
        previously called, completely
        unfiltered, from nearly every
        predicate check (CONTAINS, COVERS,
        OVERLAPS, TOUCHES, DISJOINT
        fallback), making it the dominant
        cost for large elements.

        A point exactly on the boundary is
        not reliably classified by this
        test alone - use point_on_boundary
        first to distinguish boundary
        contact from true interior
        penetration.
        Returns True if point is inside.
        """
        ray_direction = np.array([1.0, 0.0, 0.0])
        intersections = 0

        for idx, face in enumerate(faces):
            if face_bboxes is not None:
                fb = face_bboxes[idx]
                if (point[1] < fb['ymin'] or
                        point[1] > fb['ymax'] or
                        point[2] < fb['zmin'] or
                        point[2] > fb['zmax']):
                    continue

            v1 = vertices[face['v1']]
            v2 = vertices[face['v2']]
            v3 = vertices[face['v3']]

            edge1 = v2 - v1
            edge2 = v3 - v1
            h     = np.cross(ray_direction, edge2)
            a     = np.dot(edge1, h)

            if abs(a) < ALGEBRAIC_EPSILON:
                continue

            f = 1.0 / a
            s = point - v1
            u = f * np.dot(s, h)

            if u < 0.0 or u > 1.0:
                continue

            q = np.cross(s, edge1)
            v = f * np.dot(ray_direction, q)

            if v < 0.0 or u + v > 1.0:
                continue

            t = f * np.dot(edge2, q)

            if t > ALGEBRAIC_EPSILON:
                intersections += 1

        return intersections % 2 == 1

    def _point_in_triangle(self, point,
                           v1, v2, v3, tol):
        """
        Bounded point-in-triangle test:
        distance-to-plane check (using the
        geometric vertex_tolerance) combined
        with a barycentric coordinate test
        (using the dimensionless
        BARYCENTRIC_EPSILON) to confirm the
        point falls within the triangle's
        actual bounded area, not merely on
        its infinite supporting plane.
        Returns True if point lies on this
        bounded triangular face.
        """
        edge1  = v2 - v1
        edge2  = v3 - v1
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)

        if norm_len < ALGEBRAIC_EPSILON:
            return False

        normal = normal / norm_len

        dist = abs(np.dot(
            normal, point - v1))
        if dist > tol:
            return False

        v0v1 = edge1
        v0v2 = edge2
        v0p  = point - v1

        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p,  v0v1)
        d21 = np.dot(v0p,  v0v2)

        denom = d00 * d11 - d01 * d01
        if abs(denom) < ALGEBRAIC_EPSILON:
            return False

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        return (u >= -BARYCENTRIC_EPSILON and
                v >= -BARYCENTRIC_EPSILON and
                w >= -BARYCENTRIC_EPSILON)

    def point_on_boundary(self, point,
                          vertices, faces,
                          tol, face_bboxes=None):
        """
        Check if a point lies on the
        boundary (surface) of a polyhedron,
        by testing against every bounded
        triangular face.

        If face_bboxes is provided, each
        face is first cheaply rejected via
        a full 3-axis bbox-vs-point check
        before running the more expensive
        bounded point-in-triangle test.

        Used to distinguish TOUCH/COVERS
        boundary contact from strict
        interior penetration.
        Returns True if point is within
        tol of any bounded triangle.
        """
        for idx, face in enumerate(faces):
            if face_bboxes is not None:
                fb = face_bboxes[idx]
                if (point[0] < fb['xmin'] - tol or
                        point[0] > fb['xmax'] + tol or
                        point[1] < fb['ymin'] - tol or
                        point[1] > fb['ymax'] + tol or
                        point[2] < fb['zmin'] - tol or
                        point[2] > fb['zmax'] + tol):
                    continue

            v1 = vertices[face['v1']]
            v2 = vertices[face['v2']]
            v3 = vertices[face['v3']]

            if self._point_in_triangle(
                    point, v1, v2, v3, tol):
                return True
        return False

    def _point_on_segment(self, point,
                          p0, p1, tol):
        """
        Check if point lies on the bounded
        line segment p0-p1, within
        geometric tolerance.
        Used for edge-edge and
        vertex-on-edge boundary contact
        checks in TOUCHES.
        """
        seg = p1 - p0
        seg_len_sq = np.dot(seg, seg)

        if seg_len_sq < ALGEBRAIC_EPSILON:
            return np.linalg.norm(
                point - p0) < tol

        t = np.dot(point - p0, seg) / \
            seg_len_sq
        t_clamped = max(0.0, min(1.0, t))
        closest = p0 + t_clamped * seg

        return np.linalg.norm(
            point - closest) < tol

    def _segments_share_boundary(
            self, a0, a1, b0, b1, tol):
        """
        Check whether two line segments
        make boundary contact: either an
        endpoint of one lies on the other
        segment, or the segments are
        collinear and overlapping.
        Used for edge-edge TOUCH evidence,
        which vertex-only matching misses
        when two elements share a boundary
        line without a coincident
        tessellation vertex.
        """
        return (
            self._point_on_segment(
                a0, b0, b1, tol) or
            self._point_on_segment(
                a1, b0, b1, tol) or
            self._point_on_segment(
                b0, a0, a1, tol) or
            self._point_on_segment(
                b1, a0, a1, tol)
        )

    def _segment_triangle_intersect(
            self, p0, p1, v1, v2, v3):
        """
        Moller-Trumbore segment-triangle
        intersection test.
        Returns True if the line segment
        p0-p1 crosses the triangle strictly
        between its endpoints (0 < t < 1).

        NOTE: this alone does not prove
        interior penetration of a solid -
        it only proves the edge crosses
        this ONE triangular face, which
        can also happen for a tangential
        graze along a boundary. For
        OVERLAPS, use
        _segment_triangle_intersect_t
        together with
        _edge_crossing_confirms_penetration
        to additionally verify a genuine
        inside/outside transition. This
        boolean-only version remains
        appropriate for CONTAINS/DISJOINT
        boundary-crossing checks, where
        any crossing at all (graze or
        true penetration) already violates
        strict containment or disjointness.
        """
        t = self._segment_triangle_intersect_t(
            p0, p1, v1, v2, v3)
        return t is not None

    def _segment_triangle_intersect_t(
            self, p0, p1, v1, v2, v3):
        """
        Moller-Trumbore segment-triangle
        intersection test, returning the
        intersection parameter t (0 < t < 1)
        along the segment p0-p1, or None
        if no intersection exists strictly
        between the segment's endpoints.
        """
        direction = p1 - p0
        edge1 = v2 - v1
        edge2 = v3 - v1
        h     = np.cross(direction, edge2)
        a     = np.dot(edge1, h)

        if abs(a) < ALGEBRAIC_EPSILON:
            return None

        f = 1.0 / a
        s = p0 - v1
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(direction, q)
        if v < 0.0 or u + v > 1.0:
            return None

        t = f * np.dot(edge2, q)
        if ALGEBRAIC_EPSILON < t < \
                1.0 - ALGEBRAIC_EPSILON:
            return t
        return None

    def _edge_crossing_confirms_penetration(
            self, p0, p1, t,
            vertices_other, faces_other,
            face_bboxes_other=None):
        """
        Given an edge p0-p1 that crosses a
        face of another solid at parameter
        t, confirm this is a genuine
        inside/outside TRANSITION (true
        interior penetration) rather than
        a tangential graze along the
        solid's boundary.

        Samples the edge slightly before
        and slightly after the crossing
        point and tests each sample point
        against the OTHER SOLID'S FULL
        volume. A true penetration
        requires exactly one sample
        STRICTLY inside and the other
        STRICTLY outside.

        This is what distinguishes
        OVERLAPS (genuine interior
        crossing) from a face-level
        intersection that is actually
        just boundary contact (TOUCHES).
        Returns True only if a genuine
        strict inside/outside transition
        is confirmed.
        """
        direction = p1 - p0
        seg_len = np.linalg.norm(direction)
        if seg_len < ALGEBRAIC_EPSILON:
            return False

        tol = self.vertex_tolerance

        step = min(
            0.01, t * 0.5, (1.0 - t) * 0.5)
        if step <= ALGEBRAIC_EPSILON:
            return False

        p_before = p0 + (t - step) * direction
        p_after  = p0 + (t + step) * direction

        def strictly_inside(p):
            return (
                self.point_in_polyhedron(
                    p, vertices_other,
                    faces_other,
                    face_bboxes_other) and
                not self.point_on_boundary(
                    p, vertices_other,
                    faces_other, tol,
                    face_bboxes_other)
            )

        def strictly_outside(p):
            return (
                not self.point_in_polyhedron(
                    p, vertices_other,
                    faces_other,
                    face_bboxes_other) and
                not self.point_on_boundary(
                    p, vertices_other,
                    faces_other, tol,
                    face_bboxes_other)
            )

        before_in  = strictly_inside(p_before)
        before_out = strictly_outside(p_before)
        after_in   = strictly_inside(p_after)
        after_out  = strictly_outside(p_after)

        return (
            (before_in and after_out) or
            (before_out and after_in)
        )

    def _triangles_coplanar_overlap(
            self, tri_a, tri_b, tol):
        """
        Check whether two triangles are
        coplanar AND their 2D projections
        genuinely overlap in AREA (not
        merely touch at a boundary).
        Returns True only for confirmed
        positive-area coplanar overlap.
        """
        a1, a2, a3 = tri_a
        b1, b2, b3 = tri_b

        normal_a = np.cross(
            a2 - a1, a3 - a1)
        norm_len_a = np.linalg.norm(normal_a)
        if norm_len_a < ALGEBRAIC_EPSILON:
            return False
        normal_a = normal_a / norm_len_a

        for p in (b1, b2, b3):
            if abs(np.dot(
                    normal_a, p - a1)) > tol:
                return False

        normal_b = np.cross(
            b2 - b1, b3 - b1)
        norm_len_b = np.linalg.norm(normal_b)
        if norm_len_b < ALGEBRAIC_EPSILON:
            return False
        normal_b = normal_b / norm_len_b

        if abs(abs(np.dot(
                normal_a, normal_b)) - 1.0
               ) > 1e-3:
            return False

        abs_n = np.abs(normal_a)
        drop_axis = int(np.argmax(abs_n))
        axes = [i for i in range(3)
                if i != drop_axis]

        def proj(p):
            return np.array(
                [p[axes[0]], p[axes[1]]])

        poly_a = [proj(a1), proj(a2),
                  proj(a3)]
        poly_b = [proj(b1), proj(b2),
                  proj(b3)]

        return self._triangles_2d_overlap(
            poly_a, poly_b, tol)

    @staticmethod
    def _triangles_2d_overlap(poly_a,
                              poly_b, tol):
        """
        2D separating-axis test (SAT) for
        two triangles, computing genuine
        PENETRATION DEPTH rather than
        merely detecting the absence of a
        gap - a plain SAT gap-check treats
        boundary-only touching the same as
        genuine overlap, which is
        incorrect. Only a strictly
        positive minimum overlap greater
        than tol on every axis confirms
        genuine positive-area intersection.
        Returns True only for genuine
        positive-area intersection.
        """
        def edges(poly):
            return [
                poly[(i + 1) % len(poly)]
                - poly[i]
                for i in range(len(poly))
            ]

        def project(poly, axis):
            dots = [np.dot(p, axis)
                    for p in poly]
            return min(dots), max(dots)

        min_penetration = None

        for poly in (poly_a, poly_b):
            for e in edges(poly):
                axis = np.array(
                    [-e[1], e[0]])
                norm = np.linalg.norm(axis)
                if norm < ALGEBRAIC_EPSILON:
                    continue
                axis = axis / norm

                min_a, max_a = project(
                    poly_a, axis)
                min_b, max_b = project(
                    poly_b, axis)

                overlap = min(max_a, max_b) \
                    - max(min_a, min_b)

                if overlap <= tol:
                    return False

                if (min_penetration is None
                        or overlap <
                        min_penetration):
                    min_penetration = overlap

        return min_penetration is not None

    # ------------------------------------------
    # Six Spatial Relationship Checks
    #
    # Each accepts optional pre-fetched
    # geometry dicts (geo_a, geo_b, as
    # returned by get_element_geometry) to
    # avoid redundant Neo4j round trips.
    # Every nested edge/face pairwise loop
    # and every point_in_polyhedron /
    # point_on_boundary call is bbox
    # pre-filtered using the precomputed
    # edge_bboxes/face_bboxes.
    # ------------------------------------------

    def check_disjoint_bbox(self, bbox_a,
                            bbox_b):
        """
        Fast DISJOINT pre-filter using
        axis-aligned bounding boxes.
        This is a NECESSARY, not
        sufficient, condition - only a
        POSITIVE confirmation when the
        boxes provably do not overlap.
        Returns True only if bounding
        boxes are provably non-overlapping.
        """
        return not self._bbox_overlap(
            bbox_a, bbox_b,
            self.vertex_tolerance)

    def _resolve_geo(self, global_id, geo):
        if geo is not None:
            return geo
        return self.get_element_geometry(
            global_id)

    def check_equal(self, global_id_a,
                    global_id_b,
                    geo_a=None, geo_b=None):
        """
        EQUALS: int(A)=int(B), boundary(A)
        =boundary(B), ext(A)=ext(B).

        Checks boundary CORRESPONDENCE:
        every vertex of A must lie on the
        boundary of B, and every vertex of
        B must lie on the boundary of A.

        Note: this remains an approximation
        for curved/continuous boundary
        correspondence between arbitrarily
        different tessellations, evaluated
        at discrete sample points rather
        than via continuous boundary
        integration; documented as a known
        limitation.
        Returns True if EQUAL.
        """
        ga = self._resolve_geo(
            global_id_a, geo_a)
        gb = self._resolve_geo(
            global_id_b, geo_b)

        vertices_a, faces_a = \
            ga['vertices'], ga['faces']
        vertices_b, faces_b = \
            gb['vertices'], gb['faces']

        if (not vertices_a or not faces_a or
                not vertices_b or not faces_b):
            return False

        tol = self.vertex_tolerance

        for va in vertices_a:
            if not self.point_on_boundary(
                    va, vertices_b, faces_b,
                    tol, gb['face_bboxes']):
                return False

        for vb in vertices_b:
            if not self.point_on_boundary(
                    vb, vertices_a, faces_a,
                    tol, ga['face_bboxes']):
                return False

        return True

    def check_contain(self, global_id_a,
                      global_id_b,
                      geo_a=None, geo_b=None):
        """
        CONTAINS: int(B) subset int(A),
        boundary(B) does not intersect
        boundary(A), ext(A) does not
        intersect int(B) or boundary(B).

        Every vertex of B must be strictly
        inside A (not on A's boundary),
        AND no edge of B may cross any
        face of A (bbox pre-filtered).
        Returns True if A CONTAINS B.
        """
        gb = self._resolve_geo(
            global_id_b, geo_b)
        vertices_b = gb['vertices']
        edges_b    = gb['edges']
        edge_bboxes_b = gb['edge_bboxes']

        if not vertices_b:
            return False

        ga = self._resolve_geo(
            global_id_a, geo_a)
        vertices_a = ga['vertices']
        faces_a    = ga['faces']
        face_bboxes_a = ga['face_bboxes']

        if not vertices_a or not faces_a:
            return False

        tol = self.vertex_tolerance

        for vertex in vertices_b:
            if self.point_on_boundary(
                    vertex, vertices_a,
                    faces_a, tol,
                    face_bboxes_a):
                return False
            if not self.point_in_polyhedron(
                    vertex, vertices_a,
                    faces_a, face_bboxes_a):
                return False

        for i, edge in enumerate(edges_b):
            eb = edge_bboxes_b[i]
            p0 = vertices_b[edge['v1']]
            p1 = vertices_b[edge['v2']]
            for j, face in enumerate(faces_a):
                fb = face_bboxes_a[j]
                if not self._bbox_overlap(
                        eb, fb, tol):
                    continue
                fv1 = vertices_a[face['v1']]
                fv2 = vertices_a[face['v2']]
                fv3 = vertices_a[face['v3']]
                if self._segment_triangle_intersect(
                        p0, p1, fv1, fv2, fv3):
                    return False

        return True

    def check_cover(self, global_id_a,
                    global_id_b,
                    geo_a=None, geo_b=None):
        """
        COVERS: boundary(B) subset
        (int(A) union boundary(A)),
        ext(A) does not intersect
        int(B) or boundary(B), and
        int(A) intersects int(B).

        Every vertex of B must be either
        strictly inside A or on A's
        boundary, AND at least one genuine
        boundary-contact point must exist.
        Returns True if A COVERS B.
        """
        gb = self._resolve_geo(
            global_id_b, geo_b)
        vertices_b = gb['vertices']

        ga = self._resolve_geo(
            global_id_a, geo_a)
        vertices_a = ga['vertices']
        faces_a    = ga['faces']
        face_bboxes_a = ga['face_bboxes']

        if (not vertices_b or not faces_a
                or not vertices_a):
            return False

        tol = self.vertex_tolerance
        has_boundary_contact = False

        for vb in vertices_b:
            on_boundary = self.point_on_boundary(
                vb, vertices_a, faces_a, tol,
                face_bboxes_a)

            if on_boundary:
                has_boundary_contact = True
                continue

            inside = self.point_in_polyhedron(
                vb, vertices_a, faces_a,
                face_bboxes_a)

            if not inside:
                return False

        return has_boundary_contact

    def check_overlap(self, global_id_a,
                      global_id_b,
                      geo_a=None, geo_b=None):
        """
        OVERLAPS: int(A) intersects
        int(B), A not subset B, B not
        subset A, ext(A) intersects
        int(B), int(A) intersects ext(B).

        Requires genuine INTERIOR-INTERIOR
        intersection:

        1. Mutual STRICT vertex-in-volume
           (confirmed inside AND
           confirmed not on the boundary,
           both directions) - boundary
           exclusion is essential since
           ray-casting parity is
           unreliable exactly at a
           boundary, which previously
           caused flush-contact
           connections to be
           misclassified as OVERLAPS.
        2. Edge-triangle crossings
           CONFIRMED as genuine
           inside/outside transitions,
           bbox pre-filtered on both
           the edge-vs-face candidate
           search.

        Coplanar face overlap is
        deliberately NOT used as OVERLAPS
        evidence - two solids resting
        flush can share coincident face
        area while interiors remain
        disjoint, which is TOUCHES, not
        OVERLAPS (handled in check_touch).

        CONTAINS, COVERS, and EQUALS are
        implicitly excluded by the
        classification order.
        Returns True if OVERLAP.
        """
        ga = self._resolve_geo(
            global_id_a, geo_a)
        gb = self._resolve_geo(
            global_id_b, geo_b)

        vertices_a = ga['vertices']
        faces_a    = ga['faces']
        edges_a    = ga['edges']
        edge_bboxes_a = ga['edge_bboxes']
        face_bboxes_a = ga['face_bboxes']
        vertices_b = gb['vertices']
        faces_b    = gb['faces']
        edges_b    = gb['edges']
        edge_bboxes_b = gb['edge_bboxes']
        face_bboxes_b = gb['face_bboxes']

        if (not vertices_a or not faces_a or
                not vertices_b or not faces_b):
            return False

        tol = self.vertex_tolerance

        any_b_strictly_in_a = any(
            self.point_in_polyhedron(
                v, vertices_a, faces_a,
                face_bboxes_a) and
            not self.point_on_boundary(
                v, vertices_a, faces_a, tol,
                face_bboxes_a)
            for v in vertices_b)
        any_a_strictly_in_b = any(
            self.point_in_polyhedron(
                v, vertices_b, faces_b,
                face_bboxes_b) and
            not self.point_on_boundary(
                v, vertices_b, faces_b, tol,
                face_bboxes_b)
            for v in vertices_a)

        if (any_b_strictly_in_a and
                any_a_strictly_in_b):
            return True

        # Edge-triangle crossings,
        # bbox pre-filtered
        for i, edge in enumerate(edges_a):
            eb = edge_bboxes_a[i]
            p0 = vertices_a[edge['v1']]
            p1 = vertices_a[edge['v2']]
            for j, face in enumerate(faces_b):
                fb = face_bboxes_b[j]
                if not self._bbox_overlap(
                        eb, fb, tol):
                    continue
                fv1 = vertices_b[face['v1']]
                fv2 = vertices_b[face['v2']]
                fv3 = vertices_b[face['v3']]
                t = self._segment_triangle_intersect_t(
                    p0, p1, fv1, fv2, fv3)
                if t is None:
                    continue
                if self._edge_crossing_confirms_penetration(
                        p0, p1, t,
                        vertices_b, faces_b,
                        face_bboxes_b):
                    return True

        for i, edge in enumerate(edges_b):
            eb = edge_bboxes_b[i]
            p0 = vertices_b[edge['v1']]
            p1 = vertices_b[edge['v2']]
            for j, face in enumerate(faces_a):
                fb = face_bboxes_a[j]
                if not self._bbox_overlap(
                        eb, fb, tol):
                    continue
                fv1 = vertices_a[face['v1']]
                fv2 = vertices_a[face['v2']]
                fv3 = vertices_a[face['v3']]
                t = self._segment_triangle_intersect_t(
                    p0, p1, fv1, fv2, fv3)
                if t is None:
                    continue
                if self._edge_crossing_confirms_penetration(
                        p0, p1, t,
                        vertices_a, faces_a,
                        face_bboxes_a):
                    return True

        return False

    def check_touch(self, global_id_a,
                    global_id_b,
                    geo_a=None, geo_b=None):
        """
        TOUCHES: boundary(A) intersects
        boundary(B), int(A) does not
        intersect int(B).

        Boundary contact is detected using
        vertices, edges, AND faces:
          - vertex-vertex coincidence
          - vertex-of-A on an edge of B
            (and vice versa)
          - edge-of-A touching an edge
            of B (bbox pre-filtered)
          - vertex-of-A on a face of B
            (and vice versa, bbox
            pre-filtered)
          - coplanar positive-area face
            overlap (bbox pre-filtered)

        Interior-disjointness is then
        explicitly verified.

        Note: TOUCHES is evaluated LAST in
        classify_relationship, after
        CONTAINS/COVERS/OVERLAPS/EQUALS
        have already been ruled out.
        Returns True if TOUCH.
        """
        ga = self._resolve_geo(
            global_id_a, geo_a)
        gb = self._resolve_geo(
            global_id_b, geo_b)

        vertices_a = ga['vertices']
        faces_a    = ga['faces']
        edges_a    = ga['edges']
        edge_bboxes_a = ga['edge_bboxes']
        face_bboxes_a = ga['face_bboxes']
        vertices_b = gb['vertices']
        faces_b    = gb['faces']
        edges_b    = gb['edges']
        edge_bboxes_b = gb['edge_bboxes']
        face_bboxes_b = gb['face_bboxes']

        if (not vertices_a or not vertices_b):
            return False

        tol = self.vertex_tolerance
        boundary_contact = False

        # 1. Vertex-vertex coincidence
        for va in vertices_a:
            for vb in vertices_b:
                if np.linalg.norm(
                        va - vb) < tol:
                    boundary_contact = True
                    break
            if boundary_contact:
                break

        # 2. Vertex-on-edge contact
        if not boundary_contact and edges_b:
            for va in vertices_a:
                for edge in edges_b:
                    p0 = vertices_b[edge['v1']]
                    p1 = vertices_b[edge['v2']]
                    if self._point_on_segment(
                            va, p0, p1, tol):
                        boundary_contact = True
                        break
                if boundary_contact:
                    break

        if not boundary_contact and edges_a:
            for vb in vertices_b:
                for edge in edges_a:
                    p0 = vertices_a[edge['v1']]
                    p1 = vertices_a[edge['v2']]
                    if self._point_on_segment(
                            vb, p0, p1, tol):
                        boundary_contact = True
                        break
                if boundary_contact:
                    break

        # 3. Edge-edge contact,
        # bbox pre-filtered
        if (not boundary_contact and
                edges_a and edges_b):
            for i, ea in enumerate(edges_a):
                eb_a = edge_bboxes_a[i]
                a0 = vertices_a[ea['v1']]
                a1 = vertices_a[ea['v2']]
                for j, eb in enumerate(edges_b):
                    eb_b = edge_bboxes_b[j]
                    if not self._bbox_overlap(
                            eb_a, eb_b, tol):
                        continue
                    b0 = vertices_b[eb['v1']]
                    b1 = vertices_b[eb['v2']]
                    if self._segments_share_boundary(
                            a0, a1, b0, b1, tol):
                        boundary_contact = True
                        break
                if boundary_contact:
                    break

        # 4. Vertex-on-face contact,
        # bbox pre-filtered inside
        # point_on_boundary
        if not boundary_contact and faces_b:
            for va in vertices_a:
                if self.point_on_boundary(
                        va, vertices_b,
                        faces_b, tol,
                        face_bboxes_b):
                    boundary_contact = True
                    break

        if not boundary_contact and faces_a:
            for vb in vertices_b:
                if self.point_on_boundary(
                        vb, vertices_a,
                        faces_a, tol,
                        face_bboxes_a):
                    boundary_contact = True
                    break

        # 5. Coplanar positive-area face
        # overlap, bbox pre-filtered
        if (not boundary_contact and
                faces_a and faces_b):
            for i, face_a in enumerate(faces_a):
                fb_a = face_bboxes_a[i]
                tri_a = (
                    vertices_a[face_a['v1']],
                    vertices_a[face_a['v2']],
                    vertices_a[face_a['v3']]
                )
                for j, face_b in enumerate(faces_b):
                    fb_b = face_bboxes_b[j]
                    if not self._bbox_overlap(
                            fb_a, fb_b, tol):
                        continue
                    tri_b = (
                        vertices_b[face_b['v1']],
                        vertices_b[face_b['v2']],
                        vertices_b[face_b['v3']]
                    )
                    if self._triangles_coplanar_overlap(
                            tri_a, tri_b, tol):
                        boundary_contact = True
                        break
                if boundary_contact:
                    break

        if not boundary_contact:
            return False

        # Interior-disjointness
        # confirmation
        if not faces_a or not faces_b:
            return True

        any_a_in_b = any(
            self.point_in_polyhedron(
                v, vertices_b, faces_b,
                face_bboxes_b)
            for v in vertices_a)
        any_b_in_a = any(
            self.point_in_polyhedron(
                v, vertices_a, faces_a,
                face_bboxes_a)
            for v in vertices_b)

        interiors_disjoint = not (
            any_a_in_b or any_b_in_a)

        return interiors_disjoint

    def check_disjoint_full(self,
                            global_id_a,
                            global_id_b,
                            geo_a=None,
                            geo_b=None):
        """
        Explicit DISJOINT verification for
        the case where bounding boxes
        overlap but no other predicate was
        confirmed by the classification
        pipeline.

        Re-confirms the absence of every
        intersection type: shared boundary
        vertices, vertex-in-volume (both
        directions), and edge-face
        intersection (both directions,
        bbox pre-filtered).
        Returns True if DISJOINT is
        positively confirmed; False if any
        residual contact is found (caller
        falls back to TOUCHES).
        """
        ga = self._resolve_geo(
            global_id_a, geo_a)
        gb = self._resolve_geo(
            global_id_b, geo_b)

        vertices_a = ga['vertices']
        faces_a    = ga['faces']
        edges_a    = ga['edges']
        face_bboxes_a = ga['face_bboxes']
        edge_bboxes_a = ga['edge_bboxes']
        vertices_b = gb['vertices']
        faces_b    = gb['faces']
        edges_b    = gb['edges']
        face_bboxes_b = gb['face_bboxes']
        edge_bboxes_b = gb['edge_bboxes']

        if (not vertices_a or not vertices_b):
            return False

        tol = self.vertex_tolerance

        for va in vertices_a:
            for vb in vertices_b:
                if np.linalg.norm(
                        va - vb) < tol:
                    return False

        if faces_a and faces_b:
            if any(self.point_in_polyhedron(
                    v, vertices_b, faces_b,
                    face_bboxes_b)
                   for v in vertices_a):
                return False
            if any(self.point_in_polyhedron(
                    v, vertices_a, faces_a,
                    face_bboxes_a)
                   for v in vertices_b):
                return False

        if faces_b:
            for i, edge in enumerate(edges_a):
                eb_a = edge_bboxes_a[i]
                p0 = vertices_a[edge['v1']]
                p1 = vertices_a[edge['v2']]
                for j, face in enumerate(faces_b):
                    fb_b = face_bboxes_b[j]
                    if not self._bbox_overlap(
                            eb_a, fb_b, tol):
                        continue
                    fv1 = vertices_b[face['v1']]
                    fv2 = vertices_b[face['v2']]
                    fv3 = vertices_b[face['v3']]
                    if self._segment_triangle_intersect(
                            p0, p1, fv1, fv2, fv3):
                        return False

        if faces_a:
            for i, edge in enumerate(edges_b):
                eb_b = edge_bboxes_b[i]
                p0 = vertices_b[edge['v1']]
                p1 = vertices_b[edge['v2']]
                for j, face in enumerate(faces_a):
                    fb_a = face_bboxes_a[j]
                    if not self._bbox_overlap(
                            eb_b, fb_a, tol):
                        continue
                    fv1 = vertices_a[face['v1']]
                    fv2 = vertices_a[face['v2']]
                    fv3 = vertices_a[face['v3']]
                    if self._segment_triangle_intersect(
                            p0, p1, fv1, fv2, fv3):
                        return False

        return True

    # ------------------------------------------
    # Master Classification Function
    # ------------------------------------------

    def classify_relationship(self,
                              global_id_a,
                              global_id_b):
        """
        Master function to classify spatial
        relationship between two elements.

        Geometry (vertices, faces, edges,
        bbox, plus precomputed edge/face
        bounding boxes) is fetched ONCE per
        element and reused across all
        subsequent predicate checks.

        Evaluation order (aligned with the
        formal DE-9IM definitions in
        Table 4):

          1. Cache lookup              - O(1)
          2. DISJOINT bbox pre-filter  - only
             a POSITIVE confirmation when
             boxes provably do not overlap
          3. EQUALS
          4. CONTAINS
          5. COVERS
          6. OVERLAPS
          7. TOUCHES                   - last,
             the most permissive predicate
          8. DISJOINT explicit
             verification - only reached
             if bounding boxes overlap but
             no positive predicate was
             confirmed

        Caches result in graph.
        Returns relationship type string.
        """
        print(f"\nClassifying relationship:")
        print(f"  A: {global_id_a}")
        print(f"  B: {global_id_b}")

        cached = self.check_cache(
            global_id_a, global_id_b)
        if cached:
            print(f"  Result (cached): {cached}")
            return cached

        bbox_a = self.get_element_bbox(
            global_id_a)
        bbox_b = self.get_element_bbox(
            global_id_b)

        if not bbox_a or not bbox_b:
            print("  Error: element not found")
            return None

        if self.check_disjoint_bbox(
                bbox_a, bbox_b):
            self.store_relationship(
                global_id_a, global_id_b,
                'DISJOINT')
            print(f"  Result: DISJOINT "
                  f"(bbox non-overlap)")
            return 'DISJOINT'

        geo_a = self.get_element_geometry(
            global_id_a)
        geo_b = self.get_element_geometry(
            global_id_b)

        if self.check_equal(
                global_id_a, global_id_b,
                geo_a, geo_b):
            self.store_relationship(
                global_id_a, global_id_b,
                'EQUALS')
            print(f"  Result: EQUALS")
            return 'EQUALS'

        if self.check_contain(
                global_id_a, global_id_b,
                geo_a, geo_b):
            self.store_relationship(
                global_id_a, global_id_b,
                'CONTAINS')
            print(f"  Result: CONTAINS")
            return 'CONTAINS'

        if self.check_cover(
                global_id_a, global_id_b,
                geo_a, geo_b):
            self.store_relationship(
                global_id_a, global_id_b,
                'COVERS')
            print(f"  Result: COVERS")
            return 'COVERS'

        if self.check_overlap(
                global_id_a, global_id_b,
                geo_a, geo_b):
            self.store_relationship(
                global_id_a, global_id_b,
                'OVERLAPS')
            print(f"  Result: OVERLAPS")
            return 'OVERLAPS'

        if self.check_touch(
                global_id_a, global_id_b,
                geo_a, geo_b):
            self.store_relationship(
                global_id_a, global_id_b,
                'TOUCHES')
            print(f"  Result: TOUCHES")
            return 'TOUCHES'

        confirmed_disjoint = \
            self.check_disjoint_full(
                global_id_a, global_id_b,
                geo_a, geo_b)

        if confirmed_disjoint:
            self.store_relationship(
                global_id_a, global_id_b,
                'DISJOINT')
            print(f"  Result: DISJOINT "
                  f"(explicitly verified)")
            return 'DISJOINT'

        self.store_relationship(
            global_id_a, global_id_b,
            'TOUCHES')
        print(f"  Result: TOUCHES "
              f"(residual boundary contact, "
              f"conservative fallback)")
        return 'TOUCHES'


# Quick test
if __name__ == "__main__":

    import sys
    sys.path.append(
        r"C:\USB data\articles ideas\Brep Graph"
        r"\System development\BRepGraph_Article1"
        r"\BRepGraph_Article1")

    import config

    engine = SpatialEngine(
        uri              = config.NEO4J_URI,
        user             = config.NEO4J_USER,
        password         = config.NEO4J_PASSWORD,
        vertex_tolerance = config.VERTEX_TOLERANCE)

    if engine.connect():

        with engine.driver.session() as session:
            session.run("""
                MATCH ()-[r:DISJOINT|TOUCHES|
                          OVERLAPS|CONTAINS|
                          COVERS|EQUALS]->()
                DELETE r
            """)
            print("Old cached relationships "
                  "cleared")

        print("\n=== Test: COVER - Wall vs Slab ===")
        with engine.driver.session() as session:
            res = session.run("""
                MATCH (s:IfcSlab 
                      {name:'Bodenplatte'})
                RETURN s.GlobalId as gid
            """)
            record = res.single()
            slab_id = record['gid'] if record \
                else None

        if slab_id:
            start = time.time()
            r1 = engine.classify_relationship(
                '2XPyKWY018sA1ygZKgQPtU', slab_id)
            print(f"  Wall vs Slab: {r1}")
            print(f"  Time: "
                  f"{time.time()-start:.3f}s")

        engine.close()

    input("\nPress Enter to exit...")
