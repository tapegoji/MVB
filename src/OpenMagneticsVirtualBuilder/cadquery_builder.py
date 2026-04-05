"""CadQuery rendering engine for OpenMagneticsVirtualBuilder.

Implements all 21 core shape families, bobbin, and winding geometry using
the CadQuery Python CAD scripting library.  This engine requires no system
CAD installation and is suitable for headless / CI environments.

Shape classes are nested inside :class:`CadQueryBuilder` and follow the
EN 60205 naming convention.
"""

import sys
import math
import os
import json
from abc import ABCMeta, abstractmethod
import copy
import pathlib
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Any

sys.path.append(os.path.dirname(__file__))
import utils
import shape_configs
import cadquery as cq

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


# ==========================================================================
# Global Configuration for Tessellation Quality
# ==========================================================================

# Number of segments per full circle for curved surfaces in STL export
# Lower values = fewer polygons, faster rendering, smaller files
# Higher values = smoother curves, more polygons, larger files
# Default: 20 segments per circle (18° per segment)
TESSELLATION_SEGMENTS_PER_CIRCLE = 20


# Angular tolerance in degrees for STL tessellation
# This is derived from TESSELLATION_SEGMENTS_PER_CIRCLE
def get_angular_tolerance():
    """Get angular tolerance in radians based on segments per circle."""
    return 2 * math.pi / TESSELLATION_SEGMENTS_PER_CIRCLE


# Linear tolerance for STL tessellation (chord deviation)
# Smaller values = more accurate but more polygons
TESSELLATION_LINEAR_TOLERANCE = 0.1  # mm


def set_tessellation_quality(segments_per_circle: int = 20, linear_tolerance: float = 0.1):
    """Configure the tessellation quality for STL export.

    Args:
        segments_per_circle: Number of segments per full circle (default: 20).
            - 8-12: Very coarse, good for previews
            - 16-24: Medium quality, good balance
            - 32-48: High quality, smooth curves
            - 64+: Very high quality, large files
        linear_tolerance: Maximum chord deviation in mm (default: 0.1).
            Smaller values = more accurate but more polygons.

    Example:
        # Coarse quality for fast previews
        set_tessellation_quality(segments_per_circle=12)

        # High quality for final renders
        set_tessellation_quality(segments_per_circle=48, linear_tolerance=0.01)
    """
    global TESSELLATION_SEGMENTS_PER_CIRCLE, TESSELLATION_LINEAR_TOLERANCE
    TESSELLATION_SEGMENTS_PER_CIRCLE = segments_per_circle
    TESSELLATION_LINEAR_TOLERANCE = linear_tolerance


# ==========================================================================
# Enums and Data Classes for Magnetic Building
# ==========================================================================


class WireType(Enum):
    """Wire types supported."""

    round = "round"
    litz = "litz"
    rectangular = "rectangular"
    foil = "foil"
    planar = "planar"


class ColumnShape(Enum):
    """Bobbin column shapes."""

    round = "round"
    rectangular = "rectangular"
    oblong = "oblong"  # Stadium shape (rectangle with semicircular ends)
    epx = "epx"  # Stadium shape with one side flat (like EPX cores)


def resolve_dimensional_value(value: Any) -> float:
    """Extract numeric value from dimensional data (handles dict with 'nominal' or plain value)."""
    if value is None:
        return 0.0
    if isinstance(value, dict):
        return value.get("nominal", value.get("minimum", value.get("maximum", 0.0)))
    return float(value)


@dataclass
class WireDescription:
    """Description of a wire."""

    wire_type: WireType
    conducting_diameter: Optional[float] = None
    outer_diameter: Optional[float] = None
    conducting_width: Optional[float] = None
    conducting_height: Optional[float] = None
    outer_width: Optional[float] = None
    outer_height: Optional[float] = None
    number_conductors: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "WireDescription":
        wire_type_str = data.get("type", "round")
        wire_type = WireType[wire_type_str] if isinstance(wire_type_str, str) else wire_type_str

        return cls(
            wire_type=wire_type,
            conducting_diameter=resolve_dimensional_value(data.get("conductingDiameter")),
            outer_diameter=resolve_dimensional_value(data.get("outerDiameter")),
            conducting_width=resolve_dimensional_value(data.get("conductingWidth")),
            conducting_height=resolve_dimensional_value(data.get("conductingHeight")),
            outer_width=resolve_dimensional_value(data.get("outerWidth")),
            outer_height=resolve_dimensional_value(data.get("outerHeight")),
            number_conductors=data.get("numberConductors", 1),
        )


@dataclass
class TurnDescription:
    """Description of a single turn."""

    coordinates: List[float]
    winding: str = ""
    section: str = ""
    layer: str = ""
    parallel: int = 0
    turn_index: int = 0
    dimensions: Optional[List[float]] = None
    rotation: float = 0.0
    additional_coordinates: Optional[List[List[float]]] = None
    cross_sectional_shape: str = "round"

    @classmethod
    def from_dict(cls, data: dict) -> "TurnDescription":
        return cls(
            coordinates=data.get("coordinates", [0, 0]),
            winding=data.get("winding", ""),
            section=data.get("section", ""),
            layer=data.get("layer", ""),
            parallel=data.get("parallel", 0),
            turn_index=data.get("turnIndex", 0),
            dimensions=data.get("dimensions"),
            rotation=data.get("rotation", 0.0),
            additional_coordinates=data.get("additionalCoordinates"),
            cross_sectional_shape=data.get("crossSectionalShape", "round"),
        )


@dataclass
class BobbinProcessedDescription:
    """Processed bobbin description."""

    column_depth: float = 0.0
    column_width: float = 0.0
    column_thickness: float = 0.0
    wall_thickness: float = 0.0
    column_shape: ColumnShape = ColumnShape.rectangular
    winding_window_height: float = 0.0
    winding_window_width: float = 0.0
    winding_window_radial_height: float = 0.0
    winding_window_angle: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "BobbinProcessedDescription":
        shape_str = data.get("columnShape", "rectangular")
        if isinstance(shape_str, str):
            column_shape = ColumnShape[shape_str] if shape_str in ColumnShape.__members__ else ColumnShape.rectangular
        else:
            column_shape = shape_str

        # Get winding window info
        ww_height = 0.0
        ww_width = 0.0
        ww_radial_height = 0.0
        ww_angle = None

        winding_windows = data.get("windingWindows", [])
        if winding_windows and len(winding_windows) > 0:
            ww = winding_windows[0]
            ww_height = ww.get("height", 0.0)
            ww_width = ww.get("width", 0.0)
            ww_radial_height = ww.get("radialHeight", 0.0)
            ww_angle = ww.get("angle")

        return cls(
            column_depth=data.get("columnDepth", 0.0),
            column_width=data.get("columnWidth", 0.0),
            column_thickness=data.get("columnThickness", 0.0),
            wall_thickness=data.get("wallThickness", 0.0),
            column_shape=column_shape,
            winding_window_height=ww_height,
            winding_window_width=ww_width,
            winding_window_radial_height=ww_radial_height,
            winding_window_angle=ww_angle,
        )


def flatten_dimensions(data):
    return utils.flatten_dimensions(data, scale_factor=1.0)


def convert_axis(coordinates):
    # MAS coordinates are [x, y, z] — no swapping needed
    if len(coordinates) == 2:
        return [coordinates[0], coordinates[1], 0]
    elif len(coordinates) == 3:
        return [coordinates[0], coordinates[1], coordinates[2]]
    else:
        raise AssertionError("Invalid coordinates length")


class CadQueryBuilder(utils.BuilderBase):
    """Builder for 3D magnetic component geometry using CadQuery.

    This class creates 3D geometry for magnetic components including:
    - Core shapes (E, ETD, PQ, RM, toroidal, etc.)
    - Coil turns (concentric and toroidal winding styles)
    - Bobbins

    Coordinate System (MAS to CadQuery mapping):
    - For concentric cores (E, PQ, RM, etc.):
        - X axis: Core depth direction (perpendicular to winding window)
        - Y axis: Core width direction (radial, distance from central column)
        - Z axis: Core height direction (along core axis, vertical)
        - MAS coordinates[0] (radial) -> Y position
        - MAS coordinates[1] (height) -> Z position

    - For toroidal cores:
        - Y axis: Core axis (toroid revolves around Y)
        - X axis: Radial direction (negative X = inside the donut hole)
        - Z axis: Tangential direction (along circumference at Y=0)
        - MAS coordinates[0] (radial) -> distance from Y axis
        - MAS coordinates[1] (angular) -> rotation angle around Y axis

    Units:
    - All MAS input values are in meters
    - Internal geometry is built in millimeters for precision
    - Output is scaled back to meters before export
    """

    # Scale factor: build geometry in mm, scale back to meters for output
    SCALE = 1000.0

    # Number of segments for round wire polygon approximation.
    # Polygonal profiles produce flat-faced pipe sweeps that OCC can
    # fragment cleanly, enabling gmsh meshing of many-turn coils.
    # 8 segments ≈ 10% area error vs perfect circle, but produces larger
    # polygon faces (side ≈ 0.383*d) which gmsh can mesh without invalid elements.
    # 16 segments gave 0.087mm sides for 0.447mm Litz conducting diameter, causing
    # "elements remain invalid in surface" and 3D mesh failure.
    WIRE_POLYGON_SEGMENTS = 8
    CORE_POLYGON_SEGMENTS = 16  # Polygon approximation for round core features (center post, winding window)

    @staticmethod
    def polygon_cylinder(height, radius, n_segments, workplane="XY"):
        """Create a polygon-approximated cylinder (prism) instead of a NURBS cylinder.

        Eliminates curved core surfaces that cause meshing conflicts with polygon wire
        cross-sections near the air gap. All surfaces become flat facets.
        """
        import cadquery as cq
        pts = []
        for i in range(n_segments):
            angle = 2 * math.pi * i / n_segments
            pts.append((radius * math.cos(angle), radius * math.sin(angle)))
        pts.append(pts[0])  # close the polygon
        return (
            cq.Workplane(workplane)
            .polyline(pts)
            .close()
            .extrude(height / 2, both=True)
        )

    def __init__(self):
        self.shapers = {
            utils.ShapeFamily.ETD: self.Etd(),
            utils.ShapeFamily.ER: self.Er(),
            utils.ShapeFamily.EP: self.Ep(),
            utils.ShapeFamily.EPX: self.Epx(),
            utils.ShapeFamily.PQ: self.Pq(),
            utils.ShapeFamily.E: self.E(),
            utils.ShapeFamily.PM: self.Pm(),
            utils.ShapeFamily.P: self.P(),
            utils.ShapeFamily.RM: self.Rm(),
            utils.ShapeFamily.EQ: self.Eq(),
            utils.ShapeFamily.LP: self.Lp(),
            utils.ShapeFamily.PLANAR_ER: self.Er(),
            utils.ShapeFamily.PLANAR_E: self.E(),
            utils.ShapeFamily.PLANAR_EL: self.El(),
            utils.ShapeFamily.EC: self.Ec(),
            utils.ShapeFamily.EFD: self.Efd(),
            utils.ShapeFamily.U: self.U(),
            utils.ShapeFamily.UR: self.Ur(),
            utils.ShapeFamily.T: self.T(),
            utils.ShapeFamily.UT: self.Ut(),
            utils.ShapeFamily.C: self.C(),
        }

    def get_spacer(self, geometrical_data):
        spacer = cq.Workplane().box(geometrical_data["dimensions"][0], geometrical_data["dimensions"][2], geometrical_data["dimensions"][1]).translate(convert_axis(geometrical_data["coordinates"]))
        return spacer

    def get_core(self, project_name, geometrical_description, output_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../output/", save_files=True, export_files=True):
        try:
            pieces_to_export = []
            project_name = f"{project_name}_core".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

            os.makedirs(output_path, exist_ok=True)

            for index, geometrical_part in enumerate(geometrical_description):
                # Skip spacers - they are built separately via get_spacers()
                if geometrical_part['type'] == 'spacer':
                    continue
                elif geometrical_part['type'] in ['half set', 'toroidal']:
                    shape_data = geometrical_part['shape']
                    part_builder = CadQueryBuilder().factory(shape_data)

                    piece = part_builder.get_piece(data=copy.deepcopy(shape_data), name=f"Piece_{index}", save_files=False, export_files=False)

                    # rotation[0] around X, rotation[1] around Y, rotation[2] around Z
                    piece = piece.rotate((1, 0, 0), (-1, 0, 0), geometrical_part['rotation'][0] / math.pi * 180)
                    piece = piece.rotate((0, 1, 0), (0, -1, 0), geometrical_part['rotation'][1] / math.pi * 180)
                    piece = piece.rotate((0, 0, 1), (0, 0, -1), geometrical_part['rotation'][2] / math.pi * 180)

                    if "machining" in geometrical_part and geometrical_part["machining"] is not None:
                        for machining in geometrical_part["machining"]:
                            piece = part_builder.apply_machining(piece=piece, machining=machining, dimensions=flatten_dimensions(shape_data))

                    piece = piece.translate(convert_axis(geometrical_part["coordinates"]))

                    # if the piece is half a set, we add a residual gap between the pieces
                    if geometrical_part["type"] in ["half set"]:
                        residual_gap = 5e-6
                        if geometrical_part["rotation"][0] > 0:
                            piece = piece.translate((0, 0, residual_gap / 2))
                        else:
                            piece = piece.translate((0, 0, -residual_gap / 2))

                    pieces_to_export.append(piece)

            if export_files:
                from cadquery import exporters

                scaled_pieces_to_export = []
                for piece in pieces_to_export:
                    for o in piece.objects:
                        scaled_pieces_to_export.append(o.scale(1000))

                scaled_pieces_to_export = cq.Compound.makeCompound(scaled_pieces_to_export)

                exporters.export(scaled_pieces_to_export, f"{output_path}/{project_name}.step", "STEP")
                # Use configurable tessellation parameters for STL
                exporters.export(scaled_pieces_to_export, f"{output_path}/{project_name}.stl", "STL", tolerance=TESSELLATION_LINEAR_TOLERANCE, angularTolerance=get_angular_tolerance())
                return (
                    f"{output_path}/{project_name}.step",
                    f"{output_path}/{project_name}.stl",
                )
            else:
                return scaled_pieces_to_export

        except:  # noqa: E722
            return None, None

    def get_magnetic_assembly(self, project_name, assembly_data, output_path=None, save_files=True, export_files=True):
        """Build a magnetic assembly from core, bobbin, and winding data.

        Parameters
        ----------
        project_name : str
            Name for the output files.
        assembly_data : dict
            Dictionary with optional keys: 'core', 'bobbin', 'windings', 'coil'.
        output_path : str
            Directory for output files.
        save_files : bool
            Whether to save intermediate files.
        export_files : bool
            Whether to export STEP/STL files.
        """
        try:
            from cadquery import exporters

            if output_path is None:
                output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"

            os.makedirs(output_path, exist_ok=True)
            project_name = f"{project_name}".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

            pieces = []

            # Build core
            if "core" in assembly_data and assembly_data["core"].get("geometricalDescription"):
                geometrical_description = assembly_data["core"]["geometricalDescription"]
                for index, geometrical_part in enumerate(geometrical_description):
                    if geometrical_part["type"] == "spacer":
                        spacer = self.get_spacer(geometrical_part)
                        pieces.append(spacer)
                    elif geometrical_part["type"] in ["half set", "toroidal"]:
                        shape_data = geometrical_part["shape"]
                        part_builder = CadQueryBuilder().factory(shape_data)

                        piece = part_builder.get_piece(data=copy.deepcopy(shape_data), name=f"Piece_{index}", save_files=False, export_files=False)
                        if piece is None:
                            continue

                        piece = piece.rotate((1, 0, 0), (-1, 0, 0), geometrical_part["rotation"][0] / math.pi * 180)
                        piece = piece.rotate((0, 1, 0), (0, -1, 0), geometrical_part["rotation"][2] / math.pi * 180)
                        piece = piece.rotate((0, 0, 1), (0, 0, -1), geometrical_part["rotation"][1] / math.pi * 180)

                        if "machining" in geometrical_part and geometrical_part["machining"] is not None:
                            for machining in geometrical_part["machining"]:
                                piece = part_builder.apply_machining(piece=piece, machining=machining, dimensions=flatten_dimensions(shape_data))

                        piece = piece.translate(convert_axis(geometrical_part["coordinates"]))

                        if geometrical_part["type"] in ["half set"]:
                            residual_gap = 5e-6
                            if geometrical_part["rotation"][0] > 0:
                                piece = piece.translate((0, 0, residual_gap / 2))
                            else:
                                piece = piece.translate((0, 0, -residual_gap / 2))

                        pieces.append(piece)

            if not pieces:
                return None, None

            if export_files:
                scaled_pieces = []
                for piece in pieces:
                    for o in piece.objects:
                        scaled_pieces.append(o.scale(1000))

                compound = cq.Compound.makeCompound(scaled_pieces)

                exporters.export(compound, f"{output_path}/{project_name}_assembly.step", "STEP")
                exporters.export(compound, f"{output_path}/{project_name}_assembly.stl", "STL", tolerance=TESSELLATION_LINEAR_TOLERANCE, angularTolerance=get_angular_tolerance())
                return f"{output_path}/{project_name}_assembly.step", f"{output_path}/{project_name}_assembly.stl"
            else:
                return pieces

        except:  # noqa: E722
            return None, None

    def get_spacers(self, project_name, geometrical_description, output_path=f'{os.path.dirname(os.path.abspath(__file__))}/../../output/', save_files=True, export_files=True):
        """Build only the spacers from the geometrical description.
        
        Spacers are built separately so they can be rendered with a different color.
        
        Args:
            project_name: Name for the output files
            geometrical_description: Core geometrical description from MKF
            output_path: Directory for output files
            save_files: Whether to save files
            export_files: Whether to export files
            
        Returns:
            Tuple of (step_path, stl_path) or scaled compound if export_files is False
        """
        try:
            spacers_to_export = []
            project_name = f"{project_name}_spacers".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

            os.makedirs(output_path, exist_ok=True)

            for geometrical_part in geometrical_description:
                if geometrical_part['type'] == 'spacer':
                    spacer = self.get_spacer(geometrical_part)
                    spacers_to_export.append(spacer)

            if len(spacers_to_export) == 0:
                return None, None

            if export_files:
                from cadquery import exporters
                scaled_spacers_to_export = []
                for spacer in spacers_to_export:
                    for o in spacer.objects:
                        scaled_spacers_to_export.append(o.scale(1000))

                scaled_spacers_to_export = cq.Compound.makeCompound(scaled_spacers_to_export)

                exporters.export(scaled_spacers_to_export, f"{output_path}/{project_name}.step", "STEP")
                exporters.export(
                    scaled_spacers_to_export, 
                    f"{output_path}/{project_name}.stl", 
                    "STL",
                    tolerance=TESSELLATION_LINEAR_TOLERANCE,
                    angularTolerance=get_angular_tolerance()
                )
                return f"{output_path}/{project_name}.step", f"{output_path}/{project_name}.stl"
            else:
                return scaled_spacers_to_export

        except:  # noqa: E722
            return None, None

    def get_bobbin(self, bobbin_data, winding_window, name="Bobbin", output_path=None, save_files=False, export_files=True):
        if output_path is None:
            output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"

        bobbin_builder = self.StandardBobbin()
        bobbin_builder.set_output_path(output_path)
        return bobbin_builder.get_bobbin(bobbin_data, winding_window, name, save_files, export_files)

    def get_winding(self, winding_data, bobbin_dims, name="Winding", output_path=None, save_files=False, export_files=True):
        if output_path is None:
            output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"

        winding_builder = self.RoundWireWinding()
        winding_builder.set_output_path(output_path)
        return winding_builder.get_winding(winding_data, bobbin_dims, name, save_files, export_files)

    def get_core_gapping_technical_drawing(self, project_name, core_data, colors=None, output_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../output/", save_files=True, export_files=True):
        try:
            from cadquery.occ_impl.exporters.svg import getSVG

            svg_project_name = f"{project_name}_core_gaps_FrontView".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
            geometrical_description = core_data["geometricalDescription"]

            os.makedirs(output_path, exist_ok=True)

            if colors is None:
                colors = {"projection_color": "#000000", "dimension_color": "#000000"}

            pieces_to_export = []
            for index, geometrical_part in enumerate(geometrical_description):
                if geometrical_part["type"] == "spacer":
                    spacer = self.get_spacer(geometrical_part)
                    pieces_to_export.append(spacer)
                elif geometrical_part["type"] in ["half set", "toroidal"]:
                    shape_data = geometrical_part["shape"]
                    part_builder = CadQueryBuilder().factory(shape_data)

                    piece = part_builder.get_piece(data=copy.deepcopy(shape_data), name=f"Piece_{index}", save_files=False, export_files=False)
                    if piece is None:
                        continue

                    piece = piece.rotate((1, 0, 0), (-1, 0, 0), geometrical_part["rotation"][0] / math.pi * 180)
                    piece = piece.rotate((0, 1, 0), (0, -1, 0), geometrical_part["rotation"][2] / math.pi * 180)
                    piece = piece.rotate((0, 0, 1), (0, 0, -1), geometrical_part["rotation"][1] / math.pi * 180)

                    if "machining" in geometrical_part and geometrical_part["machining"] is not None:
                        for machining in geometrical_part["machining"]:
                            piece = part_builder.apply_machining(piece=piece, machining=machining, dimensions=flatten_dimensions(shape_data))

                    piece = piece.translate(convert_axis(geometrical_part["coordinates"]))
                    pieces_to_export.append(piece)

            if not pieces_to_export:
                return None

            scaled_pieces = []
            for piece in pieces_to_export:
                for o in piece.objects:
                    scaled_pieces.append(o.scale(1000))

            compound = cq.Compound.makeCompound(scaled_pieces)

            stroke_color = self.IPiece._hex_to_rgb(colors.get("projection_color", "#000000"))
            svg_opts = {
                "width": 800,
                "height": 600,
                "strokeWidth": 0.5,
                "strokeColor": stroke_color,
                "showHidden": True,
                "projectionDir": (0, 1, 0),
            }

            front_svg = getSVG(compound, svg_opts)
            svg_path = f"{output_path}/{svg_project_name}.svg"
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(front_svg)

            return front_svg

        except Exception:
            return None

    def _build_core_pieces(self, geometrical_description):
        """Build core pieces from geometrical description, returning unscaled pieces and shape info."""
        pieces = []
        family = None
        dims = None
        original_dims = None

        for index, geometrical_part in enumerate(geometrical_description):
            if geometrical_part["type"] == "spacer":
                spacer = self.get_spacer(geometrical_part)
                pieces.append(spacer)
            elif geometrical_part["type"] in ["half set", "toroidal"]:
                shape_data = geometrical_part["shape"]
                part_builder = CadQueryBuilder().factory(shape_data)

                if family is None:
                    family = shape_data.get("family", "e")
                    dims = flatten_dimensions(copy.deepcopy(shape_data))
                    original_dims = utils.flatten_dimensions(copy.deepcopy(shape_data), scale_factor=1.0)

                piece = part_builder.get_piece(data=copy.deepcopy(shape_data), name=f"Piece_{index}", save_files=False, export_files=False)
                if piece is None:
                    continue

                piece = piece.rotate((1, 0, 0), (-1, 0, 0), geometrical_part["rotation"][0] / math.pi * 180)
                piece = piece.rotate((0, 1, 0), (0, -1, 0), geometrical_part["rotation"][2] / math.pi * 180)
                piece = piece.rotate((0, 0, 1), (0, 0, -1), geometrical_part["rotation"][1] / math.pi * 180)

                if "machining" in geometrical_part and geometrical_part["machining"] is not None:
                    for machining in geometrical_part["machining"]:
                        piece = part_builder.apply_machining(piece=piece, machining=machining, dimensions=flatten_dimensions(shape_data))

                piece = piece.translate(convert_axis(geometrical_part["coordinates"]))
                pieces.append(piece)

        return pieces, family, dims, original_dims

    def _make_compound(self, pieces):
        """Create a scaled compound from a list of CadQuery workplane pieces."""
        scaled = []
        for piece in pieces:
            for o in piece.objects:
                scaled.append(o.scale(1000))
        return cq.Compound.makeCompound(scaled)

    @staticmethod
    def _get_svg_for_view(view, stroke_color):
        """Generate SVG for a single DrawingView, handling HLR failures gracefully.

        Cross-section views use showHidden=False (no hidden edges in 2D slices).
        Falls back to showHidden=False if HLR fails on any view.
        """
        import logging

        from drawing_2d import PROJECTION_DIRS, ViewType
        from cadquery.occ_impl.exporters.svg import getSVG

        proj_dir = PROJECTION_DIRS[view.plane]
        show_hidden = view.view_type != ViewType.CROSS_SECTION

        svg_opts = {
            "width": 800,
            "height": 600,
            "strokeWidth": 0.5,
            "strokeColor": stroke_color,
            "showHidden": show_hidden,
            "projectionDir": proj_dir,
        }

        shape = view.shape
        if hasattr(shape, "val"):
            shape = shape.val()

        try:
            return getSVG(shape, svg_opts)
        except (RuntimeError, ValueError):
            if show_hidden:
                logging.warning("SVG HLR failed for view %s, retrying without hidden edges", view.plane)
                svg_opts["showHidden"] = False
                return getSVG(shape, svg_opts)
            raise

    def _generate_views(self, compound, family, dims, original_dims, planes, view_types, colors, slice_offsets=None):
        """Generate DrawingView objects for given planes and view types.

        Args:
            compound: CadQuery Compound (scaled to mm).
            family: Shape family string.
            dims: Processed dimensions dict.
            original_dims: Original dimensions dict for labels.
            planes: List of ViewPlane values.
            view_types: List of ViewType values.
            colors: Color config dict.
            slice_offsets: Optional dict overriding per-plane slice offsets.

        Returns:
            Dict of {key: DrawingView} where key is e.g. 'xy_projection'.
        """
        from drawing_2d import ViewType, DrawingView, cross_section_at_plane

        views = {}
        for plane in planes:
            for vtype in view_types:
                key = f"{plane.value}_{vtype.value}"

                if vtype == ViewType.PROJECTION:
                    shape = compound
                else:
                    offset = 0.0
                    if slice_offsets and plane.value in slice_offsets:
                        offset = slice_offsets[plane.value]
                    elif family:
                        offsets = shape_configs.CROSS_SECTION_OFFSETS.get(family.lower(), {})
                        offset = offsets.get(plane.value, 0.0)
                    shape = cross_section_at_plane(compound, plane, offset)
                    if shape is None:
                        continue

                views[key] = DrawingView(
                    plane=plane,
                    view_type=vtype,
                    shape=shape,
                    dimensions=[],
                    title=f"{family}_{key}" if family else key,
                )

        return views

    def get_svg_drawings(self, project_name, geometrical_description, planes=None, view_types=None, colors=None, output_path=None, save_files=True, slice_offsets=None):
        """Generate annotated SVG drawings for a core shape.

        Args:
            project_name: Name for output files.
            geometrical_description: List of geometrical part descriptions.
            planes: List of ViewPlane values (default: all three).
            view_types: List of ViewType values (default: PROJECTION only).
            colors: Dict with 'projection_color' and 'dimension_color'.
            output_path: Directory for output files.
            save_files: Whether to write SVG files to disk.
            slice_offsets: Optional dict overriding per-plane slice offsets.

        Returns:
            Dict of {key: svg_string} where key is e.g. 'xy_projection'.
        """
        try:
            import logging

            from drawing_2d import ViewPlane, ViewType

            if planes is None:
                planes = [ViewPlane.XY, ViewPlane.XZ, ViewPlane.ZY]
            if view_types is None:
                view_types = [ViewType.PROJECTION]
            if colors is None:
                colors = {"projection_color": "#000000", "dimension_color": "#000000"}
            if output_path is None:
                output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"

            os.makedirs(output_path, exist_ok=True)
            safe_name = project_name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

            pieces, family, dims, original_dims = self._build_core_pieces(geometrical_description)
            if not pieces:
                return {}

            compound = self._make_compound(pieces)

            views = self._generate_views(compound, family, dims, original_dims, planes, view_types, colors, slice_offsets)

            stroke_color = self.IPiece._hex_to_rgb(colors.get("projection_color", "#000000"))
            results = {}
            for key, view in views.items():
                try:
                    svg = self._get_svg_for_view(view, stroke_color)
                except (RuntimeError, ValueError) as e:
                    logging.warning("Skipping SVG view %s: %s", key, e)
                    continue

                if save_files:
                    svg_path = f"{output_path}/{safe_name}_{key}.svg"
                    with open(svg_path, "w", encoding="utf-8") as f:
                        f.write(svg)

                results[key] = svg

            return results

        except Exception:
            return {}

    def get_dxf_drawings(self, project_name, geometrical_description, planes=None, view_types=None, colors=None, output_path=None, slice_offsets=None):
        """Generate DXF drawings for a core shape.

        Returns:
            Dict of {key: filepath} where key is e.g. 'xy_projection'.
        """
        try:
            from drawing_2d import ViewPlane, ViewType, export_dxf_from_shape

            if planes is None:
                planes = [ViewPlane.XY, ViewPlane.XZ, ViewPlane.ZY]
            if view_types is None:
                view_types = [ViewType.PROJECTION]
            if colors is None:
                colors = {"projection_color": "#000000", "dimension_color": "#000000"}
            if output_path is None:
                output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"

            os.makedirs(output_path, exist_ok=True)
            safe_name = project_name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

            pieces, family, dims, original_dims = self._build_core_pieces(geometrical_description)
            if not pieces:
                return {}

            compound = self._make_compound(pieces)

            views = self._generate_views(compound, family, dims, original_dims, planes, view_types, colors, slice_offsets)

            results = {}
            for key, view in views.items():
                filepath = export_dxf_from_shape(view.shape, view.plane, output_path, f"{safe_name}_{key}", view_type=view.view_type, colors=colors)
                if filepath:
                    results[key] = filepath

            return results

        except Exception:
            return {}

    def get_fcstd_sketches(self, project_name, geometrical_description, planes=None, view_types=None, output_path=None, slice_offsets=None):
        """Generate FreeCAD macro files (.FCMacro) for a core shape.

        Returns:
            Dict of {key: filepath} where key is e.g. 'xy_projection'.
        """
        try:
            from drawing_2d import ViewPlane, ViewType, export_fcstd_macro_from_shape

            if planes is None:
                planes = [ViewPlane.XY, ViewPlane.XZ, ViewPlane.ZY]
            if view_types is None:
                view_types = [ViewType.PROJECTION]
            if output_path is None:
                output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"

            os.makedirs(output_path, exist_ok=True)
            safe_name = project_name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

            pieces, family, dims, original_dims = self._build_core_pieces(geometrical_description)
            if not pieces:
                return {}

            compound = self._make_compound(pieces)

            colors = {"projection_color": "#000000", "dimension_color": "#000000"}
            views = self._generate_views(compound, family, dims, original_dims, planes, view_types, colors, slice_offsets)

            results = {}
            for key, view in views.items():
                filepath = export_fcstd_macro_from_shape(view.shape, view.plane, output_path, f"{safe_name}_{key}", view_type=view.view_type)
                if filepath:
                    results[key] = filepath

            return results

        except Exception:
            return {}

    def get_assembly_svg_drawings(self, project_name, magnetic_data, planes=None, view_types=None, colors=None, output_path=None, save_files=True, slice_offsets=None, components=None):
        """Generate annotated SVG drawings for an assembly or individual components.

        Args:
            project_name: Name for output files.
            magnetic_data: MAS magnetic data dict.
            planes: List of ViewPlane values.
            view_types: List of ViewType values.
            colors: Color config dict.
            output_path: Directory for output files.
            save_files: Whether to write SVG files to disk.
            slice_offsets: Optional per-plane slice offsets.
            components: List of components to draw: 'assembly', 'core', 'bobbin', 'winding'.
                       Default is ['assembly'].

        Returns:
            Dict of {component_key: svg_string}.
        """
        try:
            import logging

            from drawing_2d import ViewPlane, ViewType

            if planes is None:
                planes = [ViewPlane.XY, ViewPlane.XZ, ViewPlane.ZY]
            if view_types is None:
                view_types = [ViewType.PROJECTION]
            if colors is None:
                colors = {"projection_color": "#000000", "dimension_color": "#000000"}
            if output_path is None:
                output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"
            if components is None:
                components = ["assembly"]

            if "magnetic" in magnetic_data:
                magnetic_data = magnetic_data["magnetic"]

            os.makedirs(output_path, exist_ok=True)
            safe_name = project_name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

            # Build all pieces using get_magnetic logic
            all_pieces = self.get_magnetic(magnetic_data, project_name, output_path, export_files=False)
            if all_pieces is None or (isinstance(all_pieces, tuple) and all_pieces[0] is None):
                return {}

            if not isinstance(all_pieces, list):
                return {}

            # For assembly, use all pieces
            compound_map = {}
            if "assembly" in components:
                compound_map["assembly"] = self._make_compound(all_pieces)

            # For core-only views, rebuild just core pieces
            core_data = magnetic_data.get("core", {})
            geo_desc = core_data.get("geometricalDescription", [])
            if "core" in components and geo_desc:
                core_pieces, family, dims, original_dims = self._build_core_pieces(geo_desc)
                if core_pieces:
                    compound_map["core"] = self._make_compound(core_pieces)

            family = None
            dims = None
            original_dims = None
            if geo_desc:
                for part in geo_desc:
                    if part["type"] in ["half set", "toroidal"]:
                        shape_data = part["shape"]
                        family = shape_data.get("family", "e")
                        dims = flatten_dimensions(copy.deepcopy(shape_data))
                        original_dims = utils.flatten_dimensions(copy.deepcopy(shape_data), scale_factor=1.0)
                        break

            stroke_color = self.IPiece._hex_to_rgb(colors.get("projection_color", "#000000"))
            results = {}

            for comp_name, compound in compound_map.items():
                views = self._generate_views(compound, family, dims, original_dims, planes, view_types, colors, slice_offsets)

                for key, view in views.items():
                    full_key = f"{comp_name}_{key}"
                    try:
                        svg = self._get_svg_for_view(view, stroke_color)
                    except (RuntimeError, ValueError) as e:
                        logging.warning("Skipping assembly SVG view %s: %s", full_key, e)
                        continue

                    if save_files:
                        svg_path = f"{output_path}/{safe_name}_{full_key}.svg"
                        with open(svg_path, "w", encoding="utf-8") as f:
                            f.write(svg)

                    results[full_key] = svg

            return results

        except Exception:
            return {}

    def get_assembly_dxf_drawings(self, project_name, magnetic_data, planes=None, view_types=None, colors=None, output_path=None, slice_offsets=None, components=None):
        """Generate DXF drawings for assembly or individual components.

        Returns:
            Dict of {component_key: filepath}.
        """
        try:
            from drawing_2d import ViewPlane, ViewType, export_dxf_from_shape

            if planes is None:
                planes = [ViewPlane.XY, ViewPlane.XZ, ViewPlane.ZY]
            if view_types is None:
                view_types = [ViewType.PROJECTION]
            if colors is None:
                colors = {"projection_color": "#000000", "dimension_color": "#000000"}
            if output_path is None:
                output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"
            if components is None:
                components = ["assembly"]

            if "magnetic" in magnetic_data:
                magnetic_data = magnetic_data["magnetic"]

            os.makedirs(output_path, exist_ok=True)
            safe_name = project_name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

            all_pieces = self.get_magnetic(magnetic_data, project_name, output_path, export_files=False)
            if all_pieces is None or (isinstance(all_pieces, tuple) and all_pieces[0] is None):
                return {}

            if not isinstance(all_pieces, list):
                return {}

            compound_map = {}
            if "assembly" in components:
                compound_map["assembly"] = self._make_compound(all_pieces)

            core_data = magnetic_data.get("core", {})
            geo_desc = core_data.get("geometricalDescription", [])
            if "core" in components and geo_desc:
                core_pieces, _, _, _ = self._build_core_pieces(geo_desc)
                if core_pieces:
                    compound_map["core"] = self._make_compound(core_pieces)

            family = None
            dims = None
            original_dims = None
            if geo_desc:
                for part in geo_desc:
                    if part["type"] in ["half set", "toroidal"]:
                        shape_data = part["shape"]
                        family = shape_data.get("family", "e")
                        dims = flatten_dimensions(copy.deepcopy(shape_data))
                        original_dims = utils.flatten_dimensions(copy.deepcopy(shape_data), scale_factor=1.0)
                        break

            results = {}
            for comp_name, compound in compound_map.items():
                views = self._generate_views(compound, family, dims, original_dims, planes, view_types, colors, slice_offsets)

                for key, view in views.items():
                    full_key = f"{comp_name}_{key}"
                    filepath = export_dxf_from_shape(view.shape, view.plane, output_path, f"{safe_name}_{full_key}", view_type=view.view_type, colors=colors)
                    if filepath:
                        results[full_key] = filepath

            return results

        except Exception:
            return {}

    def get_assembly_fcstd_sketches(self, project_name, magnetic_data, planes=None, view_types=None, output_path=None, slice_offsets=None, components=None):
        """Generate FreeCAD macro files for assembly or individual components.

        Returns:
            Dict of {component_key: filepath}.
        """
        try:
            from drawing_2d import ViewPlane, ViewType, export_fcstd_macro_from_shape

            if planes is None:
                planes = [ViewPlane.XY, ViewPlane.XZ, ViewPlane.ZY]
            if view_types is None:
                view_types = [ViewType.PROJECTION]
            if output_path is None:
                output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"
            if components is None:
                components = ["assembly"]

            if "magnetic" in magnetic_data:
                magnetic_data = magnetic_data["magnetic"]

            os.makedirs(output_path, exist_ok=True)
            safe_name = project_name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

            all_pieces = self.get_magnetic(magnetic_data, project_name, output_path, export_files=False)
            if all_pieces is None or (isinstance(all_pieces, tuple) and all_pieces[0] is None):
                return {}

            if not isinstance(all_pieces, list):
                return {}

            compound_map = {}
            if "assembly" in components:
                compound_map["assembly"] = self._make_compound(all_pieces)

            core_data = magnetic_data.get("core", {})
            geo_desc = core_data.get("geometricalDescription", [])
            if "core" in components and geo_desc:
                core_pieces, _, _, _ = self._build_core_pieces(geo_desc)
                if core_pieces:
                    compound_map["core"] = self._make_compound(core_pieces)

            family = None
            dims = None
            original_dims = None
            if geo_desc:
                for part in geo_desc:
                    if part["type"] in ["half set", "toroidal"]:
                        shape_data = part["shape"]
                        family = shape_data.get("family", "e")
                        dims = flatten_dimensions(copy.deepcopy(shape_data))
                        original_dims = utils.flatten_dimensions(copy.deepcopy(shape_data), scale_factor=1.0)
                        break

            colors = {"projection_color": "#000000", "dimension_color": "#000000"}
            results = {}
            for comp_name, compound in compound_map.items():
                views = self._generate_views(compound, family, dims, original_dims, planes, view_types, colors, slice_offsets)

                for key, view in views.items():
                    full_key = f"{comp_name}_{key}"
                    filepath = export_fcstd_macro_from_shape(view.shape, view.plane, output_path, f"{safe_name}_{full_key}", view_type=view.view_type)
                    if filepath:
                        results[full_key] = filepath

            return results

        except Exception:
            return {}

    def get_turn(
        self,
        turn_description: TurnDescription,
        wire_description: WireDescription,
        bobbin_description: BobbinProcessedDescription,
        is_toroidal: bool = False,
    ) -> cq.Workplane:
        """Create a single turn geometry.

        Args:
            turn_description: Turn parameters (coordinates, winding, etc.)
            wire_description: Wire parameters (type, diameter, etc.)
            bobbin_description: Bobbin parameters
            is_toroidal: If True, create toroidal turn; otherwise concentric turn

        Returns:
            CadQuery Workplane with the turn geometry
        """
        if is_toroidal or bobbin_description.winding_window_angle is not None:
            return self._create_toroidal_turn(turn_description, wire_description, bobbin_description)
        else:
            return self._create_concentric_turn(turn_description, wire_description, bobbin_description)

    def _create_concentric_turn(
        self,
        turn_description: TurnDescription,
        wire_description: WireDescription,
        bobbin_description: BobbinProcessedDescription,
    ) -> cq.Workplane:
        """Create a concentric turn (for E-cores, PQ, RM, etc.).

        Following Ansyas approach with CadQuery coordinate system:
        - X axis: depth direction (along column depth, perpendicular to winding window)
        - Y axis: width direction (radial, distance from center column)
        - Z axis: height direction (along core axis, vertical)

        MAS coordinates for turns:
        - coordinates[0] = radial position (distance from center) -> maps to Y
        - coordinates[1] = height position (along core axis) -> maps to Z

        The turn is built as 4 straight tubes + 4 corner quarter-tori.
        """
        from OCP.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Circ, gp_Vec
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeRevol
        from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,
                                         BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon)
        from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipe
        from OCP.GC import GC_MakeCircle, GC_MakeArcOfCircle
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound
        import cadquery as cq

        SCALE = self.SCALE
        WIRE_POLYGON_SEGMENTS = self.WIRE_POLYGON_SEGMENTS
        
        # Get wire dimensions
        is_rectangular_wire = wire_description.wire_type == WireType.rectangular
        if is_rectangular_wire:
            # Try turn dimensions first, then fall back to wire description
            if turn_description.dimensions and len(turn_description.dimensions) >= 2:
                wire_width = turn_description.dimensions[0] * SCALE
                wire_height = turn_description.dimensions[1] * SCALE
            else:
                wire_width = (wire_description.outer_width or wire_description.conducting_width or 0.001) * SCALE
                wire_height = (wire_description.outer_height or wire_description.conducting_height or 0.001) * SCALE
            wire_radius = min(wire_width, wire_height) / 2.0  # used for corner sizing
        else:  # round, litz
            wire_diameter = (wire_description.conducting_diameter or wire_description.outer_diameter or 0.001) * SCALE
            wire_radius = wire_diameter / 2.0
            wire_width = wire_diameter  # for consistent API
            wire_height = wire_diameter
        
        # Core column half-dimensions (MAS coordinates are from core center)
        half_col_depth = bobbin_description.column_depth * SCALE
        half_col_width = bobbin_description.column_width * SCALE
        
        # Get turn position from coordinates
        coords = turn_description.coordinates
        radial_pos = coords[0] * SCALE if len(coords) > 0 else (half_col_width + wire_radius)
        height_pos = coords[1] * SCALE if len(coords) > 1 else 0
        
        # Corner radius: distance from column edge to wire center.
        # Pipe sweep needs bend_radius > half the profile width in the
        # bending plane to avoid self-intersection at corners.
        turn_turn_radius = radial_pos - half_col_width
        min_bend = max(wire_width, wire_height) / 2 * 1.02
        needs_fillet = turn_turn_radius < min_bend * 1.5
        if turn_turn_radius < min_bend:
            turn_turn_radius = min_bend
        
        if bobbin_description.column_shape == ColumnShape.round:
            # Round column: circular turn path
            turn_radius = radial_pos  # Distance from center to wire center
            
            # Build polygon path around column axis (Y) in XZ plane.
            # Polygon path eliminates NURBS curves → all-flat-facet geometry
            # that gmsh can mesh without curved-surface issues at the gap.
            # Use cadquery sweep (not raw OCC MakePipe which fails at polygon corners).
            TURN_PATH_SEGMENTS = CadQueryBuilder.CORE_POLYGON_SEGMENTS
            # Half-segment offset prevents path faces from aligning with air box
            # faces (axis-aligned planes), which causes zero-volume degenerate tets.
            path_offset = math.pi / TURN_PATH_SEGMENTS
            path_pts = []
            for seg in range(TURN_PATH_SEGMENTS):
                angle = 2 * math.pi * seg / TURN_PATH_SEGMENTS + path_offset
                path_pts.append((turn_radius * math.cos(angle),
                                 turn_radius * math.sin(angle)))
            path = (cq.Workplane("XZ")
                    .polyline(path_pts).close()
                    .translate((0, height_pos, 0)))

            if is_rectangular_wire:
                half_ww = wire_width / 2
                half_wh = wire_height / 2
                profile = (cq.Workplane("XY")
                           .moveTo(turn_radius - half_ww, height_pos - half_wh)
                           .rect(wire_width, wire_height))
                turn = profile.sweep(path)
            else:
                if wire_diameter >= 0.3:
                    prof_pts = []
                    offset = math.pi / WIRE_POLYGON_SEGMENTS
                    for seg in range(WIRE_POLYGON_SEGMENTS):
                        angle = 2 * math.pi * seg / WIRE_POLYGON_SEGMENTS + offset
                        prof_pts.append((turn_radius + wire_radius * math.cos(angle),
                                         height_pos + wire_radius * math.sin(angle)))
                    profile = cq.Workplane("XY").polyline(prof_pts).close()
                    turn = profile.sweep(path)
                else:
                    # Exact torus for thin wires (< 0.3mm)
                    torus_center = gp_Pnt(0, height_pos, 0)
                    torus_axis = gp_Ax2(torus_center, gp_Dir(0, 1, 0), gp_Dir(1, 0, 0))
                    torus = BRepPrimAPI_MakeTorus(torus_axis, turn_radius, wire_radius).Shape()
                    turn = cq.Workplane("XY").add(cq.Shape(torus))
        
        elif bobbin_description.column_shape == ColumnShape.oblong:
            # Oblong column: stadium-shaped turn path
            # Column is round in X with straight extensions in Z (depth).
            # Turn wraps with straight sections along ±Z, semicircles at ±Z ends.
            # X=width (semicircle radius), Z=depth (straight + semicircle)

            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
            from OCP.GC import GC_MakeCircle

            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)

            # Z length of the straight section (depth minus the semicircle part)
            straight_section_half_length = half_col_depth - half_col_width

            # If straight_section_half_length <= 0, the column is actually round
            if straight_section_half_length <= 0:
                turn_radius = radial_pos
                torus_center = gp_Pnt(0, height_pos, 0)
                torus_axis = gp_Ax2(torus_center, gp_Dir(0, 1, 0), gp_Dir(1, 0, 0))
                torus = BRepPrimAPI_MakeTorus(torus_axis, turn_radius, wire_radius).Shape()
                turn = cq.Workplane("XY").add(cq.Shape(torus))
            else:
                wire_x_pos = radial_pos  # radial distance in X
                tube_z_length = 2 * straight_section_half_length  # straight along Z

                # +X side tube (along Z)
                tube_px = (
                    cq.Workplane("XY")
                    .center(wire_x_pos, height_pos)
                    .circle(wire_radius)
                    .extrude(tube_z_length)
                    .translate((0, 0, -straight_section_half_length))
                )
                builder.Add(compound, tube_px.val().wrapped)

                # -X side tube (along Z)
                tube_nx = (
                    cq.Workplane("XY")
                    .center(-wire_x_pos, height_pos)
                    .circle(wire_radius)
                    .extrude(tube_z_length)
                    .translate((0, 0, -straight_section_half_length))
                )
                builder.Add(compound, tube_nx.val().wrapped)

                # Semicircular torus arcs at ±Z ends
                # Revolve around Z axis at Z = ±straight_section_half_length
                def create_half_torus(center_z: float, start_angle_deg: float):
                    circle_center = gp_Pnt(radial_pos, height_pos, center_z)
                    angle_rad = math.radians(start_angle_deg)
                    circle_normal = gp_Dir(math.cos(angle_rad), 0, math.sin(angle_rad))
                    circle_axis = gp_Ax2(circle_center, circle_normal)

                    circle = GC_MakeCircle(circle_axis, wire_radius).Value()
                    circle_edge = BRepBuilderAPI_MakeEdge(circle).Edge()
                    circle_wire = BRepBuilderAPI_MakeWire(circle_edge).Wire()
                    circle_face = BRepBuilderAPI_MakeFace(circle_wire).Face()

                    # Revolve around Z axis
                    revolve_axis = gp_Ax1(gp_Pnt(0, height_pos, center_z), gp_Dir(0, 0, 1))
                    half_torus = BRepPrimAPI_MakeRevol(circle_face, revolve_axis, math.pi).Shape()
                    return half_torus

                # +Z end semicircle
                half_torus_pz = create_half_torus(straight_section_half_length, 90)
                builder.Add(compound, half_torus_pz)
                
                # -Z end semicircle
                half_torus_nz = create_half_torus(-straight_section_half_length, -90)
                builder.Add(compound, half_torus_nz)

                # Fuse all pieces into a single solid
                from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
                from OCP.TopExp import TopExp_Explorer
                from OCP.TopAbs import TopAbs_SOLID
                fused = cq.Shape(compound)
                try:
                    solids = []
                    explorer = TopExp_Explorer(compound, TopAbs_SOLID)
                    while explorer.More():
                        solids.append(cq.Shape(explorer.Current()))
                        explorer.Next()
                    if len(solids) > 1:
                        result_shape = solids[0]
                        for s in solids[1:]:
                            result_shape = cq.Shape(BRepAlgoAPI_Fuse(result_shape.wrapped, s.wrapped).Shape())
                        turn = cq.Workplane("XY").add(result_shape)
                    else:
                        turn = cq.Workplane("XY").add(fused)
                except Exception:
                    turn = cq.Workplane("XY").add(fused)
            
        else:
            # Rectangular column: sweep a circular cross-section along a closed
            # rounded-rectangle wire path. This creates a single valid solid
            # that works correctly with Elmer's CoilSolver.
            # Path in XZ plane at Y=height_pos
            # X = width direction, Z = depth direction
            wire_x_pos = half_col_width + turn_turn_radius  # X (width)
            wire_z_pos = half_col_depth + turn_turn_radius  # Z (depth)
            y = height_pos

            # 8 transition points: rounded rectangle in XZ plane
            pts = [
                gp_Pnt(+half_col_width, y, +wire_z_pos),   # 0: start of +Z straight
                gp_Pnt(-half_col_width, y, +wire_z_pos),   # 1: end of +Z straight (along -X)
                gp_Pnt(-wire_x_pos, y, +half_col_depth),   # 2: start of -X straight (after corner)
                gp_Pnt(-wire_x_pos, y, -half_col_depth),   # 3: end of -X straight
                gp_Pnt(-half_col_width, y, -wire_z_pos),   # 4: start of -Z straight
                gp_Pnt(+half_col_width, y, -wire_z_pos),   # 5: end of -Z straight
                gp_Pnt(+wire_x_pos, y, -half_col_depth),   # 6: start of +X straight
                gp_Pnt(+wire_x_pos, y, +half_col_depth),   # 7: end of +X straight
            ]

            # Corner centers — arc normal is Y (perpendicular to XZ path plane)
            corners = [
                (gp_Pnt(-half_col_width, y, +half_col_depth), gp_Dir(0, 0, 1)),   # -X+Z
                (gp_Pnt(-half_col_width, y, -half_col_depth), gp_Dir(-1, 0, 0)),  # -X-Z
                (gp_Pnt(+half_col_width, y, -half_col_depth), gp_Dir(0, 0, -1)),  # +X-Z
                (gp_Pnt(+half_col_width, y, +half_col_depth), gp_Dir(1, 0, 0)),   # +X+Z
            ]

            wire_builder = BRepBuilderAPI_MakeWire()

            segment_pairs = [
                (0, 1, 0),  # +Z straight, then corner[0]
                (2, 3, 1),  # -X straight, then corner[1]
                (4, 5, 2),  # -Z straight, then corner[2]
                (6, 7, 3),  # +X straight, then corner[3]
            ]

            for (i_start, i_end, c_idx) in segment_pairs:
                wire_builder.Add(BRepBuilderAPI_MakeEdge(pts[i_start], pts[i_end]).Edge())

                c_center, c_xref = corners[c_idx]
                circ = gp_Circ(gp_Ax2(c_center, gp_Dir(0, -1, 0), c_xref), turn_turn_radius)
                i_arc_end = (i_end + 1) % 8
                arc = GC_MakeArcOfCircle(circ, pts[i_end], pts[i_arc_end], True).Value()
                wire_builder.Add(BRepBuilderAPI_MakeEdge(arc).Edge())

            spine = wire_builder.Wire()

            # Cross-section at pts[0], tangent direction is -X
            # Profile in YZ plane centered at pts[0]
            if is_rectangular_wire:
                # Build rectangle at pts[0] in the plane perpendicular to tangent (-X)
                # Y dimension = wire_height, Z dimension = wire_width
                p0 = pts[0]
                half_wh = wire_height / 2
                half_ww = wire_width / 2
                from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
                rect_wire = BRepBuilderAPI_MakePolygon(
                    gp_Pnt(p0.X(), p0.Y() - half_wh, p0.Z() - half_ww),
                    gp_Pnt(p0.X(), p0.Y() + half_wh, p0.Z() - half_ww),
                    gp_Pnt(p0.X(), p0.Y() + half_wh, p0.Z() + half_ww),
                    gp_Pnt(p0.X(), p0.Y() - half_wh, p0.Z() + half_ww),
                    True  # close
                ).Wire()
                profile_face = BRepBuilderAPI_MakeFace(rect_wire).Face()
            else:
                # Polygonal approximation of circle for cleaner OCC booleans/meshing
                poly = BRepBuilderAPI_MakePolygon()
                cx, cy, cz = pts[0].X(), pts[0].Y(), pts[0].Z()
                offset = math.pi / WIRE_POLYGON_SEGMENTS
                for seg in range(WIRE_POLYGON_SEGMENTS):
                    angle = 2 * math.pi * seg / WIRE_POLYGON_SEGMENTS + offset
                    py = cy + wire_radius * math.cos(angle)
                    pz = cz + wire_radius * math.sin(angle)
                    poly.Add(gp_Pnt(cx, py, pz))
                poly.Close()
                profile_face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()

            pipe = BRepOffsetAPI_MakePipe(spine, profile_face)
            turn = cq.Workplane("XY").add(cq.Shape(pipe.Shape()))

            # When the natural corner radius is tight, the inner surface of
            # the pipe sweep at corners can degenerate. Fillet only those
            # inner edges (closest to column corner centers) to smooth them.
            if needs_fillet:
                fillet_radius = wire_radius * 0.2
                try:
                    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
                    from OCP.TopExp import TopExp_Explorer
                    from OCP.TopAbs import TopAbs_EDGE
                    from OCP.BRep import BRep_Tool
                    from OCP.TopoDS import TopoDS

                    corner_pts = [
                        (+half_col_depth, +half_col_width),
                        (-half_col_depth, +half_col_width),
                        (-half_col_depth, -half_col_width),
                        (+half_col_depth, -half_col_width),
                    ]

                    # Collect inner edges: those whose midpoint is within
                    # turn_turn_radius of a column corner (the tight zone)
                    inner_edges = []
                    explorer = TopExp_Explorer(turn.val().wrapped, TopAbs_EDGE)
                    while explorer.More():
                        edge = explorer.Current()
                        try:
                            curve, u0, u1 = BRep_Tool.Curve_s(edge)
                            if curve is not None:
                                mid = curve.Value((u0 + u1) / 2)
                                for cx, cy in corner_pts:
                                    d = math.sqrt((mid.X()-cx)**2 + (mid.Y()-cy)**2)
                                    if d < turn_turn_radius:
                                        inner_edges.append(TopoDS.Edge_s(edge))
                                        break
                        except BaseException:
                            pass
                        explorer.Next()

                    if inner_edges:
                        mk = BRepFilletAPI_MakeFillet(turn.val().wrapped)
                        for e in inner_edges:
                            mk.Add(fillet_radius, e)
                        mk.Build()
                        if mk.IsDone():
                            turn = cq.Workplane("XY").add(cq.Shape(mk.Shape()))
                except BaseException:
                    pass  # If fillet fails, keep the solid as-is

        # Scale back to meters
        final_shape = turn.val()
        scaled_shape = final_shape.scale(1 / SCALE)

        return cq.Workplane("XY").add(scaled_shape)

    def _build_bobbin_geometry(
        self,
        bobbin_description: BobbinProcessedDescription,
    ) -> Optional[cq.Workplane]:
        """Create bobbin geometry for concentric (E-core, PQ, etc.) magnetics.

        Internal method used by get_magnetic for MAS-based bobbin creation.

        Args:
            bobbin_description: Processed bobbin parameters

        Returns:
            CadQuery Workplane with bobbin geometry, or None if bobbin has zero thickness
        """
        # Check if bobbin has actual thickness
        if round(bobbin_description.wall_thickness, 12) == 0 or round(bobbin_description.column_thickness, 12) == 0:
            return None

        SCALE = self.SCALE

        # Scale to mm for construction
        col_depth = bobbin_description.column_depth * SCALE
        col_width = bobbin_description.column_width * SCALE
        col_thickness = bobbin_description.column_thickness * SCALE
        wall_thickness = bobbin_description.wall_thickness * SCALE
        ww_height = bobbin_description.winding_window_height * SCALE
        ww_width = bobbin_description.winding_window_width * SCALE

        # Total bobbin dimensions
        # Outer wall/yoke = column dimension + winding window width
        # Central column = core column dimension + bobbin column thickness
        total_height = ww_height + wall_thickness * 2
        total_width = ww_width + col_width
        total_depth = ww_width + col_depth

        if bobbin_description.column_shape == ColumnShape.round:
            # Round column bobbin (polygon-approximated cylinders for mesh compatibility)
            n_seg = CadQueryBuilder.CORE_POLYGON_SEGMENTS
            bobbin = CadQueryBuilder.polygon_cylinder(total_height, total_width, n_seg, "XZ")
            neg_ww = CadQueryBuilder.polygon_cylinder(ww_height, total_width, n_seg, "XZ")
            central_col = CadQueryBuilder.polygon_cylinder(ww_height, col_width, n_seg, "XZ")
            central_hole = CadQueryBuilder.polygon_cylinder(
                total_height, col_width - col_thickness, n_seg, "XZ")

            # Subtract operations
            neg_ww_cut = neg_ww.cut(central_col)
            bobbin = bobbin.cut(neg_ww_cut)
            bobbin = bobbin.cut(central_hole)

        else:
            # Rectangular column bobbin (box-shaped)
            # .box() on XY: args map directly to X, Y, Z
            # X=width, Y=height, Z=depth
            bobbin = (
                cq.Workplane("XY")
                .box(total_width * 2, total_height, total_depth * 2)
            )
            neg_ww = (
                cq.Workplane("XY")
                .box(total_width * 2, ww_height, total_depth * 2)
            )
            # Column outer = col_width (already includes thickness)
            central_col = (
                cq.Workplane("XY")
                .box(col_width * 2, ww_height, col_depth * 2)
            )
            # Inner hole = core column = col_width - thickness
            central_hole = (
                cq.Workplane("XY")
                .box((col_width - col_thickness) * 2, total_height, (col_depth - col_thickness) * 2)
            )
            
            # Subtract operations (same as Ansyas logic)
            neg_ww_cut = neg_ww.cut(central_col)
            bobbin = bobbin.cut(neg_ww_cut)
            bobbin = bobbin.cut(central_hole)

        # Scale back to meters
        final_shape = bobbin.val()
        scaled_shape = final_shape.scale(1 / SCALE)

        return cq.Workplane("XY").add(scaled_shape)

    def _create_toroidal_turn(
        self,
        turn_description: TurnDescription,
        wire_description: WireDescription,
        bobbin_description: BobbinProcessedDescription,
    ) -> cq.Workplane:
        """Create a toroidal turn by sweeping a wire cross-section along a
        closed rounded-rectangle path, same approach as concentric rectangular.

        Toroidal coordinate system (MAS polar, converted to Cartesian XZ):
        - Y axis: Core axis (toroid revolves around Y)
        - XZ plane: The radial plane where inner/outer wire positions live
        - coordinates[0], coordinates[1] are Cartesian (x, z) of inner wire
        - additionalCoordinates[0] gives (x, z) of outer wire

        The turn path is a rounded rectangle in a plane that contains the
        Y axis and the wire's angular position.  The four straight segments:
        - Inner vertical (through core hole, along Y)
        - Top radial (across top of core, from inner to outer)
        - Outer vertical (outside core, along Y)
        - Bottom radial (across bottom of core, from outer to inner)

        For multi-layer windings the outer wire may be angularly offset from
        the inner wire; the top/bottom radial segments will tilt accordingly.
        """
        from OCP.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Circ, gp_Trsf, gp_Vec
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
        from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipe
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol, BRepPrimAPI_MakePrism
        from OCP.GC import GC_MakeArcOfCircle
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakePolygon
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS, TopoDS_Compound
        import cadquery as cq

        SCALE = self.SCALE
        WIRE_POLYGON_SEGMENTS = self.WIRE_POLYGON_SEGMENTS

        # Wire dimensions
        is_rectangular_wire = wire_description.wire_type == WireType.rectangular
        if is_rectangular_wire:
            if turn_description.dimensions and len(turn_description.dimensions) >= 2:
                wire_width = turn_description.dimensions[0] * SCALE
                wire_height = turn_description.dimensions[1] * SCALE
            else:
                wire_width = (wire_description.outer_width or wire_description.conducting_width or 0.001) * SCALE
                wire_height = (wire_description.outer_height or wire_description.conducting_height or 0.001) * SCALE
            wire_radius = min(wire_width, wire_height) / 2.0
        else:
            wire_diameter = (wire_description.conducting_diameter or wire_description.outer_diameter or 0.001) * SCALE
            wire_radius = wire_diameter / 2.0
            wire_width = wire_diameter
            wire_height = wire_diameter

        # Bobbin dimensions
        half_depth = bobbin_description.column_depth * SCALE

        # Wire positions directly from turn coordinates (Cartesian XY plane)
        coords = turn_description.coordinates
        ix = coords[0] * SCALE  # X
        iy = coords[1] * SCALE  # Y
        inner_r = math.sqrt(ix**2 + iy**2)

        # Outer wire position directly from additionalCoordinates
        add_coords = turn_description.additional_coordinates
        if add_coords and len(add_coords) > 0 and len(add_coords[0]) >= 2:
            ac = add_coords[0]
            ox = ac[0] * SCALE
            oy = ac[1] * SCALE
        else:
            ww_rh = (bobbin_description.winding_window_radial_height or 0.003) * SCALE
            ratio = (inner_r + ww_rh) / inner_r if inner_r > 1e-9 else 2.0
            ox = ix * ratio
            oy = iy * ratio
        outer_r = math.sqrt(ox**2 + oy**2)

        # Meshing gap: reduce wire_width by 1% to create small clearances
        # between adjacent turns and between turns and the core. Without
        # this, toroidal turns are packed at zero gap (touching surfaces),
        # which causes gmsh's OCC fragmentation to fail with "overlapping
        # facets". The 1% reduction is small enough to not affect simulation
        # accuracy but creates the clearance needed for robust meshing.
        # Set to 0 to disable (e.g., for meshers that handle touching surfaces).
        meshing_gap_fraction = 0.10
        wire_width = wire_width * (1.0 - meshing_gap_fraction)
        if not is_rectangular_wire:
            wire_radius = wire_width / 2

        # Corner arc radius = wire_width/2
        br = wire_width / 2

        # Inner layers need extra Z clearance so outer layers can pass over.
        ww_radial_height = (bobbin_description.winding_window_radial_height or 0.003) * SCALE
        layer_clearance = max(ww_radial_height - inner_r - max(wire_width, wire_height) / 2, 0)

        tube_length = half_depth + layer_clearance
        radial_height = tube_length + br

        # Angular position of inner wire for final rotation
        inner_angle_rad = math.atan2(iy, ix)

        # Build the path in XZ plane at Y=0 (planar), then rotate.
        # Planar paths avoid Frenet frame issues that cause gmsh errors.
        # Same approach as concentric rectangular column turns.
        wire_x_inner = -inner_r
        wire_x_outer = -outer_r
        icr_x = wire_x_inner - br
        ocr_x = wire_x_outer + br

        # Planar path in XZ at Y=0 (rounded rectangle)
        pts = [
            gp_Pnt(wire_x_inner, 0, -tube_length),      # 0: inner bottom
            gp_Pnt(wire_x_inner, 0,  tube_length),       # 1: inner top
            gp_Pnt(icr_x, 0,  radial_height),            # 2: after top inner corner
            gp_Pnt(ocr_x, 0,  radial_height),            # 3: before top outer corner
            gp_Pnt(wire_x_outer, 0,  tube_length),       # 4: outer top
            gp_Pnt(wire_x_outer, 0, -tube_length),       # 5: outer bottom
            gp_Pnt(ocr_x, 0, -radial_height),            # 6: after bot outer corner
            gp_Pnt(icr_x, 0, -radial_height),            # 7: before bot inner corner
        ]

        # Corner centers in XZ plane
        cc_ti = gp_Pnt(icr_x, 0,  tube_length)
        cc_bi = gp_Pnt(icr_x, 0, -tube_length)
        cc_to = gp_Pnt(ocr_x, 0,  tube_length)
        cc_bo = gp_Pnt(ocr_x, 0, -tube_length)

        # Arc normal: -Y (perpendicular to XZ plane, same direction as concentric)
        arc_normal = gp_Dir(0, -1, 0)

        def _dist(a, b):
            return math.sqrt((a.X()-b.X())**2 + (a.Y()-b.Y())**2 + (a.Z()-b.Z())**2)

        # Corner xref directions for XZ plane arcs
        corners_data = [
            (cc_ti, gp_Dir(1, 0, 0)),    # top inner
            (cc_to, gp_Dir(0, 0, 1)),     # top outer
            (cc_bo, gp_Dir(-1, 0, 0)),    # bot outer
            (cc_bi, gp_Dir(0, 0, -1)),    # bot inner
        ]

        # Build closed wire path in XZ plane (planar — meshable)
        wb = BRepBuilderAPI_MakeWire()
        segment_pairs = [(0,1,0), (2,3,1), (4,5,2), (6,7,3)]
        for (i_s, i_e, c_i) in segment_pairs:
            if _dist(pts[i_s], pts[i_e]) > 1e-9:
                wb.Add(BRepBuilderAPI_MakeEdge(pts[i_s], pts[i_e]).Edge())
            cc, cxref = corners_data[c_i]
            circ = gp_Circ(gp_Ax2(cc, arc_normal, cxref), br)
            arc = GC_MakeArcOfCircle(circ, pts[i_e], pts[(i_e+1)%8], True).Value()
            wb.Add(BRepBuilderAPI_MakeEdge(arc).Edge())

        spine = wb.Wire()

        # Profile at pts[0], tangent = +Z
        if is_rectangular_wire:
            p0 = pts[0]
            half_ww = wire_width / 2
            half_wh = wire_height / 2
            # In XZ plane: wide along X (radial), thin along Y
            rect = BRepBuilderAPI_MakePolygon(
                gp_Pnt(p0.X() - half_ww, -half_wh, p0.Z()),
                gp_Pnt(p0.X() + half_ww, -half_wh, p0.Z()),
                gp_Pnt(p0.X() + half_ww,  half_wh, p0.Z()),
                gp_Pnt(p0.X() - half_ww,  half_wh, p0.Z()),
                True).Wire()
            profile_face = BRepBuilderAPI_MakeFace(rect).Face()
        else:
            # Polygonal approximation of circle for cleaner OCC booleans/meshing.
            # Offset by half a segment to avoid vertex alignment with spine plane
            # (which causes gp_Dir zero-norm error at arc-straight junctions).
            p0 = pts[0]
            poly = BRepBuilderAPI_MakePolygon()
            offset = math.pi / WIRE_POLYGON_SEGMENTS  # half-segment rotation
            for seg in range(WIRE_POLYGON_SEGMENTS):
                angle = 2 * math.pi * seg / WIRE_POLYGON_SEGMENTS + offset
                px = p0.X() + wire_radius * math.cos(angle)
                py = wire_radius * math.sin(angle)
                poly.Add(gp_Pnt(px, py, p0.Z()))
            poly.Close()
            profile_face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()

        pipe = BRepOffsetAPI_MakePipe(spine, profile_face)
        turn_shape = pipe.Shape()

        # Sew if invalid (rectangular wire Frenet rotation at corners)
        from OCP.BRepCheck import BRepCheck_Analyzer
        if not BRepCheck_Analyzer(turn_shape, True).IsValid():
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
            from OCP.TopAbs import TopAbs_SHELL, TopAbs_FACE
            from OCP.TopExp import TopExp_Explorer
            sew = BRepBuilderAPI_Sewing(1e-6)
            exp = TopExp_Explorer(turn_shape, TopAbs_FACE)
            while exp.More():
                sew.Add(exp.Current())
                exp.Next()
            sew.Perform()
            sewn = sew.SewedShape()
            shell_exp = TopExp_Explorer(sewn, TopAbs_SHELL)
            if shell_exp.More():
                turn_shape = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shell_exp.Current())).Solid()

        turn = cq.Workplane("XY").add(cq.Shape(turn_shape))

        # Rotate from construction plane (XZ at 180°) to actual angular position
        rotation_deg = math.degrees(inner_angle_rad) - 180.0
        if abs(rotation_deg) > 0.001:
            turn = turn.rotate((0, 0, 0), (0, 0, 1), rotation_deg)

        # Scale back to meters
        final_shape = turn.val()
        scaled_shape = final_shape.scale(1 / SCALE)

        return cq.Workplane("XY").add(scaled_shape)

    def get_magnetic(
        self,
        magnetic_data: dict,
        project_name: str = "Magnetic",
        output_path: str = None,
        export_files: bool = True,
        include_bobbin: bool = True,
    ):
        """Build complete magnetic assembly (core + coil).

        Args:
            magnetic_data: MAS format magnetic data with 'core' and 'coil' keys
            project_name: Name for the output files
            output_path: Directory for output files
            export_files: Whether to export STEP/STL files

        Returns:
            Tuple of (step_path, stl_path) or compound if export_files is False
        """
        if output_path is None:
            output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"

        os.makedirs(output_path, exist_ok=True)
        project_name = project_name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
        
        core_pieces = []
        turn_pieces = []
        bobbin_geom = None

        # Detect if this is a toroidal core
        is_toroidal = False

        # Build core
        core_data = magnetic_data.get("core", {})
        geometrical_description = core_data.get("geometricalDescription", [])
        if geometrical_description:
            for index, geometrical_part in enumerate(geometrical_description):
                if geometrical_part["type"] == "toroidal":
                    is_toroidal = True
                if geometrical_part["type"] in ["half set", "toroidal"]:
                    shape_data = geometrical_part["shape"]
                    family = shape_data.get("family", "").lower()
                    # Check if shape family is 't' (toroidal)
                    if family == "t":
                        is_toroidal = True
                    part_builder = CadQueryBuilder().factory(shape_data)

                    piece = part_builder.get_piece(
                        data=copy.deepcopy(shape_data),
                        name=f"Core_{index}",
                        save_files=False,
                        export_files=False
                    )

                    # Apply machining (subtractive gaps) before rotation
                    if 'machining' in geometrical_part and geometrical_part['machining'] is not None:
                        for machining in geometrical_part['machining']:
                            piece = part_builder.apply_machining(
                                piece=piece,
                                machining=machining,
                                dimensions=flatten_dimensions(shape_data))

                    # rotation[0] around X, rotation[1] around Y, rotation[2] around Z
                    piece = piece.rotate((1, 0, 0), (-1, 0, 0), geometrical_part['rotation'][0] / math.pi * 180)
                    piece = piece.rotate((0, 1, 0), (0, -1, 0), geometrical_part['rotation'][1] / math.pi * 180)
                    piece = piece.rotate((0, 0, 1), (0, 0, -1), geometrical_part['rotation'][2] / math.pi * 180)

                    # Apply translation
                    piece = piece.translate(convert_axis(geometrical_part['coordinates']))

                    # Separate half-set pieces by subtractive gap length.
                    # Only shift if there's actual machining (subtractive gap).
                    # Residual gaps (5-10µm) are too small to model geometrically
                    # and cause meshing failures. Toroidal cores have no half-sets.
                    if geometrical_part['type'] in ['half set']:
                        has_machining = any(
                            gp.get('machining') for gp in geometrical_description
                        )
                        if has_machining:
                            max_gap = 0
                            for gp in geometrical_description:
                                if gp.get('machining'):
                                    for m in gp['machining']:
                                        max_gap = max(max_gap, m.get('length', 0))
                            if max_gap > 0:
                                shift = max_gap / 2
                                if geometrical_part['rotation'][0] > 0:
                                    piece = piece.translate((0, shift, 0))
                                else:
                                    piece = piece.translate((0, -shift, 0))

                    core_pieces.append(piece)

        # Build coil turns
        coil_data = magnetic_data.get("coil", {})
        bobbin_data = coil_data.get("bobbin", {})
        if isinstance(bobbin_data, str):
            # Bobbin is a reference string, no processed description available
            bobbin_processed = BobbinProcessedDescription()
        else:
            bobbin_processed_data = bobbin_data.get("processedDescription", {})
            bobbin_processed = BobbinProcessedDescription.from_dict(bobbin_processed_data)

        # Build bobbin geometry if not toroidal, bobbin has thickness, and requested
        if not is_toroidal and include_bobbin:
            bobbin_geom = self.get_bobbin(bobbin_processed)

        # Get wire info from functionalDescription
        wire_desc = WireDescription(WireType.round)  # default
        functional_desc = coil_data.get("functionalDescription", [])
        if functional_desc:
            wire_data = functional_desc[0].get("wire", {})
            if wire_data:
                wire_desc = WireDescription.from_dict(wire_data)

        # In MAS format, turnsDescription is at coil level, not inside sections/layers
        turns_data = coil_data.get("turnsDescription", [])
        for turn_data in turns_data:
            turn_desc = TurnDescription.from_dict(turn_data)

            # Get wire dimensions from turn data if available
            if turn_data.get("dimensions"):
                dims = turn_data["dimensions"]
                # dimensions is [width, height] or [diameter, diameter] for round wire
                if len(dims) >= 2:
                    wire_desc = WireDescription(
                        wire_type=WireType.round if turn_data.get("crossSectionalShape", "round") == "round" else WireType.rectangular, outer_diameter=dims[0], conducting_diameter=dims[0]
                    )

            turn_geom = self.get_turn(turn_desc, wire_desc, bobbin_processed, is_toroidal=is_toroidal)
            turn_pieces.append(turn_geom)

        # Subtract core and turns from bobbin to avoid overlapping volumes.
        if bobbin_geom is not None:
            for core_piece in core_pieces:
                try:
                    bobbin_geom = bobbin_geom.cut(core_piece)
                except Exception:
                    pass
            for turn in turn_pieces:
                try:
                    bobbin_geom = bobbin_geom.cut(turn)
                except Exception:
                    pass

        # Assemble all pieces
        all_pieces = core_pieces[:]
        if bobbin_geom is not None:
            all_pieces.append(bobbin_geom)
        all_pieces.extend(turn_pieces)

        # Export
        if export_files and all_pieces:
            from cadquery import exporters

            scaled_pieces = []
            for piece in all_pieces:
                for o in piece.objects:
                    scaled_pieces.append(o.scale(1000))

            compound = cq.Compound.makeCompound(scaled_pieces)

            step_path = f"{output_path}/{project_name}.step"
            stl_path = f"{output_path}/{project_name}.stl"
            exporters.export(compound, step_path, "STEP")
            # Use configurable tessellation parameters for STL
            exporters.export(
                compound,
                stl_path,
                "STL",
                tolerance=TESSELLATION_LINEAR_TOLERANCE,
                angularTolerance=get_angular_tolerance()
            )
            return step_path, stl_path
        elif all_pieces:
            return all_pieces
        else:
            return None, None

    class IPiece(metaclass=ABCMeta):
        def __init__(self):
            self.output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"

        def set_output_path(self, output_path):
            self.output_path = output_path

        @staticmethod
        def create_sketch():
            return cq.Sketch()

        @staticmethod
        def extrude_sketch(sketch, part_name, height):
            result = cq.Workplane().placeSketch(sketch).extrude(height)

            return result

        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]
            piece = piece.translate((0, 0, -dimensions["B"]))
            return piece

        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F"]}

        def get_plate(self, data, save_files=False, export_files=True):
            import FreeCAD

            try:
                project_name = f"{data['name']}_plate".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                data["dimensions"] = flatten_dimensions(data)

                close_file_after_finishing = False
                if FreeCAD.ActiveDocument is None:
                    close_file_after_finishing = True
                    FreeCAD.newDocument(project_name)
                document = FreeCAD.ActiveDocument

                sketch = self.get_shape_base(data)

                document = FreeCAD.ActiveDocument
                document.recompute()

                part_name = "plate"

                plate = self.extrude_sketch(sketch=sketch, part_name=part_name, height=data["dimensions"]["B"] - data["dimensions"]["D"])

                document.recompute()
                if export_files:
                    from cadquery import exporters

                    scaled_pieces_to_export = []
                    for piece in [plate]:
                        for o in piece.objects:
                            scaled_pieces_to_export.append(o.scale(1000))

                    scaled_pieces_to_export = cq.Compound.makeCompound(scaled_pieces_to_export)

                    exporters.export(scaled_pieces_to_export, f"{self.output_path}/{project_name}.step", "STEP")
                    exporters.export(scaled_pieces_to_export, f"{self.output_path}/{project_name}.stl", "STL")
                    return f"{self.output_path}/{project_name}.step", f"{self.output_path}/{project_name}.stl"

                if save_files:
                    document.saveAs(f"{self.output_path}/{project_name}.FCStd")

                if not close_file_after_finishing:
                    return plate
            except:  # noqa: E722
                return None, None

        def get_piece(self, data, name="Piece", save_files=False, export_files=True):
            try:
                project_name = f"{data['name']}_piece".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

                data["dimensions"] = flatten_dimensions(data)

                sketch = self.get_shape_base(data)

                part_name = "piece"

                base = self.extrude_sketch(sketch=sketch, part_name=part_name, height=data["dimensions"]["B"] if data["family"] != "t" else data["dimensions"]["C"])

                negative_winding_window = self.get_negative_winding_window(data["dimensions"])

                if negative_winding_window is None:
                    piece = base
                else:
                    piece = base - negative_winding_window

                piece_with_extra = self.get_shape_extras(data, piece)

                # Core shapes are built with sketch in XY, extruded along Z.
                # For concentric cores: rotate -90° around X so the extrusion
                # axis moves from Z to Y (column axis along Y, winding window in XY).
                # Toroidal cores are already correct (XY=radial plane, Z=thickness).
                if data.get("family", "").lower() != 't':
                    piece_with_extra = piece_with_extra.rotate((0, 0, 0), (1, 0, 0), -90)

                pathlib.Path(self.output_path).mkdir(parents=True, exist_ok=True)

                if export_files:
                    from cadquery import exporters

                    scaled_piece_with_extra = piece_with_extra.newObject([o.scale(1000) for o in piece_with_extra.objects])
                    exporters.export(scaled_piece_with_extra, f"{self.output_path}/{project_name}.step", "STEP")
                    exporters.export(scaled_piece_with_extra, f"{self.output_path}/{project_name}.stl", "STL")
                    return f"{self.output_path}/{project_name}.step", f"{self.output_path}/{project_name}.stl"
                else:
                    return piece_with_extra

            except:  # noqa: E722
                return (None, None) if export_files else None

        @staticmethod
        def _hex_to_rgb(hex_color):
            """Convert hex color string like '#d4d4d4' to RGB tuple (212, 212, 212)."""
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        def get_piece_technical_drawing(self, data, colors=None, save_files=False):
            try:
                from cadquery.occ_impl.exporters.svg import getSVG

                if colors is None:
                    colors = {"projection_color": "#000000", "dimension_color": "#000000"}

                project_name = f"{data['name']}_piece_scaled".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

                piece = self.get_piece(data=copy.deepcopy(data), save_files=False, export_files=False)
                if piece is None:
                    return {"top_view": None, "front_view": None}

                scaled_piece = piece.newObject([o.scale(1000) for o in piece.objects])

                pathlib.Path(self.output_path).mkdir(parents=True, exist_ok=True)

                stroke_color = self._hex_to_rgb(colors.get("projection_color", "#000000"))
                svg_opts = {
                    "width": 800,
                    "height": 600,
                    "strokeWidth": 0.5,
                    "strokeColor": stroke_color,
                    "showHidden": True,
                }

                # Top view
                top_svg = getSVG(scaled_piece.val(), {**svg_opts, "projectionDir": (0, 0, 1)})

                top_path = f"{self.output_path}/{project_name}_TopView.svg"
                with open(top_path, "w", encoding="utf-8") as f:
                    f.write(top_svg)

                # Front view
                front_svg = getSVG(scaled_piece.val(), {**svg_opts, "projectionDir": (0, 1, 0)})

                front_path = f"{self.output_path}/{project_name}_FrontView.svg"
                with open(front_path, "w", encoding="utf-8") as f:
                    f.write(front_svg)

                return {"top_view": top_svg, "front_view": front_svg}
            except Exception:
                return {"top_view": None, "front_view": None}

        def add_dimensions_and_export_view(self, data, original_dimensions, view, project_name, margin, colors, save_files, piece):
            """Generate an SVG view for a piece.

            Args:
                data: Shape data dict with 'dimensions' and 'family'.
                original_dimensions: Original dimensions for label values.
                view: Dict with 'Name' ('TopView'/'FrontView'), 'X', 'Y' keys.
                project_name: Name for output files.
                margin: Margin around the drawing.
                colors: Dict with 'projection_color' and 'dimension_color'.
                save_files: Whether to write SVG to disk.
                piece: CadQuery Workplane with the piece geometry.

            Returns:
                SVG string, or None on failure.
            """
            try:
                from cadquery.occ_impl.exporters.svg import getSVG

                view_name = view["Name"]

                # Determine projection direction based on view
                if view_name == "TopView":
                    projection_dir = (0, 0, 1)
                else:
                    projection_dir = (0, 1, 0)

                # Scale piece for SVG generation
                scaled_piece = piece.newObject([o.scale(1000) for o in piece.objects])

                stroke_color = self._hex_to_rgb(colors.get("projection_color", "#000000"))
                svg_opts = {
                    "width": 800,
                    "height": 600,
                    "strokeWidth": 0.5,
                    "strokeColor": stroke_color,
                    "showHidden": True,
                    "projectionDir": projection_dir,
                }

                svg_data = getSVG(scaled_piece.val(), svg_opts)

                if save_files:
                    pathlib.Path(self.output_path).mkdir(parents=True, exist_ok=True)
                    with open(f"{self.output_path}/{project_name}_{view_name}.svg", "w", encoding="utf-8") as f:
                        f.write(svg_data)

                return svg_data

            except Exception:
                return None

        @abstractmethod
        def get_shape_base(self, data):
            raise NotImplementedError

        @abstractmethod
        def get_negative_winding_window(self, dimensions):
            raise NotImplementedError

        def apply_machining(self, piece, machining, dimensions):
            length = dimensions["A"]
            x_coordinate = dimensions["A"] / 2
            if machining["coordinates"][0] == 0:
                width = dimensions["F"]
                length = dimensions["F"]
                y_coordinate = 0
                x_coordinate = 0
            else:
                width = dimensions["A"] / 2
                if machining["coordinates"][0] < 0:
                    y_coordinate = 0
                if machining["coordinates"][0] > 0:
                    y_coordinate = 0

            height = machining["length"]
            original_tool = cq.Workplane().box(width, length, height).translate((x_coordinate, y_coordinate, machining["coordinates"][1]))

            if machining["coordinates"][0] == 0:
                tool = original_tool
            else:
                central_column_width = dimensions["F"] * 1.001
                length = central_column_width
                width = central_column_width
                height = machining["length"]
                central_column_tool = cq.Workplane().box(width, length, height).translate((0, 0, (machining["coordinates"][1] - machining["length"] / 2)))

                tool = original_tool - central_column_tool

            machined_piece = piece - tool

            return machined_piece

    class P(IPiece):
        def get_dimensions_and_subtypes(self):
            return shape_configs.P_DIMENSIONS_AND_SUBTYPES

        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]
            familySubtype = data["familySubtype"]

            if familySubtype == "1" or familySubtype == "2":
                length = (dimensions["A"] - dimensions["F"]) / 2
                width = dimensions["G"]
                height = dimensions["D"]
                translate = (length / 2 + dimensions["F"] / 2, 0, height / 2 + dimensions["B"] - dimensions["D"])

                lateral_right_cut_box = cq.Workplane().box(length, width, height).tag("lateral_right_cut_box").translate(translate)

                translate = (-(length / 2 + dimensions["F"] / 2), 0, height / 2 + dimensions["B"] - dimensions["D"])
                lateral_left_cut_box = cq.Workplane().box(length, width, height).tag("lateral_left_cut_box").translate(translate)

                piece = piece - lateral_right_cut_box
                piece = piece - lateral_left_cut_box

                if familySubtype == "2":
                    if "C" in dimensions and dimensions["C"] > 0:
                        c = dimensions["C"] / 2
                    else:
                        c = utils.decimal_floor(dimensions["E"] * math.cos(math.asin(dimensions["G"] / dimensions["E"])) / 2, 6) * 0.95

                    length = dimensions["A"] / 2 - c
                    width = dimensions["G"]
                    height = dimensions["B"]
                    translate = (length / 2 + c, 0, height / 2)

                    right_dent_box = cq.Workplane().box(length, width, height).tag("right_dent_box").translate(translate)
                    piece = piece - right_dent_box

                    translate = (-(length / 2 + c), 0, height / 2)
                    left_dent_box = cq.Workplane().box(length, width, height).tag("left_dent_box").translate(translate)
                    piece = piece - left_dent_box
            elif familySubtype == "3":
                hole_width = (dimensions["G"]) / 2
                hole_length = (dimensions["E"] - dimensions["F"]) / 2 - hole_width
                hole_height = dimensions["B"]
                translate = (hole_width / 2 + hole_length / 2 + dimensions["F"] / 2, 0, 0)
                hole = cq.Workplane().box(hole_length, hole_width, hole_height).tag("hole").translate(translate)
                translate = (hole_width / 2 + dimensions["F"] / 2, 0, 0)
                hole_round_1 = cq.Workplane().cylinder(hole_height, hole_width / 2).tag("hole_round_1").translate(translate)
                hole = hole + hole_round_1
                translate = (hole_width / 2 + hole_length + dimensions["F"] / 2, 0, 0)
                hole_round_2 = cq.Workplane().cylinder(hole_height, hole_width / 2).tag("hole_round_2").translate(translate)
                hole = hole + hole_round_1
                hole = hole + hole_round_2
                piece = piece - hole

                translate = (-(hole_width + hole_length + dimensions["F"]), 0, 0)
                hole = hole.translate(translate)
                piece = piece - hole

            if "H" in dimensions and dimensions["H"] > 0:
                hole = cq.Workplane().cylinder(dimensions["B"], dimensions["H"] / 2).tag("hole").translate((0, 0, dimensions["B"] / 2))
                piece = piece - hole

            piece = piece.translate((0, 0, -dimensions["B"]))

            return piece

        def get_shape_base(self, data):
            dimensions = data["dimensions"]

            a = dimensions["A"] / 2

            sketch = cq.Sketch().circle(a, mode="a", tag="central_circle")
            return sketch

        def get_negative_winding_window(self, dimensions):

            winding_window_cylinder = (
                cq.Workplane().cylinder(dimensions["D"], dimensions["E"] / 2).tag("winding_window_cylinder").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )

            central_column_cylinder = (
                cq.Workplane().cylinder(dimensions["D"], dimensions["F"] / 2).tag("central_column_cylinder").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )
            negative_winding_window = winding_window_cylinder - central_column_cylinder
            return negative_winding_window

    class Pq(P):
        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F", "G"]}

        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]
            piece = piece.translate((0, 0, -dimensions["B"]))
            return piece

        def get_shape_base(self, data):
            dimensions = data["dimensions"]

            if "L" not in dimensions or dimensions["L"] == 0:
                dimensions["L"] = dimensions["F"] + (dimensions["C"] - dimensions["F"]) / 3

            if "J" not in dimensions or dimensions["J"] == 0:
                dimensions["J"] = dimensions["F"] / 2

            if "G" in dimensions:
                g_angle = math.asin(dimensions["G"] / dimensions["E"])
            else:
                g_angle = math.asin((dimensions["E"] - ((dimensions["E"] - dimensions["F"]) / 2)) / dimensions["E"])

            c = dimensions["C"] / 2
            a = dimensions["A"] / 2
            e = dimensions["E"] / 2
            f = dimensions["F"] / 2

            sketch = (
                cq.Sketch()
                .circle(f, mode="a", tag="central_circle")
                .segment((a, -c), (a, c), "top_line")
                .segment((a, c), (e * math.sin(g_angle), c), "side_top_right_line")
                .segment((a, -c), (e * math.sin(g_angle), -c), "side_top_left_line")
                .segment((e * math.sin(g_angle), c), (e * math.sin(g_angle), e * math.cos(g_angle)), "side_corner_top_right_line")
                .segment((e * math.sin(g_angle), -c), (e * math.sin(g_angle), -e * math.cos(g_angle)), "side_corner_top_left_line")
                .segment((e * math.sin(g_angle), e * math.cos(g_angle)), (dimensions["J"] / 2, dimensions["L"] / 2), "long_top_right_line")
                .segment((e * math.sin(g_angle), -e * math.cos(g_angle)), (dimensions["J"] / 2, -dimensions["L"] / 2), "long_top_left_line")
                .segment((dimensions["J"] / 2, dimensions["L"] / 2), (dimensions["J"] / 4, dimensions["L"] / 4), "short_top_right_line")
                .segment((dimensions["J"] / 2, -dimensions["L"] / 2), (dimensions["J"] / 4, -dimensions["L"] / 4), "short_top_left_line")
                .segment((dimensions["J"] / 4, dimensions["L"] / 4), (dimensions["J"] / 4, -dimensions["L"] / 4), "join_right")
                .constrain("top_line", "Fixed", None)
                .constrain("top_line", "Orientation", (0, 1))
                .constrain("long_top_right_line", "short_top_right_line", "Coincident", None)
                .constrain("long_top_left_line", "short_top_left_line", "Coincident", None)
                .segment((-a, -c), (-a, c), "bottom_line")
                .segment(
                    (
                        -a,
                        c,
                    ),
                    (-e * math.sin(g_angle), c),
                    "side_bottom_right_line",
                )
                .segment(
                    (
                        -a,
                        -c,
                    ),
                    (-e * math.sin(g_angle), -c),
                    "side_bottom_left_line",
                )
                .segment((-e * math.sin(g_angle), c), (-e * math.sin(g_angle), e * math.cos(g_angle)), "side_corner_bottom_right_line")
                .segment((-e * math.sin(g_angle), -c), (-e * math.sin(g_angle), -e * math.cos(g_angle)), "side_corner_bottom_left_line")
                .segment((-e * math.sin(g_angle), e * math.cos(g_angle)), (-dimensions["J"] / 2, dimensions["L"] / 2), "long_bottom_right_line")
                .segment((-e * math.sin(g_angle), -e * math.cos(g_angle)), (-dimensions["J"] / 2, -dimensions["L"] / 2), "long_bottom_left_line")
                .segment((-dimensions["J"] / 2, dimensions["L"] / 2), (-dimensions["J"] / 4, dimensions["L"] / 4), "short_bottom_right_line")
                .segment((-dimensions["J"] / 2, -dimensions["L"] / 2), (-dimensions["J"] / 4, -dimensions["L"] / 4), "short_bottom_left_line")
                .segment((-dimensions["J"] / 4, dimensions["L"] / 4), (-dimensions["J"] / 4, -dimensions["L"] / 4), "join_left")
                .constrain("bottom_line", "Fixed", None)
                .constrain("bottom_line", "Orientation", (0, 1))
                .constrain("long_bottom_right_line", "short_bottom_right_line", "Coincident", None)
                .constrain("long_bottom_left_line", "short_bottom_left_line", "Coincident", None)
                .constrain("short_top_right_line", "Fixed", None)
                .constrain("short_top_left_line", "Fixed", None)
                .constrain("short_bottom_right_line", "Fixed", None)
                .constrain("short_bottom_left_line", "Fixed", None)
            )

            sketch = sketch.solve().assemble()
            return sketch

    class Rm(P):
        def get_dimensions_and_subtypes(self):
            return shape_configs.RM_DIMENSIONS_AND_SUBTYPES

        def get_shape_base(self, data):
            dimensions = data["dimensions"]
            familySubtype = data["familySubtype"]

            p = math.sqrt(2) * dimensions["J"] - dimensions["A"]
            alpha = math.asin(dimensions["G"] / dimensions["E"])
            z = dimensions["E"] / 2 * math.cos(alpha)
            c = dimensions["C"] / 2
            g = dimensions["G"] / 2
            a = dimensions["A"] / 2
            e = dimensions["E"] / 2
            f = dimensions["F"] / 2

            if familySubtype == "1":
                t = 0
                n = (z - c) / g
                r = (a + p / 2 - c + n * t) / (n + 1)
                s = n * r + c
            elif familySubtype == "2":
                t = f * math.sin(math.acos(c / f))
                n = (z - c) / g
                r = (a + p / 2 - c + n * t) / (n + 1)
                s = n * r + c
            elif familySubtype == "3":
                t = c - e * math.cos(math.asin(g / e)) + g
                n = (z - c) / g
                r = (a + p / 2 - c + n * t) / (n + 1)
                s = n * r + c
            elif familySubtype == "4":
                t = 0
                n = 1
                r = (a + p / 2 - c + n * t) / (n + 1)
                s = n * r + c

            sketch = (
                cq.Sketch()
                .segment((a, -p / 2), (a, p / 2), "top_line")
                .segment((a, p / 2), (r, s), "top_right_line_45_degrees")
                .segment((r, s), (t, c), "top_right_line_x_degrees")
                .segment((-t, c), (-r, s), "bottom_right_line_x_degrees")
                .segment((-r, s), (-a, p / 2), "bottom_right_line_45_degrees")
                .segment((-a, p / 2), (-a, -p / 2), "bottom_line")
                .segment((-a, -p / 2), (-r, -s), "bottom_left_line_45_degrees")
                .segment((-r, -s), (-t, -c), "bottom_left_line_x_degrees")
                .segment((t, -c), (r, -s), "top_left_line_x_degrees")
                .segment((r, -s), (a, -p / 2), "top_left_line_45_degrees")
                .constrain("top_line", "Fixed", None)
                .constrain("bottom_line", "Fixed", None)
                .constrain("top_line", "Orientation", (0, 1))
                .constrain("bottom_line", "Orientation", (0, 1))
                .constrain("top_right_line_45_degrees", "top_right_line_x_degrees", "Coincident", None)
                .constrain("bottom_right_line_x_degrees", "bottom_right_line_45_degrees", "Coincident", None)
                .constrain("top_left_line_x_degrees", "top_left_line_45_degrees", "Coincident", None)
                .constrain("bottom_left_line_45_degrees", "bottom_left_line_x_degrees", "Coincident", None)
            )

            if familySubtype == "3":
                sketch = sketch.segment((t, c), (-t, c), "right_line")
                sketch = sketch.segment((-t, -c), (t, -c), "left_line")
                sketch = sketch.constrain("right_line", "Fixed", None)
                sketch = sketch.constrain("left_line", "Fixed", None)
                sketch = sketch.constrain("right_line", "left_line", "Angle", 0)
                sketch = sketch.constrain("top_right_line_x_degrees", "right_line", "Coincident", None)
                sketch = sketch.constrain("right_line", "bottom_right_line_x_degrees", "Coincident", None)
                sketch = sketch.constrain("left_line", "top_left_line_x_degrees", "Coincident", None)
                sketch = sketch.constrain("bottom_left_line_x_degrees", "left_line", "Coincident", None)
            if familySubtype == "4":
                sketch = sketch.constrain("bottom_left_line_x_degrees", "top_left_line_x_degrees", "Coincident", None)
                sketch = sketch.constrain("top_right_line_x_degrees", "bottom_right_line_x_degrees", "Coincident", None)

            if familySubtype == "3" or familySubtype == "4":
                sketch = sketch.constrain("top_line", "top_right_line_45_degrees", "Coincident", None)
                sketch = sketch.constrain("top_left_line_45_degrees", "top_line", "Coincident", None)
                sketch = sketch.constrain("bottom_right_line_45_degrees", "bottom_line", "Coincident", None)
                sketch = sketch.constrain("bottom_line", "bottom_left_line_45_degrees", "Coincident", None)
                sketch = sketch.constrain("top_right_line_45_degrees", "top_right_line_x_degrees", "Angle", 90)
                sketch = sketch.constrain("bottom_right_line_45_degrees", "bottom_right_line_x_degrees", "Angle", 90)
                sketch = sketch.constrain("top_left_line_45_degrees", "top_left_line_x_degrees", "Angle", 90)
                sketch = sketch.constrain("bottom_left_line_45_degrees", "bottom_left_line_x_degrees", "Angle", 90)
                sketch = sketch.constrain("top_right_line_45_degrees", "top_line", "Angle", 270)
                sketch = sketch.constrain("top_left_line_45_degrees", "top_line", "Angle", 270)
                sketch = sketch.constrain("bottom_right_line_45_degrees", "bottom_line", "Angle", 270)
                sketch = sketch.constrain("bottom_left_line_45_degrees", "bottom_line", "Angle", 270)

            if c < f:
                assert 0
                sketch = sketch.circle(f, mode="a")

            sketch = sketch.solve().assemble()

            return sketch

        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]
            if "H" in dimensions and dimensions["H"] > 0:
                hole = cq.Workplane().cylinder(dimensions["B"], dimensions["H"] / 2).tag("hole").translate((0, 0, dimensions["B"] / 2))
                piece = piece - hole

            piece = piece.translate((0, 0, -dimensions["B"]))
            return piece

    class Pm(P):
        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F", "G", "H", "b", "t", "alpha"], 2: ["A", "B", "C", "D", "E", "F", "G", "H", "b", "t", "alpha"]}

        def get_shape_base(self, data):
            dimensions = data["dimensions"]
            familySubtype = data["familySubtype"]

            h = dimensions["H"] / 2
            c = dimensions["C"] / 2
            g = dimensions["G"] / 2
            a = dimensions["A"] / 2
            e = dimensions["E"] / 2
            f = dimensions["F"] / 2
            b = dimensions["b"] / 2
            t = dimensions["t"]

            if "alpha" not in dimensions or dimensions["alpha"] == 0:
                if familySubtype == "1":
                    dimensions["alpha"] = 120
                else:
                    dimensions["alpha"] = 90

            alpha = dimensions["alpha"] / 180 * math.pi

            beta = math.asin(g / e)
            gcos = e * math.cos(beta)

            wall_thickness = a - e
            # asin = a * math.sin(beta)
            # acos = a * math.cos(beta)

            external_slope = (gcos - c) / g
            a_corner_x = g + wall_thickness * math.sin(alpha / 2)
            a_corner_y = gcos + a_corner_x * external_slope
            alpha = dimensions["alpha"]

            if familySubtype == "1":
                sketch = (
                    cq.Sketch()
                    .arc((a_corner_x, -a_corner_y), (a, 0), (a_corner_x, a_corner_y), "top_arc")
                    .segment((a_corner_x, a_corner_y), (0, c), "side_top_left_line")
                    .segment((0, c), (-a_corner_x, a_corner_y), "side_bottom_left_line")
                    .arc((-a_corner_x, a_corner_y), (-a, 0), (-a_corner_x, -a_corner_y), "bottom_arc")
                    .segment((-a_corner_x, -a_corner_y), (0, -c), "side_bottom_right_line")
                    .segment((0, -c), (a_corner_x, -a_corner_y), "side_top_right_line")
                    .constrain("side_top_left_line", "side_top_right_line", "Angle", -alpha)
                    .constrain("side_bottom_left_line", "side_bottom_right_line", "Angle", alpha)
                    .constrain("top_arc", "Radius", a)
                    .constrain("bottom_arc", "Radius", a)
                    .constrain("bottom_arc", "top_arc", "Distance", (None, None, 0))
                    .constrain("top_arc", "side_top_left_line", "Coincident", None)
                    .constrain("side_top_right_line", "top_arc", "Coincident", None)
                    .constrain("side_bottom_left_line", "bottom_arc", "Coincident", None)
                    .constrain("bottom_arc", "side_bottom_right_line", "Coincident", None)
                    .constrain("side_top_left_line", "side_bottom_left_line", "Coincident", None)
                    .constrain("side_bottom_right_line", "side_top_right_line", "Coincident", None)
                )
            else:
                sketch = (
                    cq.Sketch()
                    .arc((a_corner_x, -a_corner_y / 1.3), (a, 0), (a_corner_x, a_corner_y / 1.3), "top_arc")
                    .segment((a_corner_x, a_corner_y / 1.3), (f / 2, c), "side_top_left_line")
                    .segment((f / 2, c), (0, c), "left_top_line")
                    .segment((0, c), (-f / 2, c), "left_bottom_line")
                    .segment((-f / 2, c), (-a_corner_x, a_corner_y / 1.3), "side_bottom_left_line")
                    .arc((-a_corner_x, a_corner_y / 1.3), (-a, 0), (-a_corner_x, -a_corner_y / 1.3), "bottom_arc")
                    .segment((-a_corner_x, -a_corner_y / 1.3), (-f / 2, -c), "side_bottom_right_line")
                    .segment((-f / 2, -c), (0, -c), "right_bottom_line")
                    .segment((0, -c), (f / 2, -c), "right_top_line")
                    .segment((f / 2, -c), (a_corner_x, -a_corner_y / 1.3), "side_top_right_line")
                    # .constrain("top_arc", "right_top_line", "Distance", (None, 0, c))
                    .constrain("top_arc", "side_top_left_line", "Coincident", None)
                    .constrain("side_top_left_line", "left_top_line", "Coincident", None)
                    .constrain("left_top_line", "left_bottom_line", "Coincident", None)
                    .constrain("left_bottom_line", "side_bottom_left_line", "Coincident", None)
                    .constrain("side_bottom_left_line", "bottom_arc", "Coincident", None)
                    .constrain("bottom_arc", "side_bottom_right_line", "Coincident", None)
                    .constrain("side_bottom_right_line", "right_bottom_line", "Coincident", None)
                    .constrain("right_bottom_line", "right_top_line", "Coincident", None)
                    .constrain("right_top_line", "side_top_right_line", "Coincident", None)
                    .constrain("side_top_right_line", "top_arc", "Coincident", None)
                    .constrain("top_arc", "Radius", a)
                    .constrain("bottom_arc", "Radius", a)
                    .constrain("bottom_arc", "top_arc", "Distance", (None, None, 0))
                    .constrain("side_bottom_right_line", "right_bottom_line", "Angle", 45)
                    .constrain("side_bottom_left_line", "left_bottom_line", "Angle", -45)
                    .constrain("side_top_right_line", "right_top_line", "Angle", -45)
                    .constrain("side_top_left_line", "left_top_line", "Angle", 45)
                    .constrain("side_top_left_line", "side_bottom_left_line", "Angle", alpha)
                    .constrain("side_top_right_line", "side_bottom_right_line", "Angle", alpha)
                    .constrain("left_top_line", "FixedPoint", 1)
                    .constrain("left_bottom_line", "FixedPoint", 0)
                    .constrain("right_bottom_line", "FixedPoint", 1)
                    .constrain("right_top_line", "FixedPoint", 0)
                    # .constrain("right_top_line", "left_top_line", "Distance", (0, 0, 2 * c))
                    # .constrain("right_bottom_line", "left_bottom_line", "Distance", (0, 0, 2 * c))
                    # .constrain("right_top_line", "right_bottom_line", "Angle", 0)
                    # .constrain("left_top_line", "left_bottom_line", "Angle", 0)
                    # .constrain("left_top_line", "right_top_line", "Angle", 0)
                    .constrain("left_top_line", "Orientation", (1, 0))
                    .constrain("left_bottom_line", "Orientation", (1, 0))
                    .constrain("right_bottom_line", "Orientation", (1, 0))
                    .constrain("right_top_line", "Orientation", (1, 0))
                )

            sketch = sketch.solve().assemble()
            return sketch

        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]
            column = cq.Workplane().cylinder(dimensions["B"], dimensions["F"] / 2).tag("column").translate((0, 0, dimensions["B"] / 2))
            piece = piece + column
            if "H" in dimensions and dimensions["H"] > 0:
                hole = cq.Workplane().cylinder(dimensions["B"], dimensions["H"] / 2).tag("hole").translate((0, 0, dimensions["B"] / 2))
                piece = piece - hole

            piece = piece.translate((0, 0, -dimensions["B"]))
            return piece

    class E(IPiece):
        def get_negative_winding_window(self, dimensions):

            winding_window_cube = (
                cq.Workplane().box(dimensions["E"], dimensions["C"], dimensions["D"]).tag("winding_window_cube").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )

            central_column_cube = (
                cq.Workplane().box(dimensions["F"], dimensions["C"], dimensions["D"]).tag("central_column_cube").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )
            negative_winding_window = winding_window_cube - central_column_cube

            return negative_winding_window

        def get_shape_base(self, data):
            dimensions = data["dimensions"]

            c = dimensions["C"] / 2
            a = dimensions["A"] / 2

            result = (
                cq.Sketch()
                .segment((-a, c), (a, c), "top_line")
                .segment((a, c), (a, -c), "right_line")
                .segment((a, -c), (-a, -c), "bottom_line")
                .segment((-a, -c), (-a, c), "left_line")
                .constrain("top_line", "right_line", "Coincident", None)
                .constrain("right_line", "bottom_line", "Coincident", None)
                .constrain("bottom_line", "left_line", "Coincident", None)
                .constrain("left_line", "top_line", "Coincident", None)
                .constrain("right_line", "Orientation", (0, 1))
                .constrain("left_line", "Orientation", (0, 1))
                .constrain("top_line", "Orientation", (1, 0))
                .constrain("bottom_line", "Orientation", (1, 0))
                .solve()
                .assemble()
            )

            return result

        def apply_machining(self, piece, machining, dimensions):
            length = dimensions["A"]
            if machining["coordinates"][0] == 0:
                width = dimensions["F"]
                length = dimensions["C"]
                y_coordinate = 0
                x_coordinate = 0
                if "K" in dimensions:
                    length = dimensions["C"] - dimensions["K"]
                    x_coordinate += dimensions["K"]
            else:
                width = dimensions["A"] / 2
                if machining["coordinates"][0] < 0:
                    x_coordinate = -dimensions["A"] / 2
                if machining["coordinates"][0] > 0:
                    x_coordinate = dimensions["A"] / 2
                y_coordinate = 0

            height = machining["length"]

            original_tool = cq.Workplane().box(width, length, height).translate((x_coordinate, y_coordinate, machining["coordinates"][1]))

            if machining["coordinates"][0] == 0:
                tool = original_tool
            else:
                # central_column_tool = document.addObject("Part::Box", "central_column_tool")
                central_column_width = dimensions["F"] * 1.001
                central_column_length = dimensions["C"] * 1.001
                if "K" in dimensions:
                    central_column_length = (dimensions["C"] - dimensions["K"] * 2) * 1.001
                length = central_column_length
                width = central_column_width
                height = machining["length"]
                central_column_tool = cq.Workplane().box(width, length, height).translate((0, 0, (machining["coordinates"][1] - machining["length"] / 2)))

                tool = original_tool - central_column_tool

            machined_piece = piece - tool

            return machined_piece

    class Er(E):
        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F", "G"]}

        def get_negative_winding_window(self, dimensions):
            n_seg = CadQueryBuilder.CORE_POLYGON_SEGMENTS
            z_offset = dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])
            winding_window_cylinder = (
                CadQueryBuilder.polygon_cylinder(dimensions['D'], dimensions['E'] / 2, n_seg)
                .tag("winding_window_cylinder")
                .translate((0, 0, z_offset))
            )

            central_column_cylinder = (
                CadQueryBuilder.polygon_cylinder(dimensions['D'], dimensions['F'] / 2, n_seg)
                .tag("central_column_cylinder")
                .translate((0, 0, z_offset))
            )
            winding_window = winding_window_cylinder - central_column_cylinder
            cuts = []

            if "G" in dimensions and dimensions["G"] > dimensions["F"]:
                if dimensions["C"] > dimensions["F"]:
                    length = dimensions["G"]
                    width = dimensions["C"]
                    height = dimensions["D"]
                    translate = (0, 0, height / 2 + dimensions["B"] - dimensions["D"])
                    cube = cq.Workplane().box(length, width, height).tag("cube").translate(translate)

                    cube = cube - central_column_cylinder

                    cuts = [cube]
                else:
                    assert 0

            for cut in cuts:
                winding_window = winding_window + cut
            return winding_window

        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]
            piece = piece.translate((0, 0, -dimensions["B"]))
            return piece

        def apply_machining(self, piece, machining, dimensions):
            """Apply machining (gap) to the core piece.

            For round-center cores (ER, ETD), use a cylinder for central column machining.
            The piece's column axis is along Y (after get_piece's -90° X rotation),
            so all tools must be oriented along Y, not Z.
            """
            height = machining['length']
            y_coord = machining['coordinates'][1]

            if machining['coordinates'][0] == 0:
                # Central column machining - polygon cylinder along Y axis
                radius = dimensions["F"] / 2
                n_seg = CadQueryBuilder.CORE_POLYGON_SEGMENTS
                tool = (
                    CadQueryBuilder.polygon_cylinder(height, radius, n_seg, workplane="XZ")
                    .translate((0, y_coord, 0))
                )
                return piece - tool
            else:
                # Side column machining - rectangular tool along Y axis
                width = dimensions["A"] / 2
                length = dimensions["C"]
                x_coord = -dimensions["A"] / 2 if machining['coordinates'][0] < 0 else dimensions["A"] / 2

                tool = (
                    cq.Workplane()
                    .box(width, height, length)
                    .translate((x_coord, y_coord, 0))
                )

                # Subtract central column area (polygon cylinder along Y)
                central_radius = (dimensions["F"] / 2) * 1.001
                n_seg = CadQueryBuilder.CORE_POLYGON_SEGMENTS
                central_tool = (
                    CadQueryBuilder.polygon_cylinder(height, central_radius, n_seg, workplane="XZ")
                    .translate((0, y_coord, 0))
                )

                tool = tool - central_tool
                return piece - tool

    class El(E):
        def get_dimensions_and_subtypes(self):
            # EL cores may have F2 (oblong column depth) and R (corner radius)
            return {1: ["A", "B", "C", "D", "E", "F", "F2", "R"]}

        def get_negative_winding_window(self, dimensions):
            # Check if column is oblong (F2 defined and different from F)
            f2 = dimensions.get("F2", dimensions["F"])
            has_oblong_column = f2 and abs(f2 - dimensions["F"]) > 0.0001
            
            if has_oblong_column:
                # Stadium-shaped winding window
                # F = column width (X direction, shorter dimension - the diameter of the semicircles)
                # F2 = column depth (Y direction, longer dimension - total stadium length)
                column_height = dimensions["D"]
                z_translate = dimensions["B"] - dimensions["D"]
                
                # Create rectangular winding window cutout
                winding_window_cube = (
                    cq.Workplane()
                    .box(dimensions["E"], dimensions["C"], dimensions["D"])
                    .tag("winding_window_cube")
                    .translate((0, 0, column_height / 2 + z_translate))
                )
                
                # Create stadium-shaped central column
                # Stadium: rectangle with semicircular ends along Y axis
                half_width = dimensions["F"] / 2  # X direction (semicircle radius)
                half_depth = f2 / 2  # Y direction (total half-length)
                semicircle_radius = half_width
                rect_half_length = half_depth - semicircle_radius
                
                if rect_half_length <= 0:
                    # If F2 <= F, it's actually round
                    central_column = (
                        cq.Workplane()
                        .cylinder(column_height, half_width)
                        .translate((0, 0, column_height / 2 + z_translate))
                    )
                else:
                    # Build stadium: center rectangle + two semicircles
                    center_rect = (
                        cq.Workplane()
                        .box(dimensions["F"], rect_half_length * 2, column_height)
                        .translate((0, 0, column_height / 2 + z_translate))
                    )
                    
                    top_semicircle = (
                        cq.Workplane()
                        .cylinder(column_height, semicircle_radius)
                        .translate((0, rect_half_length, column_height / 2 + z_translate))
                    )
                    
                    bottom_semicircle = (
                        cq.Workplane()
                        .cylinder(column_height, semicircle_radius)
                        .translate((0, -rect_half_length, column_height / 2 + z_translate))
                    )
                    
                    central_column = center_rect + top_semicircle + bottom_semicircle
                
                return winding_window_cube - central_column
            else:
                # Rectangular column - use parent's implementation
                return super().get_negative_winding_window(dimensions)

    class Etd(Er):
        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F"]}

    class Lp(Er):
        def get_dimensions_and_subtypes(self):
            return {
                1: ["A", "B", "C", "D", "E", "F", "G"],
            }

        def get_negative_winding_window(self, dimensions):

            winding_window_cylinder = (
                cq.Workplane().cylinder(dimensions["D"], dimensions["E"] / 2).tag("winding_window_cylinder").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )

            central_column_cylinder = (
                cq.Workplane().cylinder(dimensions["D"], dimensions["F"] / 2).tag("central_column_cylinder").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )
            negative_winding_window = winding_window_cylinder - central_column_cylinder

            length = dimensions["G"]
            width = dimensions["C"]
            height = dimensions["D"]
            translate = (0, width / 2 + dimensions["F"] / 2, height / 2 + (dimensions["B"] - dimensions["D"]))
            lateral_top_cube = cq.Workplane().box(length, width, height).tag("lateral_top_cube").translate(translate)
            negative_winding_window = negative_winding_window + lateral_top_cube

            length = dimensions["E"]
            width = dimensions["C"]
            height = dimensions["D"]
            translate = (0, -width / 2, height / 2 + (dimensions["B"] - dimensions["D"]))
            lateral_bottom_cube = cq.Workplane().box(length, width, height).tag("lateral_bottom_cube").translate(translate)
            lateral_bottom_cube = lateral_bottom_cube - central_column_cylinder
            negative_winding_window = negative_winding_window + lateral_bottom_cube
            return negative_winding_window

    class Eq(Er):
        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F", "G"]}

        def get_negative_winding_window(self, dimensions):
            winding_window_cylinder = (
                cq.Workplane().cylinder(dimensions["D"], dimensions["E"] / 2).tag("winding_window_cylinder").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )

            central_column_cylinder = (
                cq.Workplane().cylinder(dimensions["D"], dimensions["F"] / 2).tag("central_column_cylinder").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )
            winding_window = winding_window_cylinder - central_column_cylinder

            return winding_window

    class Ec(Er):
        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F", "T", "s"]}

        def get_shape_base(self, data):
            dimensions = data["dimensions"]

            c = dimensions["C"] / 2
            a = dimensions["A"] / 2
            t = dimensions["T"] / 2
            s = dimensions["s"] / 2

            result = (
                cq.Sketch()
                .segment((-a, c), (a, c), "top_line")
                .segment((a, c), (a, s), "right_line_top")
                .segment((a, s), (t + s, s), "right_dent_top")
                .arc((t + s, s), (t, 0), (t + s, -s), "right_dent_arc")
                .segment((t + s, -s), (a, -s), "right_dent_bottom")
                .segment((a, -s), (a, -c), "right_line_bottom")
                .segment((a, -c), (-a, -c), "bottom_line")
                .segment((-a, -c), (-a, -s), "left_line_bottom")
                .segment((-a, -s), (-(t + s), -s), "left_dent_bottom")
                .arc((-(t + s), -s), (-t, 0), (-(t + s), s), "left_dent_arc")
                .segment((-(t + s), s), (-a, s), "left_dent_bottom")
                .segment((-a, s), (-a, c), "left_line_top")
                .constrain("top_line", "right_line_top", "Coincident", None)
                .constrain("right_line_top", "right_dent_top", "Coincident", None)
                .constrain("right_dent_top", "right_dent_arc", "Coincident", None)
                .constrain("right_dent_arc", "right_dent_bottom", "Coincident", None)
                .constrain("right_dent_bottom", "right_line_bottom", "Coincident", None)
                .constrain("right_line_bottom", "bottom_line", "Coincident", None)
                .constrain("bottom_line", "left_line_bottom", "Coincident", None)
                .constrain("left_line_bottom", "left_dent_bottom", "Coincident", None)
                .constrain("left_dent_bottom", "left_dent_arc", "Coincident", None)
                .constrain("left_dent_arc", "left_dent_bottom", "Coincident", None)
                .constrain("left_dent_arc", "left_dent_bottom", "Coincident", None)
                .constrain("left_dent_bottom", "left_line_top", "Coincident", None)
                .constrain("left_line_top", "top_line", "Coincident", None)
                .constrain("left_line_bottom", "Orientation", (0, 1))
                .constrain("left_line_top", "Orientation", (0, 1))
                .constrain("right_line_top", "Orientation", (0, 1))
                .constrain("right_line_bottom", "Orientation", (0, 1))
                .constrain("top_line", "Orientation", (1, 0))
                .constrain("bottom_line", "Orientation", (1, 0))
                .solve()
                .assemble()
            )

            return result

    class Ep(E):
        def get_shape_base(self, data):
            dimensions = data["dimensions"]

            a = dimensions["A"] / 2

            top_c = dimensions["C"] - dimensions["K"]
            bottom_c = dimensions["K"]

            sketch = (
                cq.Sketch()
                .segment((-a, top_c), (a, top_c), "top_line")
                .segment((a, top_c), (a, -bottom_c), "right_line")
                .segment((a, -bottom_c), (-a, -bottom_c), "bottom_line")
                .segment((-a, -bottom_c), (-a, top_c), "left_line")
                .constrain("top_line", "right_line", "Coincident", None)
                .constrain("right_line", "bottom_line", "Coincident", None)
                .constrain("bottom_line", "left_line", "Coincident", None)
                .constrain("left_line", "top_line", "Coincident", None)
                .constrain("right_line", "Orientation", (0, 1))
                .constrain("left_line", "Orientation", (0, 1))
                .constrain("top_line", "Orientation", (1, 0))
                .constrain("bottom_line", "Orientation", (1, 0))
                .solve()
                .assemble()
            )

            return sketch

        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F", "G", "K"]}

        def get_negative_winding_window(self, dimensions):

            winding_window_cylinder = (
                cq.Workplane().cylinder(dimensions["D"], dimensions["E"] / 2).tag("winding_window_cylinder").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )

            central_column_cylinder = (
                cq.Workplane().cylinder(dimensions["D"], dimensions["F"] / 2).tag("central_column_cylinder").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )
            negative_winding_window = winding_window_cylinder - central_column_cylinder

            if "G" in dimensions and dimensions["G"] > 0:
                length = dimensions["G"]
                width = dimensions["C"]
                height = dimensions["D"]
                translate = (0, width / 2 + dimensions["F"] / 2, height / 2 + (dimensions["B"] - dimensions["D"]))
                top_cube = cq.Workplane().box(length, width, height).tag("top_cube").translate(translate)
                negative_winding_window = negative_winding_window + top_cube

            length = dimensions["E"]
            width = dimensions["C"]
            height = dimensions["D"]
            translate = (0, -width / 2, height / 2 + (dimensions["B"] - dimensions["D"]))
            bottom_cube = cq.Workplane().box(length, width, height).tag("bottom_cube").translate(translate)
            bottom_cube = bottom_cube - central_column_cylinder
            negative_winding_window = negative_winding_window + bottom_cube
            return negative_winding_window

        def apply_machining(self, piece, machining, dimensions):
            if machining["coordinates"][0] == 0 and machining["coordinates"][2] == 0:
                # Gap in central column
                width = dimensions["F"]
                length = dimensions["F"]
                x_coordinate = 0
                y_coordinate = 0
            elif machining["coordinates"][0] != 0 and machining["coordinates"][2] == 0:
                # Gap in lateral column because they are not connected
                width = dimensions["A"] / 2
                length = dimensions["C"] * 2
                y_coordinate = 0
                if machining["coordinates"][0] < 0:
                    x_coordinate = -width / 2
                if machining["coordinates"][0] > 0:
                    x_coordinate = width / 2
            else:
                # Gap in lateral column but they are connected
                length = dimensions["C"] * 2
                width = dimensions["A"]
                x_coordinate = 0
                y_coordinate = 0

            height = machining["length"]

            original_tool = cq.Workplane().box(width, length, height).translate((x_coordinate, y_coordinate, machining["coordinates"][1]))

            if machining["coordinates"][0] == 0 and machining["coordinates"][2] == 0:
                tool = original_tool
            else:
                central_column_tool = cq.Workplane().cylinder(dimensions["D"] * 2, dimensions["F"] / 2 * 1.2).translate((0, 0, (machining["coordinates"][1] - machining["length"] / 2)))

                tool = original_tool - central_column_tool

            machined_piece = piece - tool

            return machined_piece

    class Epx(E):
        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F", "G", "K"]}

        def get_shape_base(self, data):
            dimensions = data["dimensions"]

            a = dimensions["A"] / 2

            column_length = dimensions["K"] + dimensions["F"] / 2
            top_c = dimensions["C"] - column_length / 2
            bottom_c = column_length / 2

            sketch = (
                cq.Sketch()
                .segment((-a, top_c), (a, top_c), "top_line")
                .segment((a, top_c), (a, -bottom_c), "right_line")
                .segment((a, -bottom_c), (-a, -bottom_c), "bottom_line")
                .segment((-a, -bottom_c), (-a, top_c), "left_line")
                .constrain("top_line", "right_line", "Coincident", None)
                .constrain("right_line", "bottom_line", "Coincident", None)
                .constrain("bottom_line", "left_line", "Coincident", None)
                .constrain("left_line", "top_line", "Coincident", None)
                .constrain("right_line", "Orientation", (0, 1))
                .constrain("left_line", "Orientation", (0, 1))
                .constrain("top_line", "Orientation", (1, 0))
                .constrain("bottom_line", "Orientation", (1, 0))
                .solve()
                .assemble()
            )

            return sketch

        def get_negative_winding_window(self, dimensions):
            rectangular_part_width = dimensions["K"] - dimensions["F"] / 2

            winding_window_cylinder = (
                cq.Workplane()
                .cylinder(dimensions["D"], dimensions["E"] / 2)
                .tag("winding_window_cylinder")
                .translate((0, rectangular_part_width / 2, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )

            column_width = dimensions["K"] + dimensions["F"] / 2
            length = dimensions["F"]
            height = dimensions["D"]
            translate = (0, 0, height / 2 + (dimensions["B"] - dimensions["D"]))
            central_column_center = cq.Workplane().box(length, rectangular_part_width, height).tag("central_column_center").translate(translate)
            central_column_top_cylinder = (
                cq.Workplane()
                .cylinder(dimensions["D"], dimensions["F"] / 2)
                .tag("central_column_top_cylinder")
                .translate((0, rectangular_part_width / 2, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )
            central_column_bottom_cylinder = (
                cq.Workplane()
                .cylinder(dimensions["D"], dimensions["F"] / 2)
                .tag("central_column_bottom_cylinder")
                .translate((0, -rectangular_part_width / 2, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )
            central_column = central_column_center + central_column_top_cylinder + central_column_bottom_cylinder

            negative_winding_window = winding_window_cylinder - central_column

            if "G" in dimensions and dimensions["G"] > 0:
                length = dimensions["G"]
                width = dimensions["C"]
                height = dimensions["D"]
                translate = (0, width / 2 + column_width / 2, height / 2 + (dimensions["B"] - dimensions["D"]))
                top_cube = cq.Workplane().box(length, width, height).tag("top_cube").translate(translate)
                negative_winding_window = negative_winding_window + top_cube

            length = dimensions["E"]
            width = dimensions["C"]
            height = dimensions["D"]
            translate = (0, -width / 2 + rectangular_part_width / 2, height / 2 + (dimensions["B"] - dimensions["D"]))
            bottom_cube = cq.Workplane().box(length, width, height).tag("bottom_cube").translate(translate)
            bottom_cube = bottom_cube - central_column
            negative_winding_window = negative_winding_window + bottom_cube
            return negative_winding_window

        def apply_machining(self, piece, machining, dimensions):
            if machining["coordinates"][0] == 0 and machining["coordinates"][2] == 0:
                # Gap in central column
                width = dimensions["F"]
                length = dimensions["K"] + dimensions["F"] / 2
                x_coordinate = 0
                y_coordinate = 0
            elif machining["coordinates"][0] != 0 and machining["coordinates"][2] == 0:
                # Gap in lateral column because they are not connected
                width = dimensions["A"] / 2
                length = dimensions["C"] * 2
                y_coordinate = 0
                if machining["coordinates"][0] < 0:
                    x_coordinate = -width / 2
                if machining["coordinates"][0] > 0:
                    x_coordinate = width / 2
            else:
                # Gap in lateral column but they are connected
                length = dimensions["C"] * 2
                width = dimensions["A"]
                x_coordinate = 0
                y_coordinate = 0

            height = machining["length"]

            original_tool = cq.Workplane().box(width, length, height).translate((x_coordinate, y_coordinate, machining["coordinates"][1]))

            if machining["coordinates"][0] == 0 and machining["coordinates"][2] == 0:
                tool = original_tool
            else:
                rectangular_part_width = dimensions["K"] - dimensions["F"] / 2

                length = dimensions["F"]
                height = dimensions["D"] * 2
                translate = (0, 0, 0)
                central_column_center = cq.Workplane().box(length, rectangular_part_width, height).tag("central_column_center").translate(translate)
                central_column_top_cylinder = cq.Workplane().cylinder(dimensions["D"], dimensions["F"] / 2).tag("central_column_top_cylinder").translate((0, rectangular_part_width / 2, 0))
                central_column_bottom_cylinder = cq.Workplane().cylinder(dimensions["D"], dimensions["F"] / 2).tag("central_column_bottom_cylinder").translate((0, -rectangular_part_width / 2, 0))
                central_column_tool = central_column_center + central_column_top_cylinder + central_column_bottom_cylinder
                tool = original_tool - central_column_tool

            machined_piece = piece - tool

            return machined_piece

    class Efd(E):
        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F", "F2", "K", "q"], 2: ["A", "B", "C", "D", "E", "F", "F2", "K", "q"]}

        def get_shape_base(self, data):
            dimensions = data["dimensions"]

            a = dimensions["A"] / 2

            top_c = dimensions["C"] - dimensions["K"] - dimensions["F2"] / 2
            bottom_c = dimensions["K"] + dimensions["F2"] / 2
            dent_height = dimensions["C"] * 2 / 5
            dent_top_width = dimensions["F"] / 2
            dent_bottom_width = dimensions["F"] / 2 - dimensions["q"]

            if dimensions["K"] > 0:
                minident_semiwidth = dimensions["F"] / 2 - dimensions["q"]
                minident_depth = dimensions["K"]
                sketch = (
                    cq.Sketch()
                    .segment((-a, top_c), (-dent_top_width, top_c), "top_line_left")
                    .segment((-dent_top_width, top_c), (-dent_bottom_width, top_c - dent_height), "dent_line_left")
                    .segment((-dent_bottom_width, top_c - dent_height), (dent_bottom_width, top_c - dent_height), "dent_line_bottom")
                    .segment((dent_bottom_width, top_c - dent_height), (dent_top_width, top_c), "dent_line_right")
                    .segment((dent_top_width, top_c), (a, top_c), "top_line_right")
                    .segment((a, top_c), (a, -bottom_c), "right_line")
                    .segment((a, -bottom_c), (minident_semiwidth, -bottom_c), "bottom_line_left")
                    .segment((minident_semiwidth, -bottom_c), (minident_semiwidth, -bottom_c + minident_depth), "minident_left_side")
                    .segment((minident_semiwidth, -bottom_c + minident_depth), (-minident_semiwidth, -bottom_c + minident_depth), "minident_bottom")
                    .segment((-minident_semiwidth, -bottom_c + minident_depth), (-minident_semiwidth, -bottom_c), "minident_right_side")
                    .segment((-minident_semiwidth, -bottom_c), (-a, -bottom_c), "bottom_line_right")
                    .segment((-a, -bottom_c), (-a, top_c), "left_line")
                    .constrain("top_line_left", "dent_line_left", "Coincident", None)
                    .constrain("dent_line_left", "dent_line_bottom", "Coincident", None)
                    .constrain("dent_line_bottom", "dent_line_right", "Coincident", None)
                    .constrain("dent_line_right", "top_line_right", "Coincident", None)
                    .constrain("top_line_right", "right_line", "Coincident", None)
                    .constrain("right_line", "bottom_line_left", "Coincident", None)
                    .constrain("bottom_line_left", "minident_left_side", "Coincident", None)
                    .constrain("minident_left_side", "minident_bottom", "Coincident", None)
                    .constrain("minident_bottom", "minident_right_side", "Coincident", None)
                    .constrain("minident_right_side", "bottom_line_right", "Coincident", None)
                    .constrain("bottom_line_right", "left_line", "Coincident", None)
                    .constrain("left_line", "top_line_left", "Coincident", None)
                    .constrain("right_line", "Orientation", (0, 1))
                    .constrain("left_line", "Orientation", (0, 1))
                    .constrain("top_line_left", "Orientation", (1, 0))
                    .constrain("top_line_right", "Orientation", (1, 0))
                    .constrain("bottom_line_left", "Orientation", (1, 0))
                    .constrain("minident_bottom", "Orientation", (1, 0))
                    .constrain("bottom_line_right", "Orientation", (1, 0))
                    .constrain("minident_left_side", "Orientation", (0, 1))
                    .constrain("minident_right_side", "Orientation", (0, 1))
                    .constrain("dent_line_bottom", "Orientation", (1, 0))
                    .solve()
                    .assemble()
                )
            else:
                sketch = (
                    cq.Sketch()
                    .segment((-a, top_c), (-dent_top_width, top_c), "top_line_left")
                    .segment((-dent_top_width, top_c), (-dent_bottom_width, top_c - dent_height), "dent_line_left")
                    .segment((-dent_bottom_width, top_c - dent_height), (dent_bottom_width, top_c - dent_height), "dent_line_bottom")
                    .segment((dent_bottom_width, top_c - dent_height), (dent_top_width, top_c), "dent_line_right")
                    .segment((dent_top_width, top_c), (a, top_c), "top_line_right")
                    .segment((a, top_c), (a, -bottom_c), "right_line")
                    .segment((a, -bottom_c), (-a, -bottom_c), "bottom_line")
                    .segment((-a, -bottom_c), (-a, top_c), "left_line")
                    .constrain("top_line_left", "dent_line_left", "Coincident", None)
                    .constrain("dent_line_left", "dent_line_bottom", "Coincident", None)
                    .constrain("dent_line_bottom", "dent_line_right", "Coincident", None)
                    .constrain("dent_line_right", "top_line_right", "Coincident", None)
                    .constrain("top_line_right", "right_line", "Coincident", None)
                    .constrain("right_line", "bottom_line", "Coincident", None)
                    .constrain("bottom_line", "left_line", "Coincident", None)
                    .constrain("left_line", "top_line_left", "Coincident", None)
                    .constrain("right_line", "Orientation", (0, 1))
                    .constrain("left_line", "Orientation", (0, 1))
                    .constrain("top_line_left", "Orientation", (1, 0))
                    .constrain("top_line_right", "Orientation", (1, 0))
                    .constrain("bottom_line", "Orientation", (1, 0))
                    .constrain("dent_line_bottom", "Orientation", (1, 0))
                    .solve()
                    .assemble()
                )

            return sketch

        def get_negative_winding_window(self, dimensions):

            winding_window_cube = (
                cq.Workplane().box(dimensions["E"], dimensions["C"] * 2, dimensions["D"]).tag("winding_window_cube").translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )
            return winding_window_cube

        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]

            column = cq.Workplane().sketch().rect(dimensions["F"], dimensions["F2"]).vertices().chamfer(dimensions["q"]).finalize().extrude(dimensions["B"]).translate((0, 0, 0))
            piece = piece + column
            piece = piece.translate((0, 0, -dimensions["B"]))
            return piece

        def apply_machining(self, piece, machining, dimensions):
            length = dimensions["A"]
            if machining["coordinates"][0] == 0:
                width = dimensions["F"]
                length = dimensions["F2"]
                y_coordinate = 0
                x_coordinate = 0
            else:
                width = dimensions["A"] / 2
                if machining["coordinates"][0] < 0:
                    x_coordinate = -dimensions["A"] / 2
                if machining["coordinates"][0] > 0:
                    x_coordinate = dimensions["A"] / 2
                y_coordinate = 0

            height = machining["length"]

            original_tool = cq.Workplane().box(width, length, height).translate((x_coordinate, y_coordinate, machining["coordinates"][1]))

            if machining["coordinates"][0] == 0:
                tool = original_tool
            else:
                central_column_width = dimensions["F"] * 1.001
                central_column_length = dimensions["F2"] * 1.001

                length = central_column_length
                width = central_column_width
                height = dimensions["D"] * 2
                central_column_tool = cq.Workplane().box(width, length, height).translate((0, 0, 0))

                tool = original_tool - central_column_tool

            machined_piece = piece - tool

            return machined_piece

    class U(IPiece):
        def get_shape_base(self, data):
            dimensions = data["dimensions"]

            c = dimensions["C"] / 2
            winding_column_width = (dimensions["A"] - dimensions["E"]) / 2
            left_a = dimensions["A"] - winding_column_width / 2
            right_a = winding_column_width / 2

            result = (
                cq.Sketch()
                .segment((right_a, c), (-left_a, c), "top_line")
                .segment((-left_a, c), (-left_a, -c), "left_line")
                .segment((-left_a, -c), (right_a, -c), "bottom_line")
                .segment((right_a, -c), (right_a, c), "right_line")
                .constrain("top_line", "left_line", "Coincident", None)
                .constrain("left_line", "bottom_line", "Coincident", None)
                .constrain("bottom_line", "right_line", "Coincident", None)
                .constrain("right_line", "top_line", "Coincident", None)
                .constrain("right_line", "Orientation", (0, 1))
                .constrain("left_line", "Orientation", (0, 1))
                .constrain("top_line", "Orientation", (1, 0))
                .constrain("bottom_line", "Orientation", (1, 0))
                .solve()
                .assemble()
            )

            return result

        def get_dimensions_and_subtypes(self):
            return shape_configs.U_DIMENSIONS_AND_SUBTYPES

        def get_negative_winding_window(self, dimensions):
            winding_column_width = (dimensions["A"] - dimensions["E"]) / 2
            negative_winding_window = (
                cq.Workplane()
                .box(dimensions["E"], dimensions["C"] * 2, dimensions["D"])
                .tag("negative_winding_window")
                .translate((-(winding_column_width / 2 + dimensions["E"] / 2), 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )
            return negative_winding_window

        def apply_machining(self, piece, machining, dimensions):
            winding_column_width = (dimensions["A"] - dimensions["E"]) / 2
            translate = convert_axis(machining["coordinates"])
            gap = cq.Workplane().box(winding_column_width, dimensions["C"], machining["length"]).tag("gap").translate(translate)

            machined_piece = piece - gap

            return machined_piece

    class Ur(IPiece):
        def get_dimensions_and_subtypes(self):
            return shape_configs.UR_DIMENSIONS_AND_SUBTYPES

        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]
            familySubtype = data["familySubtype"]
            if familySubtype == "1":
                winding_column_width = dimensions["C"]
                translate = (0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"]))
                winding_column = cq.Workplane().cylinder(dimensions["D"], dimensions["C"] / 2).tag("winding_column").translate(translate)
                translate = (-(dimensions["A"] - winding_column_width / 2 - dimensions["H"] / 2), 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"]))
                lateral_column = cq.Workplane().box(dimensions["H"], dimensions["C"], dimensions["D"]).tag("lateral_column").translate(translate)
                piece += winding_column + lateral_column
            elif familySubtype == "2":
                winding_column_width = dimensions["C"]
                translate = (0, 0, dimensions["B"] / 2)
                winding_column = cq.Workplane().cylinder(dimensions["B"], dimensions["C"] / 2).tag("winding_column").translate(translate)
                translate = (-(dimensions["A"] - winding_column_width), 0, dimensions["B"] / 2)
                lateral_column = cq.Workplane().cylinder(dimensions["B"], dimensions["C"] / 2).tag("lateral_column").translate(translate)
                piece += winding_column + lateral_column
            elif familySubtype == "3":
                winding_column_width = dimensions["F"]
                translate = (0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"]))
                winding_column = cq.Workplane().cylinder(dimensions["D"], dimensions["F"] / 2).tag("winding_column").translate(translate)
                translate = (-(dimensions["A"] - winding_column_width / 2 - dimensions["H"] / 2), 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"]))
                lateral_column = cq.Workplane().box(dimensions["H"], dimensions["C"], dimensions["D"]).tag("lateral_column").translate(translate)
                piece += winding_column + lateral_column
            elif familySubtype == "4":
                winding_column_width = dimensions["C"]
                translate = (0, 0, dimensions["B"] / 2)
                winding_column = cq.Workplane().cylinder(dimensions["B"], dimensions["F"] / 2).tag("winding_column").translate(translate)
                translate = (-(dimensions["A"] - dimensions["F"] / 2 - dimensions["F"] / 2), 0, dimensions["B"] / 2)
                lateral_column = cq.Workplane().cylinder(dimensions["B"], dimensions["F"] / 2).tag("lateral_column").translate(translate)
                piece += winding_column + lateral_column

            if "S" in dimensions:
                if "F" in dimensions:
                    winding_column_width = dimensions["F"]
                else:
                    winding_column_width = dimensions["C"]

                if "H" in dimensions:
                    lateral_column_width = dimensions["H"]
                else:
                    lateral_column_width = dimensions["F"]

                translate = (-(dimensions["A"] - lateral_column_width / 2 - dimensions["S"] / 2), 0, dimensions["B"] / 2)
                lateral_hole_round_left = cq.Workplane().cylinder(dimensions["B"], dimensions["S"] / 2).tag("lateral_hole_round_left").translate(translate)

                translate = (-(dimensions["A"] - lateral_column_width / 2 - dimensions["S"] / 4), 0, dimensions["B"] / 2)
                lateral_hole_rectangular_left = cq.Workplane().box(dimensions["S"] / 2, dimensions["S"], dimensions["B"]).tag("lateral_hole_rectangular_left").translate(translate)
                piece -= lateral_hole_round_left + lateral_hole_rectangular_left

                translate = (winding_column_width / 2 - dimensions["S"] / 2, 0, dimensions["B"] / 2)
                lateral_hole_round_right = cq.Workplane().cylinder(dimensions["B"], dimensions["S"] / 2).tag("lateral_hole_round_right").translate(translate)

                # Extend rectangular hole slightly past column surface to avoid
                # tangent-surface boolean failures in OCC kernel
                eps = 1e-5
                translate = (winding_column_width / 2 - dimensions["S"] / 4 + eps / 2, 0, dimensions["B"] / 2)
                lateral_hole_rectangular_right = cq.Workplane().box(dimensions["S"] / 2 + eps, dimensions["S"], dimensions["B"]).tag("lateral_hole_rectangular_right").translate(translate)
                piece -= lateral_hole_round_right + lateral_hole_rectangular_right

            piece = piece.translate((0, 0, -dimensions["B"]))
            return piece

        def get_shape_base(self, data):
            dimensions = data["dimensions"]
            familySubtype = data["familySubtype"]
            if familySubtype == "1":
                c = dimensions["C"] / 2
                winding_column_width = dimensions["C"]
                left_a = dimensions["A"] - winding_column_width / 2
                right_a = winding_column_width / 2

                result = (
                    cq.Sketch()
                    .segment((0, c), (-left_a, c), "top_line")
                    .segment((-left_a, c), (-left_a, -c), "left_line")
                    .segment((-left_a, -c), (0, -c), "bottom_line")
                    .arc((0, -c), (right_a, 0), (0, c), "right_arc")
                    .constrain("top_line", "left_line", "Coincident", None)
                    .constrain("left_line", "bottom_line", "Coincident", None)
                    .constrain("bottom_line", "right_arc", "Coincident", None)
                    .constrain("right_arc", "top_line", "Coincident", None)
                    .constrain("left_line", "Orientation", (0, 1))
                    .constrain("top_line", "Orientation", (1, 0))
                    .constrain("bottom_line", "Orientation", (1, 0))
                    .solve()
                    .assemble()
                )
            elif familySubtype == "2" or familySubtype == "4":
                c = dimensions["C"] / 2
                if familySubtype == "4":
                    winding_column_width = dimensions["F"]
                else:
                    winding_column_width = dimensions["C"]
                left_a = dimensions["A"] - winding_column_width

                result = (
                    cq.Sketch()
                    .segment((0, c), (-left_a, c), "top_line")
                    .segment((-left_a, c), (-left_a, -c), "left_line")
                    .segment((-left_a, -c), (0, -c), "bottom_line")
                    .segment((0, c), (0, -c), "right_line")
                    .constrain("top_line", "left_line", "Coincident", None)
                    .constrain("left_line", "bottom_line", "Coincident", None)
                    .constrain("bottom_line", "right_line", "Coincident", None)
                    .constrain("right_line", "top_line", "Coincident", None)
                    .constrain("right_line", "Orientation", (0, 1))
                    .constrain("left_line", "Orientation", (0, 1))
                    .constrain("top_line", "Orientation", (1, 0))
                    .constrain("bottom_line", "Orientation", (1, 0))
                    .solve()
                    .assemble()
                )
            elif familySubtype == "3":
                c = dimensions["C"] / 2
                winding_column_width = dimensions["F"]
                left_a = dimensions["A"] - winding_column_width / 2
                right_a = winding_column_width / 2

                result = (
                    cq.Sketch()
                    .segment((0, c), (-left_a, c), "top_line")
                    .segment((-left_a, c), (-left_a, -c), "left_line")
                    .segment((-left_a, -c), (0, -c), "bottom_line")
                    .arc((0, -c), (right_a, 0), (0, c), "right_arc")
                    .constrain("top_line", "left_line", "Coincident", None)
                    .constrain("left_line", "bottom_line", "Coincident", None)
                    .constrain("bottom_line", "right_arc", "Coincident", None)
                    .constrain("right_arc", "top_line", "Coincident", None)
                    .constrain("left_line", "Orientation", (0, 1))
                    .constrain("top_line", "Orientation", (1, 0))
                    .constrain("bottom_line", "Orientation", (1, 0))
                    .solve()
                    .assemble()
                )

            return result

        def get_negative_winding_window(self, dimensions):
            negative_winding_window = (
                cq.Workplane()
                .box(dimensions["A"] * 2, dimensions["C"] * 2, dimensions["D"])
                .tag("negative_winding_window")
                .translate((0, 0, dimensions["D"] / 2 + (dimensions["B"] - dimensions["D"])))
            )
            return negative_winding_window

        def apply_machining(self, piece, machining, dimensions):
            winding_column_width = max([dimensions["C"], dimensions["H"]])
            translate = convert_axis(machining["coordinates"])
            gap = cq.Workplane().box(winding_column_width, dimensions["C"] * 2, machining["length"]).tag("gap").translate(translate)

            machined_piece = piece - gap

            return machined_piece

    class T(IPiece):
        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C"]}

        def get_negative_winding_window(self, dimensions):
            return None

        def get_shape_base(self, data):
            dimensions = data["dimensions"]

            b = dimensions["B"] / 2
            a = dimensions["A"] / 2

            result = cq.Sketch().circle(a).circle(b, mode="s")

            return result

        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]
            c = dimensions["C"] / 2
            piece = piece.translate((0, 0, -c))
            # Rotate to match MKF's expected pre-rotation orientation
            piece = piece.rotate((0, 1, 0), (0, -1, 0), 90)
            return piece

    class Ut(IPiece):
        def get_dimensions_and_subtypes(self):
            return {1: ["A", "B", "C", "D", "E", "F"]}

        def get_shape_base(self, data):
            dimensions = data["dimensions"]
            a = dimensions["A"] / 2
            c = dimensions["C"] / 2

            result = (
                cq.Sketch()
                .segment((-a, c), (a, c), "top_line")
                .segment((a, c), (a, -c), "right_line")
                .segment((a, -c), (-a, -c), "bottom_line")
                .segment((-a, -c), (-a, c), "left_line")
                .constrain("top_line", "right_line", "Coincident", None)
                .constrain("right_line", "bottom_line", "Coincident", None)
                .constrain("bottom_line", "left_line", "Coincident", None)
                .constrain("left_line", "top_line", "Coincident", None)
                .constrain("right_line", "Orientation", (0, 1))
                .constrain("left_line", "Orientation", (0, 1))
                .constrain("top_line", "Orientation", (1, 0))
                .constrain("bottom_line", "Orientation", (1, 0))
                .solve()
                .assemble()
            )

            return result

        def get_negative_winding_window(self, dimensions):
            negative_winding_window = cq.Workplane().box(dimensions["A"] * 2, dimensions["C"] * 2, dimensions["D"]).tag("negative_winding_window").translate((0, 0, dimensions["B"] / 2))
            return negative_winding_window

        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]

            top_column = cq.Workplane().box(dimensions["F"], dimensions["C"], dimensions["D"]).tag("top_column").translate((-dimensions["A"] / 2 + dimensions["F"] / 2, 0, dimensions["B"] / 2))

            bottom_column_width = dimensions["A"] - dimensions["E"] - dimensions["F"]
            bottom_column = (
                cq.Workplane().box(bottom_column_width, dimensions["C"], dimensions["D"]).tag("bottom_column").translate((dimensions["A"] / 2 - bottom_column_width / 2, 0, dimensions["B"] / 2))
            )

            piece = piece + top_column + bottom_column
            piece = piece.translate((0, 0, -dimensions["B"]))
            return piece

    class C(U):
        def get_shape_extras(self, data, piece):
            dimensions = data["dimensions"]
            fillet_radius = (dimensions["A"] - dimensions["E"]) / 2

            piece = piece.translate((0, 0, -(dimensions["B"] - dimensions["D"]) / 2))
            # piece = piece.edges("|Y").edges("<Z").all().fillet(fillet_radius)
            piece = piece.edges("|Y").edges("<Z").fillet(fillet_radius)
            piece = piece.translate((0, 0, (dimensions["B"] - dimensions["D"]) / 2))

            piece = piece.translate((0, 0, -dimensions["B"]))
            return piece

    class IBobbin(metaclass=ABCMeta):
        def __init__(self):
            self.output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"

        def set_output_path(self, output_path):
            self.output_path = output_path

        @abstractmethod
        def get_bobbin_body(self, data, winding_window):
            raise NotImplementedError

        @abstractmethod
        def get_bobbin_flanges(self, data, winding_window):
            raise NotImplementedError

        @abstractmethod
        def get_mounting_pins(self, data, outer_radius):
            raise NotImplementedError

        def get_bobbin(self, data, winding_window, name="Bobbin", save_files=False, export_files=True):
            try:
                project_name = f"{name}".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                dims = data.get("dimensions", {})

                body = self.get_bobbin_body(data, winding_window)
                flanges = self.get_bobbin_flanges(data, winding_window)

                bobbin = body
                if flanges is not None:
                    bobbin = bobbin + flanges

                if dims.get("pinCount", 0) > 0:
                    ww_width = winding_window.get("width", 0)
                    ww_height = winding_window.get("height", 0)
                    flange_extension = dims.get("flangeExtension", 0.002)
                    if ww_width > ww_height:
                        outer_radius = ww_width / 2 + dims.get("wallThickness", 0.0005) + flange_extension
                    else:
                        outer_radius = ww_height / 2 + dims.get("wallThickness", 0.0005) + flange_extension
                    pins = self.get_mounting_pins(data, outer_radius)
                    if pins is not None:
                        bobbin = bobbin + pins

                coords = data.get("coordinates", [0, 0, 0])
                rotation = data.get("rotation", [0, 0, 0])

                if rotation[0] != 0:
                    bobbin = bobbin.rotate((1, 0, 0), (-1, 0, 0), rotation[0] / math.pi * 180)
                if rotation[1] != 0:
                    bobbin = bobbin.rotate((0, 1, 0), (0, -1, 0), rotation[1] / math.pi * 180)
                if rotation[2] != 0:
                    bobbin = bobbin.rotate((0, 0, 1), (0, 0, -1), rotation[2] / math.pi * 180)

                bobbin = bobbin.translate(convert_axis(coords))

                pathlib.Path(self.output_path).mkdir(parents=True, exist_ok=True)

                if export_files:
                    from cadquery import exporters

                    scaled_bobbin = bobbin.newObject([o.scale(1000) for o in bobbin.objects])
                    exporters.export(scaled_bobbin, f"{self.output_path}/{project_name}.step", "STEP")
                    exporters.export(scaled_bobbin, f"{self.output_path}/{project_name}.stl", "STL")
                    return f"{self.output_path}/{project_name}.step", f"{self.output_path}/{project_name}.stl"
                else:
                    return bobbin

            except Exception:
                return (None, None) if export_files else None

    class StandardBobbin(IBobbin):
        def get_bobbin_body(self, data, winding_window):
            dims = data.get("dimensions", {})
            processed = data.get("processedDescription", {})

            if processed:
                wall_thickness = processed.get("wallThickness", dims.get("wallThickness", 0.0005))
                column_shape = processed.get("columnShape", "rectangular")
                column_width = processed.get("columnWidth", 0)
                column_thickness = processed.get("columnThickness", wall_thickness)
                bobbin_ww = processed.get("windingWindows", [{}])[0] if processed.get("windingWindows") else {}
                ww_width = bobbin_ww.get("width", winding_window.get("width", 0))
                ww_height = bobbin_ww.get("height", winding_window.get("height", 0))
            else:
                wall_thickness = dims.get("wallThickness", 0.0005)
                ww_width = winding_window.get("width", 0)
                ww_height = winding_window.get("height", 0)
                column_shape = winding_window.get("columnShape", "rectangular")
                column_width = winding_window.get("columnWidth", 0)
                column_thickness = wall_thickness

            tube_height = ww_height

            if column_shape == "round":
                if column_width > 0:
                    outer_radius = column_width
                    hole_radius = column_width - column_thickness
                    if hole_radius <= 0:
                        hole_radius = outer_radius - wall_thickness
                else:
                    ww_coords = winding_window.get("coordinates", [0, 0])
                    outer_radius = abs(ww_coords[0]) if ww_coords[0] != 0 else ww_width * 0.5
                    hole_radius = outer_radius - wall_thickness

                outer_cyl = cq.Workplane("XY").cylinder(tube_height, outer_radius)
                inner_cyl = cq.Workplane("XY").cylinder(tube_height * 1.1, hole_radius)
                body = outer_cyl - inner_cyl
            else:
                depth = winding_window.get("radialHeight", ww_width) if winding_window.get("radialHeight") else ww_width
                outer_width = ww_width + wall_thickness * 2
                outer_depth = depth + wall_thickness * 2

                outer_box = cq.Workplane("XY").box(outer_width, outer_depth, tube_height)
                inner_box = cq.Workplane("XY").box(ww_width, depth, tube_height * 1.1)

                central_hole_width = depth * 0.8
                central_hole_depth = depth * 0.8
                central_hole = cq.Workplane("XY").box(central_hole_width, central_hole_depth, tube_height * 1.2)

                body = outer_box - inner_box - central_hole

            return body

        def get_bobbin_flanges(self, data, winding_window):
            dims = data.get("dimensions", {})
            flange_thickness = dims.get("flangeThickness", 0.001)
            flange_extension = dims.get("flangeExtension", 0.002)
            processed = data.get("processedDescription", {})

            if processed:
                wall_thickness = processed.get("wallThickness", dims.get("wallThickness", 0.0005))
                column_shape = processed.get("columnShape", "rectangular")
                column_width = processed.get("columnWidth", 0)
                column_depth = processed.get("columnDepth", column_width)
                column_thickness = processed.get("columnThickness", wall_thickness)
                bobbin_ww = processed.get("windingWindows", [{}])[0] if processed.get("windingWindows") else {}
                ww_width = bobbin_ww.get("width", winding_window.get("width", 0))
                ww_height = bobbin_ww.get("height", winding_window.get("height", 0))
            else:
                wall_thickness = dims.get("wallThickness", 0.0005)
                ww_width = winding_window.get("width", 0)
                ww_height = winding_window.get("height", 0)
                column_shape = winding_window.get("columnShape", "rectangular")
                column_width = winding_window.get("columnWidth", 0)
                column_depth = winding_window.get("columnDepth", column_width)
                column_thickness = wall_thickness

            if column_shape == "round":
                if column_width > 0:
                    outer_radius = column_width
                    hole_radius = column_width - column_thickness
                    if hole_radius <= 0:
                        hole_radius = outer_radius - wall_thickness
                else:
                    ww_coords = winding_window.get("coordinates", [0, 0])
                    outer_radius = abs(ww_coords[0]) if ww_coords[0] != 0 else ww_width * 0.5
                    hole_radius = outer_radius - wall_thickness

                flange_outer_x = outer_radius + ww_width + flange_extension
                flange_half_y = column_depth / 2 if column_depth > 0 else outer_radius

                top_flange_solid = cq.Workplane("XY").box(flange_outer_x * 2, flange_half_y * 2, flange_thickness).translate((0, 0, ww_height / 2 + flange_thickness / 2))
                top_hole = cq.Workplane("XY").cylinder(flange_thickness * 1.1, hole_radius).translate((0, 0, ww_height / 2 + flange_thickness / 2))
                top_flange = top_flange_solid - top_hole

                bottom_flange_solid = cq.Workplane("XY").box(flange_outer_x * 2, flange_half_y * 2, flange_thickness).translate((0, 0, -(ww_height / 2 + flange_thickness / 2)))
                bottom_hole = cq.Workplane("XY").cylinder(flange_thickness * 1.1, hole_radius).translate((0, 0, -(ww_height / 2 + flange_thickness / 2)))
                bottom_flange = bottom_flange_solid - bottom_hole
                flanges = top_flange + bottom_flange
            else:
                depth = winding_window.get("radialHeight", ww_width) if winding_window.get("radialHeight") else ww_width
                outer_width = ww_width + wall_thickness * 2
                outer_depth = depth + wall_thickness * 2
                flange_width = outer_width + flange_extension * 2
                flange_depth = outer_depth + flange_extension * 2

                central_hole_width = depth * 0.8
                central_hole_depth = depth * 0.8

                top_flange_solid = cq.Workplane("XY").box(flange_width, flange_depth, flange_thickness).translate((0, 0, ww_height / 2 + flange_thickness / 2))
                top_hole = cq.Workplane("XY").box(central_hole_width, central_hole_depth, flange_thickness * 1.1).translate((0, 0, ww_height / 2 + flange_thickness / 2))
                top_flange = top_flange_solid - top_hole

                bottom_flange_solid = cq.Workplane("XY").box(flange_width, flange_depth, flange_thickness).translate((0, 0, -(ww_height / 2 + flange_thickness / 2)))
                bottom_hole = cq.Workplane("XY").box(central_hole_width, central_hole_depth, flange_thickness * 1.1).translate((0, 0, -(ww_height / 2 + flange_thickness / 2)))
                bottom_flange = bottom_flange_solid - bottom_hole
                flanges = top_flange + bottom_flange

            return flanges

        def get_mounting_pins(self, data, outer_radius):
            dims = data.get("dimensions", {})
            pin_count = dims.get("pinCount", 0)
            if pin_count == 0:
                return None

            pin_diameter = dims.get("pinDiameter", 0.0008)
            pin_length = dims.get("pinLength", 0.003)
            flange_thickness = dims.get("flangeThickness", 0.001)
            ww_height = outer_radius

            pins = None
            angle_step = 360 / pin_count

            for i in range(pin_count):
                angle = math.radians(i * angle_step)
                x = ww_height * 0.8 * math.cos(angle)
                z = ww_height * 0.8 * math.sin(angle)

                pin = cq.Workplane("XZ").cylinder(pin_length, pin_diameter / 2).translate((x, -(flange_thickness + pin_length / 2), z))

                if pins is None:
                    pins = pin
                else:
                    pins = pins + pin

            return pins

    class IWinding(metaclass=ABCMeta):
        def __init__(self):
            self.output_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../output/"

        def set_output_path(self, output_path):
            self.output_path = output_path

        @abstractmethod
        def get_single_turn(self, data, position, _turn_index):
            raise NotImplementedError

        @abstractmethod
        def get_layer(self, data, layer_index, bobbin_inner_dims):
            raise NotImplementedError

        def calculate_turn_positions(self, data, bobbin_inner_height):
            wire_diameter = data.get("wireDiameter", 0.0005)
            insulation = data.get("insulationThickness", 0.00005)
            total_wire_diameter = wire_diameter + 2 * insulation
            num_turns = data.get("numberOfTurns", 1)
            num_layers = data.get("numberOfLayers", 1)

            turns_per_layer = num_turns // num_layers
            positions = []

            for layer in range(num_layers):
                for turn in range(turns_per_layer):
                    y_pos = -bobbin_inner_height / 2 + total_wire_diameter / 2 + turn * total_wire_diameter
                    positions.append({"layer": layer, "turn": turn, "y": y_pos, "layer_offset": layer * total_wire_diameter})

            return positions

        def get_winding(self, data, bobbin_dims, name="Winding", save_files=False, export_files=True):
            try:
                project_name = f"{name}".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")

                turns_description = data.get("turnsDescription", [])
                winding_name = data.get("windingName", name)

                if turns_description:
                    wire_diameter = data.get("wireDiameter")
                    winding = self.get_winding_from_mas(turns_description, winding_name, wire_diameter)
                    if winding is None:
                        turns_description = []

                if not turns_description:
                    num_turns = data.get("numberOfTurns", 1)
                    if num_turns > 100:
                        winding = self.get_bulk_winding(data, bobbin_dims)
                    else:
                        winding = self.get_detailed_winding(data, bobbin_dims)

                if winding is None:
                    return (None, None) if export_files else None

                coords = data.get("coordinates", [0, 0, 0])
                rotation = data.get("rotation", [0, 0, 0])

                if rotation[0] != 0:
                    winding = winding.rotate((1, 0, 0), (-1, 0, 0), rotation[0] / math.pi * 180)
                if rotation[1] != 0:
                    winding = winding.rotate((0, 1, 0), (0, -1, 0), rotation[1] / math.pi * 180)
                if rotation[2] != 0:
                    winding = winding.rotate((0, 0, 1), (0, 0, -1), rotation[2] / math.pi * 180)

                winding = winding.translate(convert_axis(coords))

                pathlib.Path(self.output_path).mkdir(parents=True, exist_ok=True)

                if export_files:
                    from cadquery import exporters

                    scaled_winding = winding.newObject([o.scale(1000) for o in winding.objects])
                    exporters.export(scaled_winding, f"{self.output_path}/{project_name}.step", "STEP")
                    exporters.export(scaled_winding, f"{self.output_path}/{project_name}.stl", "STL")
                    return f"{self.output_path}/{project_name}.step", f"{self.output_path}/{project_name}.stl"
                else:
                    return winding

            except Exception:
                return (None, None) if export_files else None

        def get_bulk_winding(self, data, bobbin_dims):
            wire_diameter = data.get("wireDiameter", 0.0005)
            insulation = data.get("insulationThickness", 0.00005)
            num_layers = data.get("numberOfLayers", 1)
            total_wire_diameter = wire_diameter + 2 * insulation

            ww_height = bobbin_dims.get("height", 0.01)
            ww_width = bobbin_dims.get("width", 0.005)

            layer_thickness = total_wire_diameter * num_layers
            winding_length = ww_height * 0.9

            bulk = cq.Workplane("XY").box(layer_thickness, ww_width * 0.8, winding_length).translate((ww_width / 2 + layer_thickness / 2, 0, 0))

            return bulk

        @abstractmethod
        def get_detailed_winding(self, data, bobbin_dims):
            raise NotImplementedError

    class RoundWireWinding(IWinding):
        def get_single_turn(self, data, position, _turn_index):
            wire_diameter = data.get("wireDiameter", 0.0005)

            radius = position.get("radius", 0.005)
            y_pos = position.get("y", 0)
            layer_offset = position.get("layer_offset", 0)

            turn_radius = radius + layer_offset

            path = cq.Workplane("XY").center(0, 0).circle(turn_radius)

            wire_profile = cq.Workplane("XZ").center(turn_radius, 0).circle(wire_diameter / 2)

            turn = wire_profile.sweep(path, isFrenet=True)
            turn = turn.translate((0, 0, y_pos))

            return turn

        def create_turn_from_description(self, turn_desc, wire_diameter=None):
            radial_pos = turn_desc.coordinates[0]
            z_pos = turn_desc.coordinates[1]

            if wire_diameter is None:
                if turn_desc.dimensions:
                    wire_diameter = turn_desc.dimensions[0]
                else:
                    wire_diameter = 0.0005

            path = cq.Workplane("XY").center(0, 0).circle(radial_pos)

            wire_profile = cq.Workplane("XZ").center(radial_pos, 0).circle(wire_diameter / 2)

            turn = wire_profile.sweep(path, isFrenet=True)
            turn = turn.translate((0, 0, z_pos))

            return turn

        def get_winding_from_mas(self, turns_description, winding_name, wire_diameter=None):
            all_turns = [TurnDescription.from_dict(t) for t in turns_description]
            winding_turns = [t for t in all_turns if t.winding == winding_name]

            if not winding_turns:
                winding_turns = [t for t in all_turns if winding_name in t.winding]

            if not winding_turns:
                return None

            winding = None
            for turn_desc in winding_turns:
                turn = self.create_turn_from_description(turn_desc, wire_diameter)

                if winding is None:
                    winding = turn
                else:
                    winding = winding + turn

            return winding

        def get_layer(self, data, layer_index, bobbin_inner_dims):
            wire_diameter = data.get("wireDiameter", 0.0005)
            insulation = data.get("insulationThickness", 0.00005)
            total_wire_diameter = wire_diameter + 2 * insulation
            num_turns = data.get("numberOfTurns", 1)
            num_layers = data.get("numberOfLayers", 1)

            ww_height = bobbin_inner_dims.get("height", 0.01)
            ww_width = bobbin_inner_dims.get("width", 0.005)
            column_width = bobbin_inner_dims.get("columnWidth", 0)
            column_shape = bobbin_inner_dims.get("columnShape", "rectangular")

            turns_per_layer = num_turns // num_layers
            layer = None

            if column_shape == "round" and column_width > 0:
                wall_thickness = 0.0005
                base_radius = column_width / 2 + wall_thickness + total_wire_diameter / 2
            else:
                base_radius = ww_width / 2 + total_wire_diameter / 2

            for turn_idx in range(turns_per_layer):
                z_pos = -ww_height / 2 + total_wire_diameter / 2 + turn_idx * total_wire_diameter

                if z_pos > ww_height / 2 - total_wire_diameter / 2:
                    break

                position = {"radius": base_radius, "y": z_pos, "layer_offset": layer_index * total_wire_diameter}

                turn = self.get_single_turn(data, position, turn_idx)

                if layer is None:
                    layer = turn
                else:
                    layer = layer + turn

            return layer

        def get_detailed_winding(self, data, bobbin_dims):
            num_layers = data.get("numberOfLayers", 1)
            winding = None

            for layer_idx in range(num_layers):
                layer = self.get_layer(data, layer_idx, bobbin_dims)
                if layer is not None:
                    if winding is None:
                        winding = layer
                    else:
                        winding = winding + layer

            return winding


if __name__ == "__main__":  # pragma: no cover
    with open(f"{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson", "r") as f:
        for ndjson_line in f.readlines():
            data = json.loads(ndjson_line)
            if data["name"] == "PQ 40/40":
                # if data["family"] in ['pm']:
                # if data["family"] not in ['ui']:
                core = CadQueryBuilder().factory(data)
                core.get_core(data, None)
                # break
