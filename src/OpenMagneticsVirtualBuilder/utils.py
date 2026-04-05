"""Utility types and helpers for OpenMagneticsVirtualBuilder.

Provides the :class:`ShapeFamily` enum listing all 21 supported core families,
dimension flattening for MAS schema data, and numeric rounding helpers.
"""

import copy
import enum

import numpy


class Meta(enum.EnumMeta):
    """Metaclass that adds ``in`` operator support to enums."""

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class ShapeFamily(enum.Enum, metaclass=Meta):
    """
    Core shape families as defined in MAS magnetic/core/shape.json#/$defs/coreShapeFamily.
    Values are MAS string values (lowercase with spaces for planar types).
    """
    C = "c"
    DRUM = "drum"
    E = "e"
    EC = "ec"
    EFD = "efd"
    EI = "ei"
    EL = "el"
    ELP = "elp"
    EP = "ep"
    EPX = "epx"
    EQ = "eq"
    ER = "er"
    ETD = "etd"
    H = "h"
    LP = "lp"
    P = "p"
    PLANAR_E = "planar e"
    PLANAR_EL = "planar el"
    PLANAR_ER = "planar er"
    PM = "pm"
    PQ = "pq"
    PQI = "pqi"
    RM = "rm"
    ROD = "rod"
    T = "t"
    U = "u"
    UI = "ui"
    UR = "ur"
    UT = "ut"


class GapType(enum.Enum, metaclass=Meta):
    """Gap types as defined in MAS magnetic/core/gap.json#/$defs/gapType."""
    ADDITIVE = "additive"
    SUBTRACTIVE = "subtractive"
    RESIDUAL = "residual"


class TurnCrossSectionalShape(enum.Enum, metaclass=Meta):
    """Turn cross-sectional shapes as defined in MAS magnetic/coil.json#/$defs/turnCrossSectionalShape."""
    ROUND = "round"
    RECTANGULAR = "rectangular"
    OVAL = "oval"


def flatten_dimensions(data, scale_factor=1.0):
    """Convert MAS min/max/nominal dimension values to a flat dict of floats.

    MAS dimensions can be specified as ``{"nominal": v}``,
    ``{"minimum": v, "maximum": v}``, or plain numbers.  This function
    resolves each to a single nominal value and applies *scale_factor*.

    Args:
        data: Dict with a ``"dimensions"`` key containing the MAS dimension
            map.
        scale_factor: Multiplicative factor applied to every value
            (e.g. ``1000`` to convert m to mm).

    Returns:
        Dict mapping dimension letter (``"A"``, ``"B"``, ...) to a float
        value.  The ``"alpha"`` key, if present, is excluded.
    """
    dimensions = copy.deepcopy(data["dimensions"])
    for k, v in dimensions.items():
        if isinstance(v, dict):
            if "nominal" not in v or v["nominal"] is None:
                if "maximum" not in v or v["maximum"] is None:
                    v["nominal"] = v["minimum"]
                elif "minimum" not in v or v["minimum"] is None:
                    v["nominal"] = v["maximum"]
                else:
                    v["nominal"] = round((v["maximum"] + v["minimum"]) / 2, 6)
        else:
            dimensions[k] = {"nominal": v}
    return {k: v["nominal"] * scale_factor for k, v in dimensions.items() if k != "alpha"}


class BuilderBase:
    """Shared base for FreeCADBuilder and CadQueryBuilder with common factory/families logic."""

    def factory(self, data):
        """Look up the shape builder for the given family name.

        Args:
            data: Dict with a ``"family"`` key.

        Returns:
            Shape builder instance from ``self.shapers``.
        """
        family = ShapeFamily[data["family"].upper().replace(" ", "_")]
        return self.shapers[family]

    def get_families(self):
        """Return dimensions and subtypes for every registered shape family."""
        return {shaper.name.lower().replace("_", " "): self.factory({"family": shaper.name}).get_dimensions_and_subtypes() for shaper in self.shapers}


def decimal_ceil(a, precision=0):
    """Ceiling rounded to *precision* decimal places.

    Args:
        a: Numeric value.
        precision: Number of decimal places.

    Returns:
        Rounded-up value as a float.
    """
    return numpy.true_divide(numpy.ceil(a * 10**precision), 10**precision)


def decimal_floor(a, precision=0):
    """Floor rounded to *precision* decimal places.

    Args:
        a: Numeric value.
        precision: Number of decimal places.

    Returns:
        Rounded-down value as a float.
    """
    return numpy.true_divide(numpy.floor(a * 10**precision), 10**precision)
