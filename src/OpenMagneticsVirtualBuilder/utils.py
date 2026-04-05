import enum
import numpy


class Meta(enum.EnumMeta):
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


def decimal_ceil(a, precision=0):
    return numpy.true_divide(numpy.ceil(a * 10**precision), 10**precision)


def decimal_floor(a, precision=0):
    return numpy.true_divide(numpy.floor(a * 10**precision), 10**precision)
