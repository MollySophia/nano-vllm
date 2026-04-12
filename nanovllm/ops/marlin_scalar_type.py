import functools
from dataclasses import dataclass
from enum import Enum


class NanRepr(Enum):
    NONE = 0
    IEEE_754 = 1
    EXTD_RANGE_MAX_MIN = 2


@dataclass(frozen=True)
class ScalarType:
    exponent: int
    mantissa: int
    signed: bool
    bias: int
    _finite_values_only: bool = False
    nan_repr: NanRepr = NanRepr.IEEE_754

    @functools.cached_property
    def id(self) -> int:
        val = 0
        offset = 0

        def or_and_advance(member, bit_width):
            nonlocal val
            nonlocal offset
            bit_mask = (1 << bit_width) - 1
            val |= (int(member) & bit_mask) << offset
            offset += bit_width

        or_and_advance(self.exponent, 8)
        or_and_advance(self.mantissa, 8)
        or_and_advance(self.signed, 1)
        or_and_advance(self.bias, 32)
        or_and_advance(self._finite_values_only, 1)
        or_and_advance(self.nan_repr.value, 8)
        return val

    @property
    def size_bits(self) -> int:
        return self.exponent + self.mantissa + int(self.signed)

    @classmethod
    def int_(cls, size_bits: int, bias: int | None = None) -> "ScalarType":
        return cls(0, size_bits - 1, True, bias or 0)

    @classmethod
    def uint(cls, size_bits: int, bias: int | None = None) -> "ScalarType":
        return cls(0, size_bits, False, bias or 0)

    @classmethod
    def float_IEEE754(cls, exponent: int, mantissa: int) -> "ScalarType":
        return cls(exponent, mantissa, True, 0)

    @classmethod
    def float_(
        cls,
        exponent: int,
        mantissa: int,
        finite_values_only: bool,
        nan_repr: NanRepr,
    ) -> "ScalarType":
        return cls(exponent, mantissa, True, 0, finite_values_only, nan_repr)


class scalar_types:
    uint4 = ScalarType.uint(4, None)
    uint8 = ScalarType.uint(8, None)
    int8 = ScalarType.int_(8, None)
    float8_e4m3fn = ScalarType.float_(4, 3, True, NanRepr.EXTD_RANGE_MAX_MIN)
    float4_e2m1f = ScalarType.float_(2, 1, True, NanRepr.NONE)
    float16 = ScalarType.float_IEEE754(5, 10)
    bfloat16 = ScalarType.float_IEEE754(8, 7)
    uint4b8 = ScalarType.uint(4, 8)
    uint8b128 = ScalarType.uint(8, 128)
