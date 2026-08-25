"""DATA_FREE_NONPRODUCTION oracle for ``phasepair-base-dropout-v1``.

The implementation is pure stdlib and stateless.  It neither imports a random
module nor creates a production training schedule or a runtime PASS receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Iterator, Mapping, Optional, Tuple


H0_DOMAIN = b"phasepair-base-dropout-v1"
INVENTORY_DOMAIN = b"phasepair-base-dropout-site-inventory-v1"
TRACE_DOMAIN = b"phasepair-base-dropout-application-trace-v1"
MASK64 = (1 << 64) - 1
PHI = 0x9E3779B97F4A7C15
MIX1 = 0xBF58476D1CE4E5B9
MIX2 = 0x94D049BB133111EB
DROP_THRESHOLD = 0x1999999999999999
KEEP_FLOAT32_BITS = 0x3F8E38E4
KEEP_FLOAT32_LE = b"\xe4\x38\x8e\x3f"
DROP_FLOAT32_LE = b"\x00\x00\x00\x00"

ARCHITECTURES = ("mime", "early", "late")
AXIS_CODES: Mapping[str, int] = MappingProxyType(
    {"P": 0x01, "B": 0x02, "H": 0x03, "T": 0x04, "D512": 0x05, "F2048": 0x06}
)


class DropoutContractError(ValueError):
    """Fail-closed base-dropout contract error."""


class InventoryMismatchError(DropoutContractError):
    pass


class RuntimePolicyError(DropoutContractError):
    pass


def _require_exact_int(
    value: object, label: str, *, minimum: int = 0, maximum: int = MASK64
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a built-in int")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _require_exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a built-in bool")
    return value


def _require_ascii_name(value: object, label: str = "site_name_ascii") -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a built-in str")
    if not value or "\x00" in value or any(ord(ch) < 0x20 for ch in value):
        raise ValueError(f"{label} must be nonempty and contain no controls")
    try:
        raw = value.encode("ascii", "strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be ASCII") from exc
    if len(raw) > 65535:
        raise ValueError(f"{label} exceeds uint16 length")
    return value


def _require_shape(value: object, *, require_pass_pair: bool = True) -> Tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError("shape must be a built-in tuple")
    if not value or len(value) > 255:
        raise ValueError("shape rank must be in [1,255]")
    numel = 1
    for index, dimension in enumerate(value):
        _require_exact_int(
            dimension, f"shape[{index}]", minimum=1, maximum=(1 << 32) - 1
        )
        numel *= dimension
        if numel >= (1 << 64):
            raise ValueError("shape numel must be less than 2^64")
    if require_pass_pair and value[0] != 2:
        raise DropoutContractError("shape[0] must be the single P=2 [AB,BA] call")
    return value


def _require_architecture(value: object) -> str:
    if type(value) is not str:
        raise TypeError("architecture must be a built-in str")
    if value not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {value!r}")
    return value


def _numel(shape: Tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def h0_preimage(
    *,
    seed: int,
    epoch_index: int,
    global_optimizer_step: int,
    site_ordinal: int,
    site_name_ascii: str,
    shape: Tuple[int, ...],
) -> bytes:
    """Encode the exact seven-field H0 tuple; no system/pass keyword exists."""

    seed = _require_exact_int(seed, "seed")
    epoch_index = _require_exact_int(epoch_index, "epoch_index", maximum=29)
    global_optimizer_step = _require_exact_int(
        global_optimizer_step, "global_optimizer_step", maximum=(1 << 32) - 1
    )
    site_ordinal = _require_exact_int(
        site_ordinal, "site_ordinal", maximum=(1 << 16) - 1
    )
    site_name_ascii = _require_ascii_name(site_name_ascii)
    shape = _require_shape(shape, require_pass_pair=True)
    name_bytes = site_name_ascii.encode("ascii")
    output = bytearray(H0_DOMAIN)
    output.append(0)
    output.extend(seed.to_bytes(8, "big"))
    output.extend(epoch_index.to_bytes(4, "big"))
    output.extend(global_optimizer_step.to_bytes(4, "big"))
    output.extend(site_ordinal.to_bytes(2, "big"))
    output.extend(len(name_bytes).to_bytes(2, "big"))
    output.extend(name_bytes)
    output.append(len(shape))
    for dimension in shape:
        output.extend(dimension.to_bytes(4, "big"))
    return bytes(output)


def h0_digest(
    *,
    seed: int,
    epoch_index: int,
    global_optimizer_step: int,
    site_ordinal: int,
    site_name_ascii: str,
    shape: Tuple[int, ...],
) -> bytes:
    return hashlib.sha256(
        h0_preimage(
            seed=seed,
            epoch_index=epoch_index,
            global_optimizer_step=global_optimizer_step,
            site_ordinal=site_ordinal,
            site_name_ascii=site_name_ascii,
            shape=shape,
        )
    ).digest()


def splitmix64_word(base: int, flat_index: int) -> int:
    """Return the contract word for C-flat index ``flat_index``."""

    base = _require_exact_int(base, "base")
    flat_index = _require_exact_int(flat_index, "flat_index", maximum=MASK64 - 1)
    z = (base + PHI * (flat_index + 1)) & MASK64
    z = ((z ^ (z >> 30)) * MIX1) & MASK64
    z = ((z ^ (z >> 27)) * MIX2) & MASK64
    z = (z ^ (z >> 31)) & MASK64
    return z


def keep_from_mixed_word(word: int) -> bool:
    """Integer threshold predicate; TH-1 drops and TH keeps."""

    word = _require_exact_int(word, "word")
    return word >= DROP_THRESHOLD


def iter_keep_bits(base: int, numel: int) -> Iterator[bool]:
    base = _require_exact_int(base, "base")
    numel = _require_exact_int(numel, "numel", minimum=1, maximum=MASK64)
    for flat_index in range(numel):
        yield keep_from_mixed_word(splitmix64_word(base, flat_index))


def pack_keep_bits_lsb_first(bits: Tuple[bool, ...]) -> bytes:
    if type(bits) is not tuple:
        raise TypeError("bits must be a built-in tuple")
    if not bits:
        raise ValueError("bits must not be empty")
    output = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        _require_exact_bool(bit, f"bits[{index}]")
        if bit:
            output[index // 8] |= 1 << (index % 8)
    return bytes(output)


@dataclass(frozen=True, slots=True)
class DropoutMask:
    shape: Tuple[int, ...]
    h0: bytes
    base: int
    first32_keep: str
    drop_count: int
    packed_keep_bits: bytes
    expanded_float32_le: bytes

    @property
    def numel(self) -> int:
        return _numel(self.shape)

    @property
    def packed_sha256(self) -> str:
        return hashlib.sha256(self.packed_keep_bits).hexdigest()

    @property
    def expanded_sha256(self) -> str:
        return hashlib.sha256(self.expanded_float32_le).hexdigest()


def generate_mask(
    *,
    seed: int,
    epoch_index: int,
    global_optimizer_step: int,
    site_ordinal: int,
    site_name_ascii: str,
    shape: Tuple[int, ...],
) -> DropoutMask:
    """Generate one stateless vectorized P=2 oracle mask."""

    shape = _require_shape(shape, require_pass_pair=True)
    digest = h0_digest(
        seed=seed,
        epoch_index=epoch_index,
        global_optimizer_step=global_optimizer_step,
        site_ordinal=site_ordinal,
        site_name_ascii=site_name_ascii,
        shape=shape,
    )
    base = int.from_bytes(digest[:8], "big")
    numel = _numel(shape)
    packed = bytearray((numel + 7) // 8)
    expanded = bytearray(numel * 4)
    first = []
    drop_count = 0
    for flat_index, keep in enumerate(iter_keep_bits(base, numel)):
        if flat_index < 32:
            first.append("1" if keep else "0")
        offset = 4 * flat_index
        if keep:
            packed[flat_index // 8] |= 1 << (flat_index % 8)
            expanded[offset : offset + 4] = KEEP_FLOAT32_LE
        else:
            drop_count += 1
            expanded[offset : offset + 4] = DROP_FLOAT32_LE
    return DropoutMask(
        shape,
        digest,
        base,
        "".join(first),
        drop_count,
        bytes(packed),
        bytes(expanded),
    )


@dataclass(frozen=True, slots=True)
class DropoutSite:
    ordinal: int
    name: str
    axes: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_exact_int(self.ordinal, "site ordinal", maximum=(1 << 16) - 1)
        _require_ascii_name(self.name, "site name")
        if type(self.axes) is not tuple or not self.axes:
            raise TypeError("site axes must be a nonempty built-in tuple")
        for index, axis in enumerate(self.axes):
            if type(axis) is not str or axis not in AXIS_CODES:
                raise ValueError(f"unknown axis at axes[{index}]")
        if self.axes[0] != "P":
            raise ValueError("every site must lead with P")


def _rebuild_site(value: object, label: str) -> DropoutSite:
    if type(value) is not DropoutSite:
        raise TypeError(f"{label} must be exactly DropoutSite")
    return DropoutSite(value.ordinal, value.name, value.axes)


def _rebuild_site_tuple(value: object, label: str) -> Tuple[DropoutSite, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a built-in tuple")
    return tuple(
        _rebuild_site(site, f"{label}[{index}]")
        for index, site in enumerate(value)
    )


PBHTT = ("P", "B", "H", "T", "T")
PBTD = ("P", "B", "T", "D512")
PBTF = ("P", "B", "T", "F2048")


def _mime_sites() -> Tuple[DropoutSite, ...]:
    suffixes = (
        ("self_a.attn", PBHTT),
        ("self_a.proj", PBTD),
        ("self_b.attn", PBHTT),
        ("self_b.proj", PBTD),
        ("cross_a_from_b.attn", PBHTT),
        ("cross_a_from_b.proj", PBTD),
        ("cross_b_from_a.attn", PBHTT),
        ("cross_b_from_a.proj", PBTD),
        ("ffn_a.inner", PBTF),
        ("ffn_a.ffn", PBTD),
        ("ffn_b.inner", PBTF),
        ("ffn_b.ffn", PBTD),
    )
    sites: list[DropoutSite] = []
    for block in range(4):
        for q, (suffix, axes) in enumerate(suffixes):
            sites.append(
                DropoutSite(12 * block + q, f"blocks.{block:02d}.{suffix}", axes)
            )
    sites.append(DropoutSite(48, "fusion.inner", PBTD))
    return tuple(sites)


def _early_sites() -> Tuple[DropoutSite, ...]:
    suffixes = (
        ("self.attn", PBHTT),
        ("self.proj", PBTD),
        ("ffn.inner", PBTF),
        ("ffn.ffn", PBTD),
    )
    return tuple(
        DropoutSite(4 * block + q, f"blocks.{block:02d}.{suffix}", axes)
        for block in range(4)
        for q, (suffix, axes) in enumerate(suffixes)
    )


def _late_sites() -> Tuple[DropoutSite, ...]:
    suffixes = (
        ("self.attn", PBHTT),
        ("self.proj", PBTD),
        ("ffn.inner", PBTF),
        ("ffn.ffn", PBTD),
    )
    sites: list[DropoutSite] = []
    for actor_index, actor in enumerate(("actor_a", "actor_b")):
        for block in range(4):
            for q, (suffix, axes) in enumerate(suffixes):
                sites.append(
                    DropoutSite(
                        16 * actor_index + 4 * block + q,
                        f"{actor}.blocks.{block:02d}.{suffix}",
                        axes,
                    )
                )
    return tuple(sites)


SITE_INVENTORIES: Mapping[str, Tuple[DropoutSite, ...]] = MappingProxyType(
    {"mime": _mime_sites(), "early": _early_sites(), "late": _late_sites()}
)


@dataclass(frozen=True, slots=True)
class ByteIdentity:
    byte_count: int
    sha256: str


INVENTORY_IDENTITIES: Mapping[str, ByteIdentity] = MappingProxyType(
    {
        "mime": ByteIdentity(
            1632, "aa348eaeb6045a66bfb4efe435eac7d77d6c09e0a30d89fb2a36d199c06d66a9"
        ),
        "early": ByteIdentity(
            487, "d2ea7874af61dcac27092a23ae3688718b0b876b9d7f630a208d3f0421e1537b"
        ),
        "late": ByteIdentity(
            1187, "8da922063b717cff35abdb11402d4c4607498649992b57b98cc61996bdb6ad16"
        ),
    }
)

TRACE_IDENTITIES: Mapping[str, ByteIdentity] = MappingProxyType(
    {
        "mime": ByteIdentity(
            152, "77a59a881b589b399ab6c782c5639942e4080893d6c8106b968eea102917312d"
        ),
        "early": ByteIdentity(
            86, "7c41f6589db05caebe80e114355c9bfd5fa749b0c92e7948a2975ed390143868"
        ),
        "late": ByteIdentity(
            118, "6825fd7926d4c3ddf9ccee706994da69f9a589c52b4f9bbad86d9fffd3470a8e"
        ),
    }
)

MIME_SWAP_ORDINAL_1_3_INVENTORY_SHA256 = (
    "793983f7bbfdb23b9c59390a2c419613ab1e9247544c018a2154bb7887cb3ca8"
)
MIME_SWAP_ORDINAL_1_3_TRACE_SHA256 = (
    "eb3f1e8246907f43e164dfc6edca88619465f71fffae452e13d3c35ec344d08f"
)


def site_inventory(architecture: str) -> Tuple[DropoutSite, ...]:
    return SITE_INVENTORIES[_require_architecture(architecture)]


def encode_inventory(sites: Tuple[DropoutSite, ...]) -> bytes:
    """Encode rows in supplied traversal order; validation is a separate gate."""

    checked_sites = _rebuild_site_tuple(sites, "sites")
    if not checked_sites or len(checked_sites) > (1 << 16) - 1:
        raise ValueError("site count must be in [1,65535]")
    output = bytearray(INVENTORY_DOMAIN)
    output.append(0)
    output.extend(len(checked_sites).to_bytes(2, "big"))
    for site in checked_sites:
        name = site.name.encode("ascii")
        output.extend(site.ordinal.to_bytes(2, "big"))
        output.extend(len(name).to_bytes(2, "big"))
        output.extend(name)
        output.append(len(site.axes))
        output.extend(AXIS_CODES[axis] for axis in site.axes)
    return bytes(output)


def validate_canonical_inventory(
    architecture: str, sites: Tuple[DropoutSite, ...]
) -> ByteIdentity:
    architecture = _require_architecture(architecture)
    checked_sites = _rebuild_site_tuple(sites, "sites")
    if checked_sites != SITE_INVENTORIES[architecture]:
        raise InventoryMismatchError(
            "site inventory must use exact architecture traversal and ordinals"
        )
    encoded = encode_inventory(checked_sites)
    actual = ByteIdentity(len(encoded), hashlib.sha256(encoded).hexdigest())
    if actual != INVENTORY_IDENTITIES[architecture]:
        raise InventoryMismatchError("site inventory byte identity mismatch")
    return actual


def canonical_inventory_bytes(architecture: str) -> bytes:
    architecture = _require_architecture(architecture)
    sites = SITE_INVENTORIES[architecture]
    validate_canonical_inventory(architecture, sites)
    return encode_inventory(sites)


def site_shape(site: DropoutSite, *, batch_size: int, time_steps: int) -> Tuple[int, ...]:
    checked_site = _rebuild_site(site, "site")
    batch_size = _require_exact_int(batch_size, "batch_size", minimum=1, maximum=128)
    time_steps = _require_exact_int(time_steps, "time_steps", minimum=1, maximum=300)
    dimensions = {
        "P": 2,
        "B": batch_size,
        "H": 4,
        "T": time_steps,
        "D512": 512,
        "F2048": 2048,
    }
    return tuple(dimensions[axis] for axis in checked_site.axes)


def generate_site_mask(
    architecture: str,
    *,
    seed: int,
    epoch_index: int,
    global_optimizer_step: int,
    site_ordinal: int,
    batch_size: int,
    time_steps: int,
) -> DropoutMask:
    architecture = _require_architecture(architecture)
    site_ordinal = _require_exact_int(
        site_ordinal, "site_ordinal", maximum=(1 << 16) - 1
    )
    sites = SITE_INVENTORIES[architecture]
    if site_ordinal >= len(sites):
        raise DropoutContractError("site_ordinal is outside the canonical inventory")
    site = sites[site_ordinal]
    if site.ordinal != site_ordinal:
        raise InventoryMismatchError("canonical site ordinal drift")
    return generate_mask(
        seed=seed,
        epoch_index=epoch_index,
        global_optimizer_step=global_optimizer_step,
        site_ordinal=site.ordinal,
        site_name_ascii=site.name,
        shape=site_shape(site, batch_size=batch_size, time_steps=time_steps),
    )


def application_trace_bytes(
    sites: Tuple[DropoutSite, ...], global_optimizer_steps: Tuple[int, ...]
) -> bytes:
    checked_sites = _rebuild_site_tuple(sites, "sites")
    if type(global_optimizer_steps) is not tuple:
        raise TypeError("global_optimizer_steps must be a built-in tuple")
    if not checked_sites or len(checked_sites) > (1 << 16) - 1:
        raise ValueError("site count must be in [1,65535]")
    if len(global_optimizer_steps) > (1 << 32) - 1:
        raise ValueError("step count must be in [0,2^32-1]")
    output = bytearray(TRACE_DOMAIN)
    output.append(0)
    output.extend(len(global_optimizer_steps).to_bytes(4, "big"))
    output.extend(len(checked_sites).to_bytes(2, "big"))
    previous: Optional[int] = None
    for index, step in enumerate(global_optimizer_steps):
        step = _require_exact_int(
            step, f"global_optimizer_steps[{index}]", maximum=(1 << 32) - 1
        )
        if previous is not None and step <= previous:
            raise DropoutContractError("global optimizer steps must be strictly increasing")
        previous = step
        output.extend(step.to_bytes(4, "big"))
        for site in checked_sites:
            output.extend(site.ordinal.to_bytes(2, "big"))
    return bytes(output)


def canonical_one_step_trace(architecture: str) -> bytes:
    architecture = _require_architecture(architecture)
    sites = SITE_INVENTORIES[architecture]
    validate_canonical_inventory(architecture, sites)
    trace = application_trace_bytes(sites, (0,))
    identity = ByteIdentity(len(trace), hashlib.sha256(trace).hexdigest())
    if identity != TRACE_IDENTITIES[architecture]:
        raise InventoryMismatchError("one-step application trace identity mismatch")
    return trace


@dataclass(frozen=True, slots=True)
class GoldenFixture:
    label: str
    seed: int
    epoch_index: int
    global_optimizer_step: int
    site_ordinal: int
    site_name_ascii: str
    shape: Tuple[int, ...]
    h0_sha256: str
    base_hex: str
    first32_keep: str
    drop_count: int
    packed_sha256: str
    expanded_sha256: str
    packed_hex: Optional[str] = None
    preimage_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        _require_ascii_name(self.label, "fixture label")
        _require_exact_int(self.seed, "fixture seed")
        _require_exact_int(self.epoch_index, "fixture epoch_index", maximum=29)
        _require_exact_int(
            self.global_optimizer_step,
            "fixture global_optimizer_step",
            maximum=(1 << 32) - 1,
        )
        _require_exact_int(
            self.site_ordinal, "fixture site_ordinal", maximum=(1 << 16) - 1
        )
        _require_ascii_name(self.site_name_ascii, "fixture site_name_ascii")
        _require_shape(self.shape, require_pass_pair=True)
        for label, value, length in (
            ("h0_sha256", self.h0_sha256, 64),
            ("base_hex", self.base_hex, 16),
            ("packed_sha256", self.packed_sha256, 64),
            ("expanded_sha256", self.expanded_sha256, 64),
        ):
            if (
                type(value) is not str
                or len(value) != length
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise TypeError(f"fixture {label} must be lowercase hex[{length}]")
        if (
            type(self.first32_keep) is not str
            or len(self.first32_keep) != 32
            or any(character not in "01" for character in self.first32_keep)
        ):
            raise TypeError("fixture first32_keep must be an exact 32-bit string")
        _require_exact_int(
            self.drop_count,
            "fixture drop_count",
            maximum=_numel(self.shape),
        )
        if self.packed_hex is not None:
            if (
                type(self.packed_hex) is not str
                or len(self.packed_hex) != 2 * ((_numel(self.shape) + 7) // 8)
                or any(
                    character not in "0123456789abcdef"
                    for character in self.packed_hex
                )
            ):
                raise TypeError("fixture packed_hex is not canonical lowercase hex")
        if self.preimage_bytes is not None:
            _require_exact_int(
                self.preimage_bytes, "fixture preimage_bytes", minimum=1
            )


GOLDEN_FIXTURES: Tuple[GoldenFixture, ...] = (
    GoldenFixture(
        "MIME_SMALL",
        1729,
        0,
        0,
        0,
        "blocks.00.self_a.attn",
        (2, 2, 4, 3, 3),
        "8cea86bd9c255cb42379fa23d7f5fa776f45437e2a1142351a4e847fc8b81048",
        "8cea86bd9c255cb4",
        "11111101110111110111111111101111",
        18,
        "e28f5de37639f08d12606eb44efaeeb1f36b2b0a8408cfd481a2ad40712b0e28",
        "69a38dc3bbfd1fa19a0a9a34077f7ad22f92d0d5b671d3a4e0daca862feb1646",
        "bffbfef7bf75fff7fffefffd7dfff7fdf3bf",
        88,
    ),
    GoldenFixture(
        "MIME_B128",
        1729,
        0,
        0,
        0,
        "blocks.00.self_a.attn",
        (2, 128, 4, 3, 3),
        "94bf4ff117cbf6d73c33b2803005429296f0f218fd61fad04a1dea4379b71e87",
        "94bf4ff117cbf6d7",
        "11111101111111101111111111111111",
        943,
        "2d379e24e65e14a334ecbe8cd1e7747d63eae29786acca5b108aadaad5fb038e",
        "5cf16c06bd8c127660137fad206cdcbc0d0e72a2f1302b81326931c76c6bb776",
        None,
        88,
    ),
    GoldenFixture(
        "EARLY_SMALL",
        1729,
        0,
        0,
        0,
        "blocks.00.self.attn",
        (2, 2, 4, 3, 3),
        "3051926bad90bacf8f188cae87d00587fb0b102b851fb7a0e66ca47de4c1d217",
        "3051926bad90bacf",
        "11111111011111111111111101100100",
        17,
        "97a5f8e7d96cf008a93eb14a35d1fd2c816966503eea7273b7ff0ad504e6c40f",
        "3a99d4693584652b3cb16d901bf624e069b232af518189464ede175679673f34",
    ),
    GoldenFixture(
        "EARLY_B128",
        1729,
        0,
        0,
        0,
        "blocks.00.self.attn",
        (2, 128, 4, 3, 3),
        "a43368d722cfd72d1cb713b9bb790170c1ccfa5fefef127a541c9e9fbc9b8e3b",
        "a43368d722cfd72d",
        "10100111111111111101110111011101",
        943,
        "a1cd6e3e630e396c915a32b546346cf3eaf4f056fefaa226b967f52164838265",
        "67301d4cead8bf316fea2724589e2d71aa534cc64e2e613e94bdf08d30ac602d",
    ),
    GoldenFixture(
        "LATE_SMALL",
        1729,
        0,
        0,
        0,
        "actor_a.blocks.00.self.attn",
        (2, 2, 4, 3, 3),
        "eb5ff8b764dd63901ac7bd5978eca12080dd303700c22d5e6ffbcc95d3ea806e",
        "eb5ff8b764dd6390",
        "11111101011111111111111111101111",
        20,
        "fc456a9bcca5c28c8a2da16c1b9d24f968fdf7edd11ffcc184085ec7eef67054",
        "bdf5690c01e1b0ebec14e4fae722b0c5f257a6d41c7434b6c41d8274ced8de7f",
    ),
    GoldenFixture(
        "LATE_B128",
        1729,
        0,
        0,
        0,
        "actor_a.blocks.00.self.attn",
        (2, 128, 4, 3, 3),
        "b1187d67b5b173580bcd251677f2dbe9c90aeab0b8968cdd218b279c10b84c76",
        "b1187d67b5b17358",
        "10111111011011111111100111110111",
        901,
        "c292544341af0ecc3e358922b9e4e725386dc43e86a3058f61348dcdd1a7b02b",
        "acbb7276723d661a3845e83199025b0d59ba2050f71161c2102ac52bd78b1d88",
    ),
    GoldenFixture(
        "MIME_NONZERO_ENDIAN_ORDER",
        2718,
        1,
        44,
        9,
        "blocks.00.ffn_a.ffn",
        (2, 7, 5, 512),
        "cb77be17ec4b6f6f4cd8812507270bb586656d888c03f91f60c32e16a2d01833",
        "cb77be17ec4b6f6f",
        "11111111101001111111111111111100",
        3594,
        "b98cdfc19bc9d1f049948bbae1db748723eeddf618f77f75ab44a235ceed5b71",
        "b7b82ac341c663931cf5fa87ac9418cba45fbabe7b6977367834d68d790a25b8",
        None,
        82,
    ),
)


def verify_golden(fixture: GoldenFixture) -> DropoutMask:
    if type(fixture) is not GoldenFixture:
        raise TypeError("fixture must be exactly GoldenFixture")
    checked = GoldenFixture(
        fixture.label,
        fixture.seed,
        fixture.epoch_index,
        fixture.global_optimizer_step,
        fixture.site_ordinal,
        fixture.site_name_ascii,
        fixture.shape,
        fixture.h0_sha256,
        fixture.base_hex,
        fixture.first32_keep,
        fixture.drop_count,
        fixture.packed_sha256,
        fixture.expanded_sha256,
        fixture.packed_hex,
        fixture.preimage_bytes,
    )
    canonical = tuple(value for value in GOLDEN_FIXTURES if value.label == checked.label)
    if len(canonical) != 1 or checked != canonical[0]:
        raise DropoutContractError("fixture is not one of the exact pinned goldens")
    fixture = checked
    preimage = h0_preimage(
        seed=fixture.seed,
        epoch_index=fixture.epoch_index,
        global_optimizer_step=fixture.global_optimizer_step,
        site_ordinal=fixture.site_ordinal,
        site_name_ascii=fixture.site_name_ascii,
        shape=fixture.shape,
    )
    if fixture.preimage_bytes is not None and len(preimage) != fixture.preimage_bytes:
        raise DropoutContractError(f"{fixture.label} preimage length mismatch")
    mask = generate_mask(
        seed=fixture.seed,
        epoch_index=fixture.epoch_index,
        global_optimizer_step=fixture.global_optimizer_step,
        site_ordinal=fixture.site_ordinal,
        site_name_ascii=fixture.site_name_ascii,
        shape=fixture.shape,
    )
    actual = (
        mask.h0.hex(),
        f"{mask.base:016x}",
        mask.first32_keep,
        mask.drop_count,
        mask.packed_sha256,
        mask.expanded_sha256,
    )
    expected = (
        fixture.h0_sha256,
        fixture.base_hex,
        fixture.first32_keep,
        fixture.drop_count,
        fixture.packed_sha256,
        fixture.expanded_sha256,
    )
    if actual != expected:
        raise DropoutContractError(f"{fixture.label} golden mismatch")
    if fixture.packed_hex is not None and mask.packed_keep_bits.hex() != fixture.packed_hex:
        raise DropoutContractError(f"{fixture.label} packed bytes mismatch")
    return mask


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    world_size: int = 1
    gradient_checkpointing: bool = False
    hidden_recomputation: bool = False
    fused_dropout: bool = False
    framework_rng_calls: int = 0
    framework_dropout_calls: int = 0
    resolved_text_nonzero_dropout_sites: int = 0
    eval_schedule_accesses: int = 0

    def __post_init__(self) -> None:
        _require_exact_int(self.world_size, "world_size", minimum=1, maximum=(1 << 32) - 1)
        _require_exact_bool(self.gradient_checkpointing, "gradient_checkpointing")
        _require_exact_bool(self.hidden_recomputation, "hidden_recomputation")
        _require_exact_bool(self.fused_dropout, "fused_dropout")
        _require_exact_int(self.framework_rng_calls, "framework_rng_calls")
        _require_exact_int(self.framework_dropout_calls, "framework_dropout_calls")
        _require_exact_int(
            self.resolved_text_nonzero_dropout_sites,
            "resolved_text_nonzero_dropout_sites",
        )
        _require_exact_int(self.eval_schedule_accesses, "eval_schedule_accesses")


def validate_runtime_policy(policy: RuntimePolicy) -> str:
    if type(policy) is not RuntimePolicy:
        raise TypeError("policy must be exactly RuntimePolicy")
    checked = RuntimePolicy(
        policy.world_size,
        policy.gradient_checkpointing,
        policy.hidden_recomputation,
        policy.fused_dropout,
        policy.framework_rng_calls,
        policy.framework_dropout_calls,
        policy.resolved_text_nonzero_dropout_sites,
        policy.eval_schedule_accesses,
    )
    if checked.world_size != 1:
        raise RuntimePolicyError("world_size must be exactly 1")
    if checked.gradient_checkpointing:
        raise RuntimePolicyError("gradient checkpointing must be off")
    if checked.hidden_recomputation:
        raise RuntimePolicyError("hidden recomputation must be off")
    if checked.fused_dropout:
        raise RuntimePolicyError("fused dropout must be off")
    if checked.framework_rng_calls != 0 or checked.framework_dropout_calls != 0:
        raise RuntimePolicyError("framework RNG/dropout calls must be zero")
    if checked.resolved_text_nonzero_dropout_sites != 0:
        raise RuntimePolicyError("resolved text active dropout requires HOLD")
    if checked.eval_schedule_accesses != 0:
        raise RuntimePolicyError("eval must consume no H0, mask, or counter")
    return "STATIC_RUNTIME_POLICY_VALIDATED_NONPRODUCTION"


def eval_mask() -> None:
    """Eval has no request object and consumes no schedule state."""

    return None


__all__ = [
    "ARCHITECTURES",
    "AXIS_CODES",
    "ByteIdentity",
    "DROP_FLOAT32_LE",
    "DROP_THRESHOLD",
    "DropoutContractError",
    "DropoutMask",
    "DropoutSite",
    "GOLDEN_FIXTURES",
    "GoldenFixture",
    "H0_DOMAIN",
    "INVENTORY_IDENTITIES",
    "InventoryMismatchError",
    "KEEP_FLOAT32_BITS",
    "KEEP_FLOAT32_LE",
    "MIME_SWAP_ORDINAL_1_3_INVENTORY_SHA256",
    "MIME_SWAP_ORDINAL_1_3_TRACE_SHA256",
    "RuntimePolicy",
    "RuntimePolicyError",
    "SITE_INVENTORIES",
    "TRACE_IDENTITIES",
    "application_trace_bytes",
    "canonical_inventory_bytes",
    "canonical_one_step_trace",
    "encode_inventory",
    "eval_mask",
    "generate_mask",
    "generate_site_mask",
    "h0_digest",
    "h0_preimage",
    "iter_keep_bits",
    "keep_from_mixed_word",
    "pack_keep_bits_lsb_first",
    "site_inventory",
    "site_shape",
    "splitmix64_word",
    "validate_canonical_inventory",
    "validate_runtime_policy",
    "verify_golden",
]
