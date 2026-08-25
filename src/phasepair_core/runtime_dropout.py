"""One-forward PyTorch adapter for ``phasepair-base-dropout-v1``.

This adapter is deliberately data-free and non-production.  It creates no
schedule, receipt, authority statement, checkpoint, or training result.  A
fresh instance can serve exactly one motion-encoder forward and is terminal
after either a failed ``take`` or a successful ``assert_complete``.

The public uppercase values below are informational.  Runtime decisions are
sealed into the bundle class at module import so later public or private name
rebinding cannot alter or bypass the frozen law.

Ordinary attribute assignment/deletion is rejected.  Deliberate use of
``object.__setattr__`` or mutation of Python closure cells through interpreter
introspection is outside this ordinary-code boundary, just as direct process
memory modification would be.
"""

from __future__ import annotations

import hashlib
import struct
import sys
import threading
from types import MappingProxyType

import torch

from . import dropout


STATUS = "DATA_FREE_NONPRODUCTION_AUTHORITY0"
LAW_ID = "phasepair-base-dropout-v1"
H0_DOMAIN = b"phasepair-base-dropout-v1"
KEEP_FLOAT32_BITS = 0x3F8E38E4
ARCHITECTURES = ("mime", "early", "late")
SITE_COUNTS = MappingProxyType({"mime": 49, "early": 16, "late": 32})


class RuntimeDropoutContractError(ValueError):
    """A one-forward runtime-dropout invariant was violated."""


class RuntimeDropoutBurnedError(RuntimeDropoutContractError):
    """The single-use bundle is terminal and cannot be replayed."""


def _make_bundle_class(
    *,
    original_sha256=hashlib.sha256,
    original_pack=struct.pack,
    tensor_type=torch.Tensor,
    strided_layout=torch.strided,
    float32_dtype=torch.float32,
    uint8_dtype=torch.uint8,
    contiguous_format=torch.contiguous_format,
    tensor_frombuffer=torch.frombuffer,
    tensor_is_contiguous=torch.Tensor.is_contiguous,
    tensor_reshape=torch.Tensor.reshape,
    tensor_clone=torch.Tensor.clone,
    tensor_to=torch.Tensor.to,
    tensor_detach=torch.Tensor.detach,
    tensor_view=torch.Tensor.view,
    tensor_tolist=torch.Tensor.tolist,
    cpu_device=torch.device("cpu"),
    lock_factory=threading.Lock,
    lock_type=type(threading.Lock()),
    lock_acquire=type(threading.Lock()).acquire,
    lock_release=type(threading.Lock()).release,
    object_getattribute=object.__getattribute__,
    object_setattr=object.__setattr__,
    runtime_policy_type=dropout.RuntimePolicy,
    dropout_mask_type=dropout.DropoutMask,
    canonical_generator=dropout.generate_site_mask,
    host_byteorder=sys.byteorder,
    contract_error=RuntimeDropoutContractError,
    burned_error=RuntimeDropoutBurnedError,
):
    """Seal every semantic dependency into closures, then discard this factory."""

    def require_int(
        value: object,
        label: str,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        if type(value) is not int:
            raise TypeError(f"{label} must be an exact built-in int")
        if value < minimum or value > maximum:
            raise ValueError(f"{label} must be in [{minimum},{maximum}]")
        return value

    def literal_sites(architecture: str) -> tuple[tuple[int, str, str], ...]:
        if architecture == "mime":
            suffixes = (
                ("self_a.attn", "attn"),
                ("self_a.proj", "d512"),
                ("self_b.attn", "attn"),
                ("self_b.proj", "d512"),
                ("cross_a_from_b.attn", "attn"),
                ("cross_a_from_b.proj", "d512"),
                ("cross_b_from_a.attn", "attn"),
                ("cross_b_from_a.proj", "d512"),
                ("ffn_a.inner", "f2048"),
                ("ffn_a.ffn", "d512"),
                ("ffn_b.inner", "f2048"),
                ("ffn_b.ffn", "d512"),
            )
            rows = [
                (12 * block + offset, f"blocks.{block:02d}.{suffix}", kind)
                for block in range(4)
                for offset, (suffix, kind) in enumerate(suffixes)
            ]
            rows.append((48, "fusion.inner", "d512"))
            return tuple(rows)
        if architecture == "early":
            suffixes = (
                ("self.attn", "attn"),
                ("self.proj", "d512"),
                ("ffn.inner", "f2048"),
                ("ffn.ffn", "d512"),
            )
            return tuple(
                (4 * block + offset, f"blocks.{block:02d}.{suffix}", kind)
                for block in range(4)
                for offset, (suffix, kind) in enumerate(suffixes)
            )
        if architecture == "late":
            suffixes = (
                ("self.attn", "attn"),
                ("self.proj", "d512"),
                ("ffn.inner", "f2048"),
                ("ffn.ffn", "d512"),
            )
            return tuple(
                (
                    16 * actor_index + 4 * block + offset,
                    f"{actor}.blocks.{block:02d}.{suffix}",
                    kind,
                )
                for actor_index, actor in enumerate(("actor_a", "actor_b"))
                for block in range(4)
                for offset, (suffix, kind) in enumerate(suffixes)
            )
        raise contract_error(f"unknown motion architecture: {architecture!r}")

    inventories = MappingProxyType(
        {
            "mime": literal_sites("mime"),
            "early": literal_sites("early"),
            "late": literal_sites("late"),
        }
    )
    if tuple(len(inventories[name]) for name in ("mime", "early", "late")) != (
        49,
        16,
        32,
    ):
        raise contract_error("private site count drift")
    for architecture_name in ("mime", "early", "late"):
        rows = inventories[architecture_name]
        if tuple(row[0] for row in rows) != tuple(range(len(rows))):
            raise contract_error("private site ordinal drift")

    def literal_shape(
        kind: str, batch_size: int, time_steps: int
    ) -> tuple[int, ...]:
        if kind == "attn":
            return (2, batch_size, 4, time_steps, time_steps)
        if kind == "d512":
            return (2, batch_size, time_steps, 512)
        if kind == "f2048":
            return (2, batch_size, time_steps, 2048)
        raise contract_error("private dropout shape-kind drift")

    def policy_fields(policy: object) -> tuple[object, ...]:
        return (
            policy.world_size,
            policy.gradient_checkpointing,
            policy.hidden_recomputation,
            policy.fused_dropout,
            policy.framework_rng_calls,
            policy.framework_dropout_calls,
            policy.resolved_text_nonzero_dropout_sites,
            policy.eval_schedule_accesses,
        )

    def validate_policy(
        policy: object,
    ) -> tuple[int, bool, bool, bool, int, int, int, int]:
        if type(policy) is not runtime_policy_type:
            raise TypeError("policy must be exactly dropout.RuntimePolicy")
        snapshot = policy_fields(policy)
        world_size = require_int(
            snapshot[0],
            "policy.world_size",
            minimum=1,
            maximum=(1 << 32) - 1,
        )
        for index, label in enumerate(
            (
                "policy.gradient_checkpointing",
                "policy.hidden_recomputation",
                "policy.fused_dropout",
            ),
            start=1,
        ):
            if type(snapshot[index]) is not bool:
                raise TypeError(f"{label} must be an exact built-in bool")
        counters = tuple(
            require_int(snapshot[index], label, minimum=0, maximum=(1 << 64) - 1)
            for index, label in (
                (4, "policy.framework_rng_calls"),
                (5, "policy.framework_dropout_calls"),
                (6, "policy.resolved_text_nonzero_dropout_sites"),
                (7, "policy.eval_schedule_accesses"),
            )
        )
        checked = (
            world_size,
            snapshot[1],
            snapshot[2],
            snapshot[3],
            counters[0],
            counters[1],
            counters[2],
            counters[3],
        )
        if checked != (1, False, False, False, 0, 0, 0, 0):
            raise contract_error(
                "runtime policy must be literal single-process/no-recompute/"
                "no-RNG/no-eval"
            )
        reconstructed = runtime_policy_type(*snapshot)
        if (
            type(reconstructed) is not runtime_policy_type
            or policy_fields(reconstructed) != snapshot
        ):
            raise contract_error("runtime policy exact reconstruction drift")
        return checked

    def expected_mask(
        *,
        seed: int,
        epoch_index: int,
        global_optimizer_step: int,
        site_ordinal: int,
        site_name: str,
        shape: tuple[int, ...],
    ) -> tuple[tuple[int, ...], bytes, int, str, int, bytes, bytes]:
        name = site_name.encode("ascii", "strict")
        preimage = bytearray(b"phasepair-base-dropout-v1")
        preimage.append(0)
        preimage.extend(seed.to_bytes(8, "big"))
        preimage.extend(epoch_index.to_bytes(4, "big"))
        preimage.extend(global_optimizer_step.to_bytes(4, "big"))
        preimage.extend(site_ordinal.to_bytes(2, "big"))
        preimage.extend(len(name).to_bytes(2, "big"))
        preimage.extend(name)
        preimage.append(len(shape))
        for dimension in shape:
            preimage.extend(dimension.to_bytes(4, "big"))
        digest = original_sha256(bytes(preimage)).digest()
        base = int.from_bytes(digest[:8], "big")

        numel = 1
        for dimension in shape:
            numel *= dimension
        packed = bytearray((numel + 7) // 8)
        expanded = bytearray(4 * numel)
        first: list[str] = []
        drops = 0
        keep_bytes = original_pack("<I", 0x3F8E38E4)
        for flat_index in range(numel):
            z = (
                base + 0x9E3779B97F4A7C15 * (flat_index + 1)
            ) & 0xFFFFFFFFFFFFFFFF
            z = (
                (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
            ) & 0xFFFFFFFFFFFFFFFF
            z = (
                (z ^ (z >> 27)) * 0x94D049BB133111EB
            ) & 0xFFFFFFFFFFFFFFFF
            z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
            keep = z >= 0x1999999999999999
            if flat_index < 32:
                first.append("1" if keep else "0")
            byte_offset = 4 * flat_index
            if keep:
                packed[flat_index // 8] |= 1 << (flat_index % 8)
                expanded[byte_offset : byte_offset + 4] = keep_bytes
            else:
                drops += 1
                expanded[byte_offset : byte_offset + 4] = b"\x00\x00\x00\x00"
        return (
            shape,
            digest,
            base,
            "".join(first),
            drops,
            bytes(packed),
            bytes(expanded),
        )

    def validate_generated(
        generated: object,
        expected: tuple[tuple[int, ...], bytes, int, str, int, bytes, bytes],
    ) -> bytes:
        if type(generated) is not dropout_mask_type:
            raise TypeError("dropout generator must return exactly DropoutMask")
        generated_shape = generated.shape
        if type(generated_shape) is not tuple:
            raise TypeError("generated dropout shape must be an exact tuple")
        for index, dimension in enumerate(generated_shape):
            if type(dimension) is not int:
                raise TypeError(
                    f"generated dropout shape[{index}] must be an exact int"
                )
        actual = (
            generated_shape,
            generated.h0,
            generated.base,
            generated.first32_keep,
            generated.drop_count,
            generated.packed_keep_bits,
            generated.expanded_float32_le,
        )
        if (
            type(actual[1]) is not bytes
            or type(actual[2]) is not int
            or type(actual[3]) is not str
            or type(actual[4]) is not int
            or type(actual[5]) is not bytes
            or type(actual[6]) is not bytes
        ):
            raise TypeError("generated dropout fields have noncanonical base types")
        if actual != expected:
            raise contract_error(
                "generated mask disagrees with literal H0/SplitMix64/raw-bit law"
            )
        return expected[6]

    def acquire_transition_lock(instance: object):
        transition_lock = object_getattribute(instance, "_transition_lock")
        if type(transition_lock) is not lock_type:
            raise TypeError("bundle transition lock identity/type drift")
        lock_acquire(transition_lock)
        return transition_lock

    def begin_take_locked(instance: object) -> tuple[int, str, str]:
        lifecycle = object_getattribute(instance, "_lifecycle")
        site_count = object_getattribute(instance, "_site_count")
        if lifecycle % 2 != 0 or lifecycle >= 2 * site_count:
            raise burned_error("dropout bundle is terminal")
        expected_ordinal = lifecycle // 2
        ordinals = object_getattribute(instance, "_ordinals")
        if ordinals[expected_ordinal] != expected_ordinal:
            object_setattr(instance, "_lifecycle", lifecycle + 1)
            raise contract_error("per-instance site ordinal drift")
        sites = object_getattribute(instance, "_sites")
        site = sites[expected_ordinal]
        object_setattr(instance, "_lifecycle", lifecycle + 1)
        return site

    class OneForwardCanonicalDropoutBundle:
        """Single-use implementation of the motion model's dropout protocol."""

        __slots__ = (
            "_architecture",
            "_seed",
            "_epoch_index",
            "_global_optimizer_step",
            "_batch_size",
            "_time_steps",
            "_policy",
            "_sites",
            "_site_count",
            "_ordinals",
            "_lifecycle",
            "_device",
            "_transition_lock",
        )

        def __init__(
            self,
            *,
            architecture: str,
            seed: int,
            epoch_index: int,
            global_optimizer_step: int,
            batch_size: int,
            time_steps: int,
            policy: dropout.RuntimePolicy,
        ) -> None:
            if type(architecture) is not str:
                raise TypeError("architecture must be an exact built-in str")
            if architecture == "mime":
                source_sites = inventories["mime"]
                literal_count = 49
            elif architecture == "early":
                source_sites = inventories["early"]
                literal_count = 16
            elif architecture == "late":
                source_sites = inventories["late"]
                literal_count = 32
            else:
                raise contract_error(
                    f"unknown motion architecture: {architecture!r}"
                )
            sites = tuple((row[0], row[1], row[2]) for row in source_sites)
            site_count = len(sites)
            ordinals = tuple(row[0] for row in sites)
            if site_count != literal_count or ordinals != tuple(range(literal_count)):
                raise contract_error("per-instance site count/ordinal reconstruction drift")
            for index, row in enumerate(sites):
                if (
                    type(row) is not tuple
                    or len(row) != 3
                    or type(row[0]) is not int
                    or type(row[1]) is not str
                    or type(row[2]) is not str
                ):
                    raise TypeError(f"per-instance site row {index} has mutable/bad types")
            seed = require_int(
                seed, "seed", minimum=0, maximum=0xFFFFFFFFFFFFFFFF
            )
            epoch_index = require_int(
                epoch_index, "epoch_index", minimum=0, maximum=29
            )
            global_optimizer_step = require_int(
                global_optimizer_step,
                "global_optimizer_step",
                minimum=0,
                maximum=0xFFFFFFFF,
            )
            batch_size = require_int(
                batch_size, "batch_size", minimum=1, maximum=128
            )
            time_steps = require_int(
                time_steps, "time_steps", minimum=1, maximum=300
            )
            policy_snapshot = validate_policy(policy)
            if host_byteorder != "little":
                raise contract_error(
                    "runtime adapter requires a little-endian host for raw float32 bytes"
                )

            transition_lock = lock_factory()
            if type(transition_lock) is not lock_type:
                raise TypeError("captured lock factory returned a noncanonical lock")
            object_setattr(self, "_architecture", architecture)
            object_setattr(self, "_seed", seed)
            object_setattr(self, "_epoch_index", epoch_index)
            object_setattr(self, "_global_optimizer_step", global_optimizer_step)
            object_setattr(self, "_batch_size", batch_size)
            object_setattr(self, "_time_steps", time_steps)
            object_setattr(self, "_policy", policy_snapshot)
            object_setattr(self, "_sites", sites)
            object_setattr(self, "_site_count", site_count)
            object_setattr(self, "_ordinals", ordinals)
            object_setattr(self, "_lifecycle", 0)
            object_setattr(self, "_device", None)
            object_setattr(self, "_transition_lock", transition_lock)

        def __setattr__(self, name: str, value: object) -> None:
            raise TypeError("runtime dropout bundles are ordinary-immutable")

        def __delattr__(self, name: str) -> None:
            raise TypeError("runtime dropout bundles are ordinary-immutable")

        @property
        def architecture(self) -> str:
            return object_getattribute(self, "_architecture")

        @property
        def batch_size(self) -> int:
            return object_getattribute(self, "_batch_size")

        @property
        def time_steps(self) -> int:
            return object_getattribute(self, "_time_steps")

        def take(self, site_ordinal: int, *, like: torch.Tensor) -> torch.Tensor:
            """Burn, validate, derive, and return one private canonical mask."""

            transition_lock = acquire_transition_lock(self)
            try:
                expected_ordinal, site_name, kind = begin_take_locked(self)
                architecture = object_getattribute(self, "_architecture")
                seed = object_getattribute(self, "_seed")
                epoch_index = object_getattribute(self, "_epoch_index")
                global_optimizer_step = object_getattribute(
                    self, "_global_optimizer_step"
                )
                batch_size = object_getattribute(self, "_batch_size")
                time_steps = object_getattribute(self, "_time_steps")
                bound_device = object_getattribute(self, "_device")
                if type(site_ordinal) is not int:
                    raise TypeError("site_ordinal must be an exact built-in int")
                if site_ordinal != expected_ordinal:
                    raise contract_error(
                        "dropout sites must be requested in exact canonical order"
                    )
                if type(like) is not tensor_type:
                    raise TypeError("like must be an exact base torch.Tensor")
                if like.layout is not strided_layout:
                    raise contract_error("like must have strided layout")
                if like.dtype is not float32_dtype:
                    raise contract_error("like must be float32")
                if like.device.type not in ("cpu", "cuda"):
                    raise contract_error("like device must be CPU or CUDA")
                if not tensor_is_contiguous(like, memory_format=contiguous_format):
                    raise contract_error("like must be contiguous")
                expected_shape = literal_shape(kind, batch_size, time_steps)
                if tuple(like.shape) != expected_shape:
                    raise contract_error(
                        "like shape is not canonical for this site"
                    )
                target_device = like.device
                if bound_device is not None and target_device != bound_device:
                    raise contract_error(
                        "all sites in one forward must use the same exact device"
                    )
                if type(like.requires_grad) is not bool:
                    raise TypeError("like.requires_grad must be an exact bool")

                expected = expected_mask(
                    seed=seed,
                    epoch_index=epoch_index,
                    global_optimizer_step=global_optimizer_step,
                    site_ordinal=expected_ordinal,
                    site_name=site_name,
                    shape=expected_shape,
                )
                generated = canonical_generator(
                    architecture,
                    seed=seed,
                    epoch_index=epoch_index,
                    global_optimizer_step=global_optimizer_step,
                    site_ordinal=expected_ordinal,
                    batch_size=batch_size,
                    time_steps=time_steps,
                )
                expanded = validate_generated(generated, expected)

                host_bytes = bytearray(expanded)
                host_view = tensor_frombuffer(host_bytes, dtype=float32_dtype)
                host_reshaped = tensor_reshape(host_view, expected_shape)
                host_snapshot = tensor_clone(
                    host_reshaped, memory_format=contiguous_format
                )
                if target_device.type == "cuda":
                    output = tensor_to(host_snapshot, device=target_device)
                else:
                    output = host_snapshot
                output = tensor_clone(
                    tensor_detach(output), memory_format=contiguous_format
                )
                if (
                    type(output) is not tensor_type
                    or output.dtype is not float32_dtype
                    or output.device != target_device
                    or tuple(output.shape) != expected_shape
                    or not tensor_is_contiguous(
                        output, memory_format=contiguous_format
                    )
                    or output.requires_grad
                ):
                    raise contract_error("private mask tensor snapshot drift")
                verification = (
                    output
                    if target_device.type == "cpu"
                    else tensor_to(output, device=cpu_device)
                )
                raw_bytes_view = tensor_view(verification, uint8_dtype)
                flat_raw_bytes = tensor_reshape(raw_bytes_view, (-1,))
                actual_raw_bytes = bytes(tensor_tolist(flat_raw_bytes))
                if actual_raw_bytes != expected[6]:
                    raise contract_error(
                        "materialized mask bytes differ from the exact canonical mask"
                    )

                if bound_device is None:
                    object_setattr(self, "_device", target_device)
                object_setattr(
                    self, "_lifecycle", 2 * (expected_ordinal + 1)
                )
                return output
            finally:
                lock_release(transition_lock)

        def assert_complete(self) -> None:
            """Succeed exactly once, and only after every canonical site."""

            transition_lock = acquire_transition_lock(self)
            try:
                lifecycle = object_getattribute(self, "_lifecycle")
                site_count = object_getattribute(self, "_site_count")
                completion_point = 2 * site_count
                if lifecycle == completion_point:
                    object_setattr(self, "_lifecycle", completion_point + 1)
                    return None
                if lifecycle < completion_point and lifecycle % 2 == 0:
                    object_setattr(self, "_lifecycle", completion_point + 2)
                    raise contract_error(
                        "assert_complete requires every canonical site exactly once"
                    )
                raise burned_error(
                    "failed or completed dropout bundle is terminal"
                )
            finally:
                lock_release(transition_lock)

    return OneForwardCanonicalDropoutBundle


OneForwardCanonicalDropoutBundle = _make_bundle_class()
del _make_bundle_class


__all__ = [
    "ARCHITECTURES",
    "H0_DOMAIN",
    "KEEP_FLOAT32_BITS",
    "LAW_ID",
    "OneForwardCanonicalDropoutBundle",
    "RuntimeDropoutBurnedError",
    "RuntimeDropoutContractError",
    "SITE_COUNTS",
    "STATUS",
]
