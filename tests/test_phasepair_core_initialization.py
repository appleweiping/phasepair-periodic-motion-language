from __future__ import annotations

from contextlib import ExitStack
import gc
import hashlib
import inspect
import json
import math
import struct
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from phasepair_core import batching, contracts  # noqa: E402
from phasepair_core import initialization as init  # noqa: E402
from phasepair_core import torch_models  # noqa: E402


EXPECTED_LEDGERS = {
    "mime": (244, 35_111_936, init.MIME_EARLY_DOMAIN),
    "early": (69, 13_014_016, init.MIME_EARLY_DOMAIN),
    "late": (142, 26_017_280, init.LATE_DOMAIN),
}

LATE_SEED_1729_GOLDENS = {
    "actor_a.input_proj.weight": "60da6b3dca74adbcee9415bdfb70153d",
    "actor_b.blocks.00.self_attn.q_proj.weight": "d95d1c3dbe0309bdced321bd41801cbc",
    "actor_a.pool.query": "93abd03bbf45b0b9e20f0bbd2ceed43c",
    "fusion.proj.weight": "eeada13aaa597fbc4abdb83bd9f3c5bc",
}


GLOBAL_REBINDS = (
    (init, "_UINT64_MAX", 0),
    (init, "_FLOAT32_BYTES", 8),
    (init, "_CANONICAL_PE_SHA256", "00" * 32),
    (init, "_MODEL_TYPES", {}),
    (init, "_CAPTURED", object()),
    (init, "_LATE_GOLDENS", ()),
    (init, "_RECEIPT_KEYS", frozenset()),
    (init, "AUTHORITY", 99),
    (init, "STATUS", "FORGED_READY"),
    (init, "RECEIPT_SCHEMA", "forged-receipt"),
    (init, "INVENTORY_SCHEMA", "forged-inventory"),
    (init, "MIME_EARLY_DOMAIN", "forged-phasepair-domain"),
    (init, "LATE_DOMAIN", "forged-late-domain"),
    (init, "STATE_DOMAIN", b"forged-state-domain"),
    (init, "InitializationContractError", RuntimeError),
    (init, "hashlib", object()),
    (init, "json", object()),
    (init, "math", object()),
    (init, "struct", object()),
    (init, "sys", object()),
    (init, "torch", object()),
    (init, "Tensor", object()),
    (init, "nn", object()),
    (init, "OrderedDict", object()),
    (init, "_MODULE_BASE_INSTANCE_FIELDS", ()),
    (init, "_MODULE_HOOK_MAP_FIELDS", ()),
    (init, "contracts", object()),
    (init, "batching", object()),
    (init, "torch_models", object()),
    (contracts, "ARCHITECTURES", ()),
    (contracts, "MATRIX_WEIGHT", "FORGED_MATRIX"),
    (contracts, "BIAS", "FORGED_BIAS"),
    (contracts, "LAYERNORM_GAMMA", "FORGED_GAMMA"),
    (contracts, "LAYERNORM_BETA", "FORGED_BETA"),
    (contracts, "EMBEDDING_WEIGHT", "FORGED_EMBEDDING"),
    (contracts, "QUERY_OR_OTHER_WEIGHT", "FORGED_QUERY"),
    (contracts, "SEMANTIC_CLASSES", frozenset()),
    (contracts, "MOTION_PARTITION_CONSTANTS", {}),
    (contracts, "canonical_motion_rows", lambda _architecture: ()),
    (batching, "MAX_PADDED_TIME", 1),
    (torch_models, "VECTOR_WIDTH", 1),
    (torch_models, "MimeMotionEncoder", object),
    (torch_models, "EarlyFusionMotionEncoder", object),
    (torch_models, "LateFusionMotionEncoder", object),
    (torch_models, "_linear_pairs", object()),
    (torch_models, "_attention_pairs", object()),
    (torch_models, "_layer_norm_pairs", object()),
    (torch.nn.Module, "named_parameters", object()),
)

GLOBAL_REBINDS += tuple(
    (init, name, object())
    for name in (
        "_rows_for",
        "_ledger_for",
        "_mime_registered_mapping",
        "_early_registered_mapping",
        "_late_registered_mapping",
        "_checked_registered_mapping",
        "_module_type_for_registered_path",
        "_module_instance_field_schema",
        "_registered_module_specs",
        "_model_spec",
        "_require_seed",
        "_require_architecture",
        "_initializer_domain",
        "_serialization_rows",
        "_canonical_json_bytes",
        "_tensor_raw_bytes",
        "_parameter_interval",
        "_assert_disjoint",
        "_validate_model_impl",
        "_constant_bytes",
        "_counter_bytes",
        "_late_fan_in",
        "_derive_parameter_raw",
        "_canonical_parameter_bytes_impl",
        "_canonical_initializer_inventory_bytes_impl",
        "_update_state_hash_record",
        "_state_sha256_from_raw",
        "_canonical_motion_state_sha256_impl",
        "_snapshot_originals",
        "_prepare_tensors",
        "_normalized_identity_evidence",
        "_late_golden_evidence",
        "_late_counterpart_evidence",
        "_runtime_evidence",
        "_build_receipt",
        "_source_tensor",
        "_transactional_commit",
        "_initialize_motion_model_impl",
        "_is_lower_hex",
        "_reject_json_number",
        "_parse_receipt_static_semantics",
        "_parse_receipt",
        "_verify_motion_initializer_receipt_impl",
        "_captured_contract_snapshot_bytes_impl",
        "_transactional_copy_for_test_impl",
        "_bind_public",
    )
)


def _enter_all_global_rebinds(stack: ExitStack) -> None:
    for target, name, replacement in GLOBAL_REBINDS:
        stack.enter_context(mock.patch.object(target, name, replacement))


def _independent_phasepair_v1_word(
    name: str,
    seed: int,
    index: int,
    bound: float,
    post_scale: float | None = None,
) -> bytes:
    name_raw = name.encode("utf-8")
    digest = hashlib.sha256(
        b"phasepair-init-v1"
        + seed.to_bytes(8, "big")
        + len(name_raw).to_bytes(2, "big")
        + name_raw
        + index.to_bytes(8, "big")
    ).digest()
    counter = int.from_bytes(digest[:8], "big")
    unit = ((counter << 1) + 1) / float(1 << 65)
    initialized = (2.0 * unit - 1.0) * bound
    if post_scale is not None:
        initialized *= post_scale
    return struct.pack("<f", initialized)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().numpy().tobytes(order="C")


def _registered_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(_tensor_bytes(parameter))
    return digest.hexdigest()


def test_exact_ledgers_and_domain_specific_inventory_orders() -> None:
    for architecture, (tensor_count, numel, domain) in EXPECTED_LEDGERS.items():
        rows = contracts.canonical_motion_rows(architecture)
        model = torch_models.build_motion_encoder(architecture, device="meta")
        assert len(rows) == tensor_count
        assert sum(row.numel for row in rows) == numel
        payload = json.loads(init.canonical_initializer_inventory_bytes(architecture))
        assert payload["architecture"] == architecture
        assert payload["initializer_domain"] == domain
        assert payload["schema"] == init.INVENTORY_SCHEMA
        assert all(row["dtype"] == "float32" for row in payload["rows"])
        names = tuple(row["name"] for row in payload["rows"])
        expected_names = tuple(row.name for row in rows)
        if architecture == "late":
            assert names == expected_names
        else:
            assert names == tuple(sorted(expected_names, key=lambda name: name.encode("utf-8")))
        model_spec = next(
            spec for spec in init._CAPTURED.model_specs if spec[1] == architecture
        )
        registered_mapping = model_spec[2]
        observed_registered_names = tuple(
            name for name, _ in model.named_parameters(remove_duplicate=False)
        )
        assert observed_registered_names == tuple(
            registered for registered, _ in registered_mapping
        )
        mapped_names = tuple(canonical for _, canonical in registered_mapping)
        assert len(mapped_names) == len(set(mapped_names)) == tensor_count
        assert set(mapped_names) == set(expected_names)
        rows_by_name = {row.name: row for row in rows}
        observed_pairs = tuple(model.named_parameters(remove_duplicate=False))
        assert len({id(parameter) for _, parameter in observed_pairs}) == tensor_count
        for (registered_name, parameter), mapping_pair in zip(
            observed_pairs, registered_mapping, strict=True
        ):
            expected_registered_name, canonical_name = mapping_pair
            assert registered_name == expected_registered_name
            assert tuple(parameter.shape) == rows_by_name[canonical_name].shape


@pytest.mark.parametrize("architecture", ("mime", "early", "late"))
def test_direct_registry_schema_accepts_train_and_eval_for_every_architecture(
    architecture: str,
) -> None:
    model = torch_models.build_motion_encoder(architecture, device="cpu")
    model.eval()
    assert init._validate_model(model).architecture == architecture
    model.train()
    assert init._validate_model(model).architecture == architecture
    del model
    gc.collect()


@pytest.mark.parametrize("bad_seed", (True, -1, 1 << 64, 1729.0, "1729"))
def test_seed_requires_a_bounded_exact_built_in_int(bad_seed: object) -> None:
    error = TypeError if type(bad_seed) is not int else ValueError
    with pytest.raises(error):
        init.canonical_parameter_bytes(
            "early", "tmr.input.linear.bias", seed=bad_seed  # type: ignore[arg-type]
        )


def test_late_seed_1729_matches_all_four_frozen_raw_goldens() -> None:
    for name, expected in LATE_SEED_1729_GOLDENS.items():
        raw = init.canonical_parameter_bytes("late", name, seed=1729)
        assert raw[:16].hex() == expected


@pytest.mark.parametrize(
    ("architecture", "name", "fan_out", "fan_in", "indices"),
    (
        ("mime", "mime.input.a.linear.weight", 512, 262, (0, 1, 17, 134_143)),
        ("early", "tmr.input.linear.weight", 512, 786, (0, 3, 257, 402_431)),
    ),
)
def test_mime_and_early_xavier_bytes_match_an_independent_oracle(
    architecture: str,
    name: str,
    fan_out: int,
    fan_in: int,
    indices: tuple[int, ...],
) -> None:
    raw = init.canonical_parameter_bytes(architecture, name, seed=2718)
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    for index in indices:
        assert raw[4 * index : 4 * index + 4] == _independent_phasepair_v1_word(
            name, 2718, index, bound
        )


def test_constant_bits_and_seed_separation_follow_the_two_domains() -> None:
    assert set(
        init.canonical_parameter_bytes(
            "mime", "mime.input.a.linear.bias", seed=0
        )
    ) == {0}
    mime_gamma = init.canonical_parameter_bytes(
        "mime", "mime.input.a.ln.weight", seed=(1 << 64) - 1
    )
    assert mime_gamma[:4] == bytes.fromhex("0000803f")
    assert len(set(mime_gamma[index : index + 4] for index in range(0, len(mime_gamma), 4))) == 1

    late_beta = init.canonical_parameter_bytes(
        "late", "actor_a.input_ln.bias", seed=1729
    )
    assert set(late_beta) == {0}
    late_bias_1729_a = init.canonical_parameter_bytes(
        "late", "actor_a.input_proj.bias", seed=1729
    )
    late_bias_1729_b = init.canonical_parameter_bytes(
        "late", "actor_a.input_proj.bias", seed=1729
    )
    late_bias_2718 = init.canonical_parameter_bytes(
        "late", "actor_a.input_proj.bias", seed=2718
    )
    assert late_bias_1729_a == late_bias_1729_b
    assert late_bias_1729_a != late_bias_2718
    assert any(late_bias_1729_a)


@pytest.mark.parametrize(
    ("architecture", "name"),
    (
        ("mime", "mime.input.actor_embedding.a"),
        ("mime", "mime.pool.query"),
        ("early", "tmr.pool.query"),
    ),
)
def test_embedding_and_query_multiply_order_matches_an_independent_oracle(
    architecture: str, name: str
) -> None:
    raw = init.canonical_parameter_bytes(architecture, name, seed=31415)
    for index in (0, 1, 31, 511):
        assert raw[4 * index : 4 * index + 4] == _independent_phasepair_v1_word(
            name,
            31415,
            index,
            math.sqrt(3.0),
            0.02,
        )


def test_complete_early_state_is_reproducible_hashed_and_non_authoritative() -> None:
    model = torch_models.EarlyFusionMotionEncoder(device="cpu")
    position_before = _tensor_bytes(model.position_encoding)
    forbidden = AssertionError("initializer called a torch RNG")
    with (
        mock.patch.object(torch, "rand", side_effect=forbidden),
        mock.patch.object(torch, "randn", side_effect=forbidden),
        mock.patch.object(torch, "rand_like", side_effect=forbidden),
        mock.patch.object(torch, "randn_like", side_effect=forbidden),
        mock.patch.object(torch, "manual_seed", side_effect=forbidden),
        mock.patch.object(torch.Tensor, "uniform_", side_effect=forbidden),
        mock.patch.object(torch.Tensor, "normal_", side_effect=forbidden),
    ):
        first_receipt = init.initialize_motion_model(model, seed=1729)
        with ExitStack() as stack:
            _enter_all_global_rebinds(stack)
            second_receipt = init.initialize_motion_model(model, seed=1729)
            assert (
                init.verify_motion_initializer_receipt(
                    first_receipt, model, seed=1729
                )
                == first_receipt
            )
    assert first_receipt == second_receipt
    parsed = json.loads(first_receipt)
    assert first_receipt == (
        json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert parsed["schema"] == init.RECEIPT_SCHEMA
    assert parsed["status"] == init.STATUS
    assert parsed["authority"] == 0
    assert parsed["production"] is False
    assert parsed["training_authorized"] is False
    assert parsed["architecture"] == "early"
    assert parsed["initializer_domain"] == init.MIME_EARLY_DOMAIN
    assert parsed["parameter_tensor_count"] == 69
    assert parsed["parameter_numel"] == 13_014_016
    assert parsed["parameter_object_count"] == 69
    assert parsed["parameter_storage_range_count"] == 69
    assert parsed["actor_shared_parameter_object_count"] == 0
    assert parsed["actor_shared_storage_range_count"] == 0
    assert parsed["late_nonconstant_counterpart_count"] == 0
    assert parsed["late_nonconstant_counterpart_distinct_sha256_count"] == 0
    assert parsed["late_seed_1729_golden_evidence"] == {
        "applicable": False,
        "expected_first_four_le_hex": {},
        "observed_first_four_le_hex": {},
        "schema": "phasepair-late-seed1729-goldens-v1",
        "status": "NOT_APPLICABLE",
    }
    assert parsed["phasepair_v1_golden_status"] == (
        "HOLD_HASH_SELECTED_16_INDEX_RULE_UNRESOLVED"
    )
    assert parsed["evidence_hold_reasons"] == [
        "HOLD_RUNTIME_LIBM_ORACLE_NOT_CONTENT_PINNED",
        "HOLD_PHASEPAIR_INIT_V1_HASH_SELECTED_16_INDEX_RULE_UNRESOLVED",
    ]
    assert parsed["oracle_runtime"]["libm_oracle_status"] == (
        "UNPINNED_RUNTIME_DEPENDENT_HOLD"
    )
    runtime_bytes = (
        json.dumps(
            parsed["oracle_runtime"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert hashlib.sha256(runtime_bytes).hexdigest() == parsed[
        "oracle_runtime_sha256"
    ]
    assert _tensor_bytes(model.position_encoding) == position_before
    assert hashlib.sha256(position_before).hexdigest() == parsed["position_encoding_sha256"]
    assert init.canonical_motion_state_sha256(model).hex() == parsed["full_state_sha256"]
    assert (
        parsed["full_state_sha256"]
        == "76c313fbe4a3d61ff2a072b9975f2281426f57293fa2f06f025848b936d3d79a"
    )
    assert (
        parsed["inventory_sha256"]
        == "0a40218dad46af61aaf3d99c74b24e2a5b70a6c38e86fe24841e6134c213b051"
    )

    parameters = dict(init._validate_model(model).pairs)
    serialized_names = tuple(item["name"] for item in parsed["parameter_tensors"])
    assert serialized_names == tuple(sorted(parameters, key=lambda name: name.encode("utf-8")))
    for item in parsed["parameter_tensors"]:
        parameter = parameters[item["name"]]
        assert list(parameter.shape) == item["shape"]
        assert hashlib.sha256(_tensor_bytes(parameter)).hexdigest() == item["sha256"]
    assert _tensor_bytes(parameters["tmr.input.linear.bias"])[:4] == b"\x00\x00\x00\x00"
    assert _tensor_bytes(parameters["tmr.input.ln.weight"])[:4] == bytes.fromhex(
        "0000803f"
    )

    forged_cases = []
    for field, value in (
        ("seed", -1),
        ("parameter_tensor_count", 0),
        ("parameter_numel", 0),
        ("inventory_sha256", "0" * 64),
        ("identity_evidence_schema", "forged-identity"),
        ("position_encoding_sha256", "0" * 64),
        ("phasepair_v1_golden_status", "PASS"),
        ("architecture", "late"),
    ):
        candidate = json.loads(first_receipt)
        candidate[field] = value
        forged_cases.append(candidate)
    candidate = json.loads(first_receipt)
    candidate["evidence_hold_reasons"] = []
    forged_cases.append(candidate)
    candidate = json.loads(first_receipt)
    candidate["oracle_runtime"]["python_version"] = "forged"
    forged_cases.append(candidate)
    candidate = json.loads(first_receipt)
    candidate["late_seed_1729_golden_evidence"]["status"] = "PASS"
    forged_cases.append(candidate)
    candidate = json.loads(first_receipt)
    candidate["parameter_tensors"][0]["shape"] = [1]
    forged_cases.append(candidate)
    for candidate in forged_cases:
        forged_receipt = (
            json.dumps(
                candidate,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        with pytest.raises(init.InitializationContractError):
            init._parse_receipt(forged_receipt, model, 1729, init._CAPTURED)

    tampered = dict(parsed)
    tampered["authority"] = 1
    tampered_receipt = (
        json.dumps(
            tampered,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(init.InitializationContractError, match="authority/status"):
        init.verify_motion_initializer_receipt(tampered_receipt, model, seed=1729)
    tampered = json.loads(first_receipt)
    tampered["full_state_sha256"] = "0" * 64
    tampered_receipt = (
        json.dumps(
            tampered,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(init.InitializationContractError, match="cryptographic fields"):
        init.verify_motion_initializer_receipt(tampered_receipt, model, seed=1729)
    with pytest.raises(TypeError, match="exact built-in bytes"):
        init.verify_motion_initializer_receipt(
            bytearray(first_receipt), model, seed=1729  # type: ignore[arg-type]
        )
    del model
    gc.collect()


def test_forged_shared_parameter_fails_before_any_parameter_write() -> None:
    model = torch_models.EarlyFusionMotionEncoder(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    model.input_norm.bias = model.input_norm.weight
    before = _registered_state_digest(model)
    with pytest.raises(init.InitializationContractError, match="share an object"):
        init.initialize_motion_model(model, seed=1729)
    assert _registered_state_digest(model) == before
    del model
    gc.collect()


def test_instance_named_views_cannot_forge_a_qk_swap() -> None:
    model = torch_models.EarlyFusionMotionEncoder(device="cpu")
    root_modules = object.__getattribute__(model, "_modules")
    block_zero = object.__getattribute__(root_modules["blocks"], "_modules")["0"]
    attention = object.__getattribute__(block_zero, "_modules")["attention"]
    attention_modules = object.__getattribute__(attention, "_modules")
    q_parameter = object.__getattribute__(
        attention_modules["q"], "_parameters"
    )["weight"]
    k_parameter = object.__getattribute__(
        attention_modules["k"], "_parameters"
    )["weight"]
    q_name = "blocks.0.attention.q.weight"
    k_name = "blocks.0.attention.k.weight"

    def forged_named_view(*_args: object, **_kwargs: object):
        return iter(((q_name, k_parameter), (k_name, q_parameter)))

    first_parameter = object.__getattribute__(root_modules["input_linear"], "_parameters")[
        "weight"
    ]
    before = _tensor_bytes(first_parameter)
    object.__setattr__(model, "_named_members", forged_named_view)
    object.__setattr__(model, "named_parameters", forged_named_view)
    assert dict(model.named_parameters())[q_name] is k_parameter
    with pytest.raises(
        init.InitializationContractError, match="instance field name/order mismatch"
    ):
        init.initialize_motion_model(model, seed=1729)
    assert _tensor_bytes(first_parameter) == before
    del model
    gc.collect()


def test_every_extra_instance_field_fails_without_inspecting_its_value() -> None:
    class SlotBox:
        __slots__ = ("payload",)

        def __init__(self, payload: object) -> None:
            self.payload = payload

    class DictAndSlotsBox:
        __slots__ = ("payload", "__dict__")

        def __init__(self, payload: object) -> None:
            self.payload = payload
            self.decoy = "present"

    class HostileDict(dict):
        def __iter__(self):
            raise AssertionError("initializer iterated an unregistered dict subclass")

        def keys(self):
            raise AssertionError("initializer called keys on an unregistered value")

        def values(self):
            raise AssertionError("initializer called values on an unregistered value")

        def items(self):
            raise AssertionError("initializer called items on an unregistered value")

    class HostileList(list):
        def __iter__(self):
            raise AssertionError("initializer iterated an unregistered list subclass")

    model = torch_models.EarlyFusionMotionEncoder(device="cpu")
    instance_dict = object.__getattribute__(model, "__dict__")
    root_modules = object.__getattribute__(model, "_modules")
    first_parameter = object.__getattribute__(root_modules["input_linear"], "_parameters")[
        "weight"
    ]
    before = _tensor_bytes(first_parameter)
    hidden_values = (
        ("hidden_scalar", 7),
        ("hidden_tensor", torch.zeros((1,), dtype=torch.float32)),
        (
            "hidden_parameter",
            torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32)),
        ),
        ("hidden_module", torch.nn.Identity()),
        ("hidden_slot_tensor", SlotBox(torch.zeros((1,), dtype=torch.float32))),
        (
            "hidden_slot_parameter",
            SlotBox(torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32))),
        ),
        ("hidden_slot_module", SlotBox(torch.nn.Identity())),
        (
            "hidden_dict_and_slots",
            DictAndSlotsBox(torch.zeros((1,), dtype=torch.float32)),
        ),
        (
            "hidden_hostile_dict",
            HostileDict(payload=torch.zeros((1,), dtype=torch.float32)),
        ),
        (
            "hidden_hostile_list",
            HostileList((torch.zeros((1,), dtype=torch.float32),)),
        ),
    )
    for attribute_name, hidden_value in hidden_values:
        object.__setattr__(model, attribute_name, hidden_value)
        with pytest.raises(
            init.InitializationContractError,
            match="instance field name/order mismatch",
        ):
            init.initialize_motion_model(model, seed=1729)
        assert _tensor_bytes(first_parameter) == before
        del instance_dict[attribute_name]

    model.eval()
    assert init._validate_model(model).architecture == "early"
    model.train()
    assert init._validate_model(model).architecture == "early"
    del model
    gc.collect()


def test_module_base_and_literal_instance_fields_are_exact() -> None:
    model = torch_models.EarlyFusionMotionEncoder(device="cpu")
    root_dict = object.__getattribute__(model, "__dict__")
    root_modules = root_dict["_modules"]
    first_parameter = object.__getattribute__(root_modules["input_linear"], "_parameters")[
        "weight"
    ]
    before = _tensor_bytes(first_parameter)

    mutations = (
        (root_dict, "training", 1, "training flag"),
        (root_dict, "_is_full_backward_hook", False, "backward-hook flag"),
        (root_dict, "_forward_hooks", {}, "hook registry"),
        (
            object.__getattribute__(root_modules["input_linear"], "__dict__"),
            "in_features",
            785,
            "literal instance field",
        ),
        (
            object.__getattribute__(root_modules["input_norm"], "__dict__"),
            "width",
            511,
            "literal instance field",
        ),
        (
            object.__getattribute__(
                object.__getattribute__(root_modules["blocks"], "_modules")["0"],
                "__dict__",
            ),
            "block_index",
            1,
            "literal instance field",
        ),
    )
    for target_dict, field_name, forged_value, message in mutations:
        original = target_dict[field_name]
        target_dict[field_name] = forged_value
        with pytest.raises(init.InitializationContractError, match=message):
            init.initialize_motion_model(model, seed=1729)
        assert _tensor_bytes(first_parameter) == before
        target_dict[field_name] = original

    training = root_dict.pop("training")
    root_dict["training"] = training
    with pytest.raises(
        init.InitializationContractError, match="instance field name/order mismatch"
    ):
        init.initialize_motion_model(model, seed=1729)
    assert _tensor_bytes(first_parameter) == before
    del model
    gc.collect()

    late = torch_models.LateFusionMotionEncoder(device="cpu")
    late_root_modules = object.__getattribute__(late, "_modules")
    actor_dict = object.__getattribute__(late_root_modules["actor_a"], "__dict__")
    late_first = object.__getattribute__(
        object.__getattribute__(late_root_modules["actor_a"], "_modules")[
            "input_projection"
        ],
        "_parameters",
    )["weight"]
    late_before = _tensor_bytes(late_first)
    for field_name, forged_value in (("actor", "actor_b"), ("actor_index", 1)):
        original = actor_dict[field_name]
        actor_dict[field_name] = forged_value
        with pytest.raises(
            init.InitializationContractError, match="literal instance field"
        ):
            init.initialize_motion_model(late, seed=1729)
        assert _tensor_bytes(late_first) == late_before
        actor_dict[field_name] = original
    del late
    gc.collect()


def test_direct_registry_census_rejects_extras_omissions_and_dict_subclasses() -> None:
    class DictSubclass(dict):
        pass

    class NameSubclass(str):
        pass

    model = torch_models.EarlyFusionMotionEncoder(device="cpu")
    root_parameters = object.__getattribute__(model, "_parameters")
    root_buffers = object.__getattribute__(model, "_buffers")
    root_modules = object.__getattribute__(model, "_modules")
    first_parameter = next(model.parameters())
    before = _tensor_bytes(first_parameter)

    root_parameters["hidden"] = torch.nn.Parameter(
        torch.zeros((1,), dtype=torch.float32)
    )
    with pytest.raises(init.InitializationContractError, match="registry name/order"):
        init._validate_model(model)
    del root_parameters["hidden"]

    root_buffers["hidden"] = torch.zeros((1,), dtype=torch.float32)
    with pytest.raises(init.InitializationContractError, match="registry name/order"):
        init._validate_model(model)
    del root_buffers["hidden"]

    root_modules["hidden"] = torch.nn.Identity()
    with pytest.raises(init.InitializationContractError, match="registry name/order"):
        init._validate_model(model)
    del root_modules["hidden"]

    pool = root_parameters.pop("pool_query")
    with pytest.raises(init.InitializationContractError, match="registry name/order"):
        init._validate_model(model)
    root_parameters["pool_query"] = pool

    input_linear = root_modules["input_linear"]
    original_linear_parameters = object.__getattribute__(input_linear, "_parameters")
    object.__setattr__(
        input_linear,
        "_parameters",
        DictSubclass(original_linear_parameters),
    )
    with pytest.raises(init.InitializationContractError, match="exact built-in dict"):
        init._validate_model(model)
    object.__setattr__(input_linear, "_parameters", original_linear_parameters)

    subclassed_names = {
        (NameSubclass(name) if name == "weight" else name): parameter
        for name, parameter in original_linear_parameters.items()
    }
    object.__setattr__(input_linear, "_parameters", subclassed_names)
    with pytest.raises(init.InitializationContractError, match="registry name/order"):
        init._validate_model(model)
    object.__setattr__(input_linear, "_parameters", original_linear_parameters)

    assert _tensor_bytes(first_parameter) == before
    assert init._validate_model(model).architecture == "early"
    del model
    gc.collect()


def test_direct_registry_rejects_module_and_storage_aliases() -> None:
    model = torch_models.EarlyFusionMotionEncoder(device="cpu")
    root_modules = object.__getattribute__(model, "_modules")
    blocks = root_modules["blocks"]
    block_modules = object.__getattribute__(blocks, "_modules")
    original_block_one = block_modules["1"]
    block_modules["1"] = block_modules["0"]
    with pytest.raises(init.InitializationContractError, match="cycle or alias"):
        init._validate_model(model)
    block_modules["1"] = original_block_one

    block_zero = block_modules["0"]
    attention = object.__getattribute__(block_zero, "_modules")["attention"]
    attention_modules = object.__getattribute__(attention, "_modules")
    q_parameters = object.__getattribute__(attention_modules["q"], "_parameters")
    k_parameters = object.__getattribute__(attention_modules["k"], "_parameters")
    original_q = q_parameters["weight"]
    original_k = k_parameters["weight"]
    shared = torch.empty((512, 512), dtype=torch.float32)
    q_parameters["weight"] = torch.nn.Parameter(shared)
    k_parameters["weight"] = torch.nn.Parameter(shared)
    q_decoy = torch.empty((512, 512), dtype=torch.float32)
    k_decoy = torch.empty((512, 512), dtype=torch.float32)
    q_parameters["weight"].untyped_storage = q_decoy.untyped_storage
    k_parameters["weight"].untyped_storage = k_decoy.untyped_storage
    actual_shared_storage = torch.Tensor.untyped_storage(q_parameters["weight"])
    actual_shared_storage.data_ptr = q_decoy.untyped_storage().data_ptr
    actual_shared_storage.nbytes = lambda: q_decoy.numel() * q_decoy.element_size()
    with pytest.raises(init.InitializationContractError, match="storage ranges overlap"):
        init._validate_model(model)
    q_parameters["weight"] = original_q
    k_parameters["weight"] = original_k
    assert init._validate_model(model).architecture == "early"
    del model
    gc.collect()


def test_tensor_and_storage_method_instance_fields_do_not_rebind_validation_or_state() -> None:
    model = torch_models.EarlyFusionMotionEncoder(device="cpu")
    baseline = init.canonical_motion_state_sha256(model)
    parameter = next(model.parameters())
    position_encoding = model.position_encoding

    def bomb(*_: object, **__: object) -> object:
        raise AssertionError("instance-level Tensor/Storage method was dispatched")

    for tensor in (parameter, position_encoding):
        tensor.detach = bomb
        tensor.numpy = bomb
        tensor.contiguous = bomb
        tensor.clone = bomb
        tensor.to = bomb
        tensor.is_contiguous = bomb
        tensor.untyped_storage = bomb
        tensor.storage_offset = bomb
        tensor.numel = bomb
        tensor.element_size = bomb
        storage = torch.Tensor.untyped_storage(tensor)
        storage.data_ptr = bomb
        storage.nbytes = bomb

    assert init._validate_model(model).architecture == "early"
    assert init.canonical_motion_state_sha256(model) == baseline
    del model, parameter, position_encoding, storage
    gc.collect()


def test_shape_correct_zero_position_encoding_fails_before_parameter_write() -> None:
    model = torch_models.EarlyFusionMotionEncoder(device="cpu")
    first_parameter = next(model.parameters())
    before = _tensor_bytes(first_parameter)
    with torch.no_grad():
        model.position_encoding.zero_()
    with pytest.raises(
        init.InitializationContractError,
        match=r"position_encoding bytes.*frozen \[300,512\] oracle",
    ):
        init.initialize_motion_model(model, seed=1729)
    assert _tensor_bytes(first_parameter) == before
    del model
    gc.collect()


def test_exact_model_type_cannot_be_widened_by_global_rebind() -> None:
    class EarlySubclass(torch_models.EarlyFusionMotionEncoder):
        pass

    model = EarlySubclass(device="cpu")
    with mock.patch.object(init, "_MODEL_TYPES", {EarlySubclass: "early"}):
        with pytest.raises(TypeError, match="exact PhasePair motion encoder type"):
            init._validate_model(model)
    del model
    gc.collect()


def test_each_runtime_global_rebind_is_individually_ignored() -> None:
    model = torch_models.EarlyFusionMotionEncoder(device="cpu")
    baseline_contract = init._captured_contract_snapshot_bytes()
    baseline_parameter = init.canonical_parameter_bytes(
        "early", "tmr.pool.query", seed=31415
    )
    baseline_inventory = init.canonical_initializer_inventory_bytes("early")
    baseline_state = init.canonical_motion_state_sha256(model)

    for target, name, replacement in GLOBAL_REBINDS:
        label = f"{getattr(target, '__name__', type(target).__name__)}.{name}"
        with mock.patch.object(target, name, replacement):
            assert init._captured_contract_snapshot_bytes() == baseline_contract, label
            assert (
                init.canonical_parameter_bytes(
                    "early", "tmr.pool.query", seed=31415
                )
                == baseline_parameter
            ), label
            assert (
                init.canonical_initializer_inventory_bytes("early")
                == baseline_inventory
            ), label
            assert init._validate_model(model).architecture == "early", label
            if name == "STATE_DOMAIN":
                assert init.canonical_motion_state_sha256(model) == baseline_state, label
    del model
    gc.collect()


def test_transaction_write_failure_and_silent_noop_roll_back_exactly() -> None:
    parameters = (
        torch.nn.Parameter(torch.full((2,), 0.25, dtype=torch.float32)),
        torch.nn.Parameter(torch.full((2,), -0.5, dtype=torch.float32)),
    )
    originals = tuple(_tensor_bytes(parameter) for parameter in parameters)
    targets = (
        struct.pack("<ff", 1.0, 2.0),
        struct.pack("<ff", 3.0, 4.0),
    )
    original_copy = torch.Tensor.copy_
    calls = 0

    def fail_second(parameter: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("primary-copy-mutant")
        return original_copy(parameter, source)

    with pytest.raises(RuntimeError, match="primary-copy-mutant"):
        init._transactional_copy_for_test(
            parameters, targets, primary_copy=fail_second
        )
    assert calls == 2
    assert tuple(_tensor_bytes(parameter) for parameter in parameters) == originals

    def silent_noop(parameter: torch.Tensor, _source: torch.Tensor) -> torch.Tensor:
        return parameter

    with pytest.raises(init.InitializationContractError, match="post-write byte mismatch"):
        init._transactional_copy_for_test(
            parameters, targets, primary_copy=silent_noop
        )
    assert tuple(_tensor_bytes(parameter) for parameter in parameters) == originals


def test_transaction_cleanup_failure_takes_priority() -> None:
    parameters = (
        torch.nn.Parameter(torch.full((2,), 0.25, dtype=torch.float32)),
        torch.nn.Parameter(torch.full((2,), -0.5, dtype=torch.float32)),
    )
    targets = (
        struct.pack("<ff", 1.0, 2.0),
        struct.pack("<ff", 3.0, 4.0),
    )
    original_copy = torch.Tensor.copy_
    calls = 0

    def fail_second(parameter: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("primary-copy-mutant")
        return original_copy(parameter, source)

    def fail_cleanup(_parameter: torch.Tensor, _source: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("cleanup-copy-mutant")

    with pytest.raises(
        init.InitializationContractError,
        match="cleanup failure takes priority over primary RuntimeError",
    ) as captured:
        init._transactional_copy_for_test(
            parameters,
            targets,
            primary_copy=fail_second,
            rollback_copy=fail_cleanup,
        )
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "cleanup-copy-mutant"


def test_initializer_source_has_no_framework_rng_or_default_initializer_calls() -> None:
    source = inspect.getsource(init)
    forbidden_tokens = (
        "torch.rand(",
        "torch.randn(",
        ".uniform_(",
        ".normal_(",
        "manual_seed(",
        "reset_parameters(",
        "nn.init.",
        "named_parameters(",
        "named_buffers(",
        "state_dict(",
    )
    assert not any(token in source for token in forbidden_tokens)
