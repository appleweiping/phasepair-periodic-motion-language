"""DATA_FREE_NONPRODUCTION tests for the PhasePair core static contracts."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import inspect
import struct
import unittest

from phasepair_core import contracts
from phasepair_core import dropout


SOURCE_HASH = bytes.fromhex("11" * 32)


def resolved_rows() -> tuple[contracts.ParameterRow, ...]:
    specs = (
        (
            "ln_final.bias",
            (4,),
            contracts.LAYERNORM_BETA,
        ),
        (
            "ln_final.weight",
            (4,),
            contracts.LAYERNORM_GAMMA,
        ),
        (
            "positional_embedding",
            (77, 4),
            contracts.EMBEDDING_WEIGHT,
        ),
        (
            "token_embedding.weight",
            (49_408, 4),
            contracts.EMBEDDING_WEIGHT,
        ),
        (
            "transformer.resblocks.0.attn.in_proj_bias",
            (12,),
            contracts.BIAS,
        ),
        (
            "transformer.resblocks.0.attn.in_proj_weight",
            (12, 4),
            contracts.MATRIX_WEIGHT,
        ),
    )
    rows = tuple(
        contracts.ParameterRow(
            f"text.clip.{state_key}",
            shape,
            semantic_class,
            contracts.CLIP_TEXT_COMPONENT,
            True,
            state_key,
            SOURCE_HASH,
        )
        for state_key, shape, semantic_class in specs
    )
    return tuple(sorted(rows, key=lambda row: row.name.encode("utf-8")))


def frozen_rows() -> tuple[contracts.ParameterRow, ...]:
    return (
        contracts.ParameterRow(
            "text.pretrained_projection.weight",
            (4, 512),
            contracts.FROZEN,
            contracts.PRETRAINED_PROJECTION_COMPONENT,
            False,
            None,
            SOURCE_HASH,
        ),
    )


def partition(architecture: str = "late") -> contracts.OptimizerPartition:
    resolved = resolved_rows()
    full = contracts.expected_full_trainable_rows(architecture, resolved, 4)
    return contracts.validate_optimizer_partition(
        architecture,
        full_trainable_rows=full,
        resolved_text_rows=resolved,
        hidden_size=4,
        frozen_rows=frozen_rows(),
    )


class StrictPrimitiveTests(unittest.TestCase):
    def test_parameter_row_rejects_non_builtin_name_shape_int_bool(self) -> None:
        with self.assertRaises(TypeError):
            contracts.ParameterRow(  # type: ignore[arg-type]
                b"x", (1,), contracts.MATRIX_WEIGHT, contracts.MOTION_COMPONENT, True
            )
        with self.assertRaises(TypeError):
            contracts.ParameterRow(  # type: ignore[arg-type]
                "x", [1], contracts.MATRIX_WEIGHT, contracts.MOTION_COMPONENT, True
            )
        with self.assertRaises(TypeError):
            contracts.ParameterRow(
                "x", (True,), contracts.MATRIX_WEIGHT, contracts.MOTION_COMPONENT, True
            )
        with self.assertRaises(TypeError):
            contracts.ParameterRow(  # type: ignore[arg-type]
                "x", (1,), contracts.MATRIX_WEIGHT, contracts.MOTION_COMPONENT, 1
            )
        with self.assertRaises(ValueError):
            contracts.ParameterRow(
                "bad\x00name",
                (1,),
                contracts.MATRIX_WEIGHT,
                contracts.MOTION_COMPONENT,
                True,
            )

    def test_sha_requires_exact_builtin_bytes(self) -> None:
        with self.assertRaises(TypeError):
            contracts.require_raw_sha256("00" * 32, "digest")
        with self.assertRaises(ValueError):
            contracts.require_raw_sha256(b"\x00" * 31, "digest")
        self.assertEqual(
            contracts.require_raw_sha256(b"\x00" * 32, "digest"), b"\x00" * 32
        )


class MotionOptimizerTests(unittest.TestCase):
    def test_all_motion_ledgers_derive_from_exact_rows(self) -> None:
        derived = contracts.audit_motion_partition_constants()
        expected = {
            "mime": ((89, 35_020_288), (155, 91_648), (244, 35_111_936)),
            "early": ((26, 12_985_856), (43, 28_160), (69, 13_014_016)),
            "late": ((53, 25_959_424), (89, 57_856), (142, 26_017_280)),
        }
        for architecture, triples in expected.items():
            ledger = derived[architecture]
            self.assertEqual(
                (
                    (ledger.decay.tensor_count, ledger.decay.numel),
                    (ledger.no_decay.tensor_count, ledger.no_decay.numel),
                    (ledger.total.tensor_count, ledger.total.numel),
                ),
                triples,
            )
            self.assertEqual(
                contracts.validate_motion_inventory(
                    architecture, contracts.canonical_motion_rows(architecture)
                ),
                ledger,
            )

    def test_motion_inventory_rejects_missing_unknown_duplicate_and_reorder(self) -> None:
        rows = contracts.canonical_motion_rows("late")
        with self.assertRaises(contracts.MissingNameError):
            contracts.validate_motion_inventory("late", rows[:-1])
        unknown = contracts.ParameterRow(
            "unknown.weight",
            (1,),
            contracts.MATRIX_WEIGHT,
            contracts.MOTION_COMPONENT,
            True,
        )
        with self.assertRaises(contracts.UnknownNameError):
            contracts.validate_motion_inventory("late", rows + (unknown,))
        with self.assertRaises(contracts.DuplicateNameError):
            contracts.validate_motion_inventory("late", rows + (rows[0],))
        with self.assertRaises(contracts.OrderingError):
            contracts.validate_motion_inventory("late", tuple(sorted(rows, key=lambda r: r.name)))

    def test_missing_resolved_inventory_is_explicit_hold(self) -> None:
        for missing in (None, ()):
            with self.subTest(missing=missing):
                with self.assertRaises(contracts.HoldResolvedTextInventory) as caught:
                    contracts.validate_optimizer_partition(
                        "late",
                        full_trainable_rows=(),
                        resolved_text_rows=missing,
                        hidden_size=4,
                        frozen_rows=(),
                    )
                self.assertEqual(caught.exception.status, contracts.HOLD_STATUS)

    def test_exact_two_groups_full_union_and_accounting_for_all_models(self) -> None:
        resolved = resolved_rows()
        resolved_decay = sum(
            row.numel for row in resolved if row.semantic_class in contracts.DECAY_CLASSES
        )
        resolved_no_decay = sum(
            row.numel
            for row in resolved
            if row.semantic_class in contracts.NO_DECAY_CLASSES
        )
        resolved_decay_count = sum(
            row.semantic_class in contracts.DECAY_CLASSES for row in resolved
        )
        resolved_no_decay_count = sum(
            row.semantic_class in contracts.NO_DECAY_CLASSES for row in resolved
        )
        for architecture in contracts.ARCHITECTURES:
            with self.subTest(architecture=architecture):
                result = partition(architecture)
                self.assertEqual(result.status, contracts.STATIC_STATUS)
                self.assertEqual(
                    tuple(group.name for group in result.groups),
                    contracts.OPTIMIZER_GROUP_ORDER,
                )
                self.assertEqual(
                    tuple(group.weight_decay for group in result.groups), ("1e-4", "0")
                )
                full_names = {row.name for row in result.full_trainable_rows}
                decay_names = set(result.groups[0].parameter_names)
                no_decay_names = set(result.groups[1].parameter_names)
                self.assertEqual(decay_names | no_decay_names, full_names)
                self.assertFalse(decay_names & no_decay_names)
                self.assertEqual(
                    result.groups[0].parameter_names,
                    tuple(
                        sorted(result.groups[0].parameter_names, key=lambda n: n.encode("utf-8"))
                    ),
                )
                motion = contracts.MOTION_PARTITION_CONSTANTS[architecture]
                self.assertEqual(
                    (result.groups[0].tensor_count, result.groups[0].numel),
                    (
                        motion.decay.tensor_count + resolved_decay_count + 1,
                        motion.decay.numel + resolved_decay + 512 * 4,
                    ),
                )
                self.assertEqual(
                    (result.groups[1].tensor_count, result.groups[1].numel),
                    (
                        motion.no_decay.tensor_count + resolved_no_decay_count + 2,
                        motion.no_decay.numel + resolved_no_decay + 513,
                    ),
                )
                self.assertIn("actor_a.pool.query" if architecture == "late" else (
                    "mime.pool.query" if architecture == "mime" else "tmr.pool.query"
                ), decay_names)
                self.assertIn("text.logit_scale", no_decay_names)

    def test_full_inventory_rejects_missing_unknown_duplicate_mismatch_and_frozen(self) -> None:
        resolved = resolved_rows()
        full = contracts.expected_full_trainable_rows("late", resolved, 4)
        kwargs = dict(
            architecture="late",
            resolved_text_rows=resolved,
            hidden_size=4,
            frozen_rows=frozen_rows(),
        )
        with self.assertRaises(contracts.MissingNameError):
            contracts.validate_optimizer_partition(full_trainable_rows=full[:-1], **kwargs)
        unknown = contracts.ParameterRow(
            "zzz.unknown.weight",
            (1,),
            contracts.MATRIX_WEIGHT,
            contracts.MOTION_COMPONENT,
            True,
        )
        with self.assertRaises(contracts.UnknownNameError):
            contracts.validate_optimizer_partition(
                full_trainable_rows=tuple(
                    sorted(full + (unknown,), key=lambda row: row.name.encode("utf-8"))
                ),
                **kwargs,
            )
        with self.assertRaises(contracts.DuplicateNameError):
            contracts.validate_optimizer_partition(
                full_trainable_rows=full + (full[-1],), **kwargs
            )
        changed = list(full)
        index = next(i for i, row in enumerate(changed) if row.name.endswith("input_proj.weight"))
        changed[index] = replace(changed[index], semantic_class=contracts.BIAS)
        with self.assertRaises(contracts.RowMismatchError):
            contracts.validate_optimizer_partition(
                full_trainable_rows=tuple(changed), **kwargs
            )
        leaked = frozen_rows()[0]
        with self.assertRaises(contracts.FrozenLeakError):
            contracts.validate_optimizer_partition(
                full_trainable_rows=tuple(
                    sorted(full + (leaked,), key=lambda row: row.name.encode("utf-8"))
                ),
                **kwargs,
            )

    def test_resolution_and_frozen_receipts_fail_closed(self) -> None:
        resolved = resolved_rows()
        with self.assertRaises(contracts.OrderingError):
            contracts.validate_resolved_text_rows(tuple(reversed(resolved)), 4)
        with self.assertRaises(contracts.DuplicateNameError):
            contracts.validate_resolved_text_rows(resolved + (resolved[0],), 4)
        no_hash = replace(resolved[0], source_checkpoint_sha256=None)
        mutated = tuple(sorted((no_hash,) + resolved[1:], key=lambda row: row.name.encode()))
        with self.assertRaises(contracts.HoldResolvedTextInventory):
            contracts.validate_resolved_text_rows(mutated, 4)
        with self.assertRaises(contracts.MissingNameError):
            contracts.validate_frozen_rows(())
        bad_frozen = replace(frozen_rows()[0], name="other.frozen")
        with self.assertRaises(contracts.UnknownNameError):
            contracts.validate_frozen_rows((bad_frozen,))

    def test_group_mutants_are_rejected(self) -> None:
        result = partition()
        contracts.validate_optimizer_groups(result, result.groups)
        with self.assertRaises(contracts.RowMismatchError):
            contracts.validate_optimizer_groups(result, tuple(reversed(result.groups)))
        with self.assertRaises(contracts.RowMismatchError):
            replace(result.groups[0], weight_decay="0")

    def test_public_partition_and_receipt_rederive_every_summary(self) -> None:
        result = partition()
        evil = contracts.ParameterRow(
            "evil",
            (1,),
            contracts.BIAS,
            contracts.MOTION_COMPONENT,
            True,
        )
        forged_groups = (
            contracts.OptimizerGroup(0, "decay", "1e-4", (), 0, 0),
            contracts.OptimizerGroup(1, "no_decay", "0", ("evil",), 1, 1),
        )
        with self.assertRaises(contracts.PhasePairContractError):
            contracts.OptimizerPartition(
                "late", contracts.STATIC_STATUS, (evil,), forged_groups
            )

        # Even an object assembled without the dataclass constructor is not
        # trusted by either public consumer.
        forged = object.__new__(contracts.OptimizerPartition)
        object.__setattr__(forged, "architecture", "late")
        object.__setattr__(forged, "status", contracts.STATIC_STATUS)
        object.__setattr__(forged, "full_trainable_rows", (evil,))
        object.__setattr__(forged, "groups", forged_groups)
        with self.assertRaises(contracts.PhasePairContractError):
            contracts.validate_optimizer_groups(forged, forged_groups)
        with self.assertRaises(contracts.PhasePairContractError):
            contracts.OptimizerReceiptDraft.from_partition(forged)

        self.assertEqual(
            contracts.OptimizerReceiptDraft.from_partition(result).static_status,
            contracts.STATIC_STATUS,
        )

    def test_object_forged_equal_leaves_and_direct_receipt_are_rejected(self) -> None:
        result = partition()
        forged_group = object.__new__(contracts.OptimizerGroup)
        object.__setattr__(forged_group, "ordinal", False)
        object.__setattr__(forged_group, "name", "decay")
        object.__setattr__(forged_group, "weight_decay", "1e-4")
        object.__setattr__(
            forged_group, "parameter_names", result.groups[0].parameter_names
        )
        object.__setattr__(forged_group, "tensor_count", float(result.groups[0].tensor_count))
        object.__setattr__(forged_group, "numel", float(result.groups[0].numel))
        with self.assertRaises(TypeError):
            contracts.OptimizerPartition(
                result.architecture,
                result.status,
                result.full_trainable_rows,
                (forged_group, result.groups[1]),
            )
        with self.assertRaises(TypeError):
            contracts.validate_optimizer_groups(
                result, (forged_group, result.groups[1])
            )

        rows = list(result.full_trainable_rows)
        index = next(i for i, row in enumerate(rows) if row.name == "text.logit_scale")
        canonical = rows[index]
        forged_row = object.__new__(contracts.ParameterRow)
        object.__setattr__(forged_row, "name", canonical.name)
        object.__setattr__(forged_row, "shape", (True,))
        object.__setattr__(forged_row, "semantic_class", canonical.semantic_class)
        object.__setattr__(forged_row, "component", canonical.component)
        object.__setattr__(forged_row, "requires_grad", 1)
        object.__setattr__(forged_row, "resolved_state_key", None)
        object.__setattr__(forged_row, "source_checkpoint_sha256", None)
        rows[index] = forged_row
        with self.assertRaises(TypeError):
            contracts.OptimizerPartition(
                result.architecture, result.status, tuple(rows), result.groups
            )

        with self.assertRaises(TypeError):
            contracts.OptimizerReceiptDraft(  # type: ignore[call-arg]
                "late",
                contracts.STATIC_STATUS,
                b"A" * 32,
                b"B" * 32,
                b"C" * 32,
            )

    def test_receipt_draft_never_fabricates_runtime_hashes(self) -> None:
        draft = contracts.OptimizerReceiptDraft.from_partition(partition())
        self.assertEqual(draft.runtime.capture_status, "RUNTIME_EVIDENCE_NOT_CAPTURED")
        self.assertIsNone(draft.runtime.optimizer_source_sha256)
        self.assertIsNone(draft.runtime.runtime_sha256)
        self.assertIsNone(draft.runtime.step0_state_sha256)
        self.assertIsNone(draft.runtime.first_step_state_sha256)
        with self.assertRaises(TypeError):
            contracts.OptimizerRuntimeEvidence(runtime_sha256="00" * 32)  # type: ignore[arg-type]


class DropoutInventoryTests(unittest.TestCase):
    def test_inventory_counts_bytes_and_hashes(self) -> None:
        expected_counts = {"mime": 49, "early": 16, "late": 32}
        for architecture, count in expected_counts.items():
            with self.subTest(architecture=architecture):
                sites = dropout.site_inventory(architecture)
                self.assertEqual(len(sites), count)
                self.assertEqual(tuple(site.ordinal for site in sites), tuple(range(count)))
                encoded = dropout.canonical_inventory_bytes(architecture)
                identity = dropout.INVENTORY_IDENTITIES[architecture]
                self.assertEqual(len(encoded), identity.byte_count)
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), identity.sha256)

    def test_one_step_trace_bytes_and_hashes(self) -> None:
        for architecture in dropout.ARCHITECTURES:
            with self.subTest(architecture=architecture):
                trace = dropout.canonical_one_step_trace(architecture)
                identity = dropout.TRACE_IDENTITIES[architecture]
                self.assertEqual(len(trace), identity.byte_count)
                self.assertEqual(hashlib.sha256(trace).hexdigest(), identity.sha256)

    def test_zero_step_eval_trace_contains_no_site_access(self) -> None:
        trace = dropout.application_trace_bytes(dropout.site_inventory("mime"), ())
        self.assertEqual(trace[: len(dropout.TRACE_DOMAIN) + 1], dropout.TRACE_DOMAIN + b"\x00")
        self.assertEqual(trace[-6:], b"\x00\x00\x00\x00\x001")
        self.assertEqual(len(trace), len(dropout.TRACE_DOMAIN) + 1 + 4 + 2)

    def test_site_reorder_and_utf8_lexical_sort_are_rejected(self) -> None:
        canonical = dropout.site_inventory("mime")
        reordered = list(canonical)
        reordered[1], reordered[3] = reordered[3], reordered[1]
        with self.assertRaises(dropout.InventoryMismatchError):
            dropout.validate_canonical_inventory("mime", tuple(reordered))
        lexical = tuple(sorted(canonical, key=lambda site: site.name.encode("utf-8")))
        self.assertNotEqual(lexical, canonical)
        with self.assertRaises(dropout.InventoryMismatchError):
            dropout.validate_canonical_inventory("mime", lexical)

    def test_pinned_ordinal_1_3_mutant_hashes(self) -> None:
        canonical = dropout.site_inventory("mime")
        payload_swap = list(canonical)
        payload_swap[1] = replace(
            canonical[1], name=canonical[3].name, axes=canonical[3].axes
        )
        payload_swap[3] = replace(
            canonical[3], name=canonical[1].name, axes=canonical[1].axes
        )
        self.assertEqual(
            hashlib.sha256(dropout.encode_inventory(tuple(payload_swap))).hexdigest(),
            dropout.MIME_SWAP_ORDINAL_1_3_INVENTORY_SHA256,
        )
        order_swap = list(canonical)
        order_swap[1], order_swap[3] = order_swap[3], order_swap[1]
        self.assertEqual(
            hashlib.sha256(
                dropout.application_trace_bytes(tuple(order_swap), (0,))
            ).hexdigest(),
            dropout.MIME_SWAP_ORDINAL_1_3_TRACE_SHA256,
        )


class DropoutGoldenTests(unittest.TestCase):
    def test_all_small_batch128_and_nonzero_goldens(self) -> None:
        self.assertEqual(len(dropout.GOLDEN_FIXTURES), 7)
        for fixture in dropout.GOLDEN_FIXTURES:
            with self.subTest(fixture=fixture.label):
                mask = dropout.verify_golden(fixture)
                self.assertEqual(len(mask.first32_keep), 32)
                self.assertEqual(mask.numel, mask.drop_count + (
                    mask.numel - mask.drop_count
                ))
        mime_small = dropout.verify_golden(dropout.GOLDEN_FIXTURES[0])
        self.assertEqual(len(mime_small.packed_keep_bits), 18)
        self.assertEqual(len(mime_small.expanded_float32_le), 576)
        self.assertEqual(
            mime_small.packed_keep_bits.hex(),
            "bffbfef7bf75fff7fffefffd7dfff7fdf3bf",
        )

    def test_b128_exact_site_recomputes_pinned_h0(self) -> None:
        fixture = dropout.GOLDEN_FIXTURES[1]
        self.assertEqual(fixture.site_name_ascii, "blocks.00.self_a.attn")
        self.assertEqual(
            dropout.h0_digest(
                seed=fixture.seed,
                epoch_index=fixture.epoch_index,
                global_optimizer_step=fixture.global_optimizer_step,
                site_ordinal=fixture.site_ordinal,
                site_name_ascii=fixture.site_name_ascii,
                shape=fixture.shape,
            ).hex(),
            "94bf4ff117cbf6d73c33b2803005429296f0f218fd61fad04a1dea4379b71e87",
        )

    def test_keep_scale_is_exact_raw_float32_and_not_double_scaled(self) -> None:
        self.assertEqual(dropout.KEEP_FLOAT32_BITS, 0x3F8E38E4)
        self.assertEqual(dropout.KEEP_FLOAT32_LE, struct.pack("<I", 0x3F8E38E4))
        mask = dropout.verify_golden(dropout.GOLDEN_FIXTURES[0])
        words = tuple(
            mask.expanded_float32_le[index : index + 4]
            for index in range(0, len(mask.expanded_float32_le), 4)
        )
        self.assertEqual(set(words), {dropout.DROP_FLOAT32_LE, dropout.KEEP_FLOAT32_LE})

    def test_pass_slices_are_distinct_elements_of_one_p2_tensor(self) -> None:
        mask = dropout.verify_golden(dropout.GOLDEN_FIXTURES[0])
        slice_numel = mask.numel // 2
        first = mask.expanded_float32_le[: 4 * slice_numel]
        second = mask.expanded_float32_le[4 * slice_numel :]
        self.assertNotEqual(first, second)


class DropoutInvalidTests(unittest.TestCase):
    BASE = dict(
        seed=1729,
        epoch_index=0,
        global_optimizer_step=0,
        site_ordinal=0,
        site_name_ascii="blocks.00.self_a.attn",
        shape=(2, 2, 4, 3, 3),
    )

    def test_system_and_pass_injection_are_not_api_fields(self) -> None:
        with self.assertRaises(TypeError):
            dropout.h0_preimage(**self.BASE, system_id="00")  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            dropout.generate_mask(**self.BASE, pass_id="AB")  # type: ignore[call-arg]
        first = dropout.h0_digest(**self.BASE)
        second = dropout.h0_digest(**self.BASE)
        self.assertEqual(first, second)

    def test_d0_one_and_bad_shape_types_are_rejected(self) -> None:
        with self.assertRaises(dropout.DropoutContractError):
            dropout.h0_preimage(**{**self.BASE, "shape": (1, 2, 4, 3, 3)})
        with self.assertRaises(TypeError):
            dropout.h0_preimage(**{**self.BASE, "shape": [2, 2, 4, 3, 3]})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            dropout.h0_preimage(**{**self.BASE, "shape": (2, True, 4, 3, 3)})
        with self.assertRaises(TypeError):
            dropout.h0_preimage(**{**self.BASE, "seed": True})
        with self.assertRaises(TypeError):
            dropout.h0_preimage(**{**self.BASE, "site_name_ascii": b"site"})  # type: ignore[arg-type]

    def test_integer_threshold_boundaries(self) -> None:
        self.assertFalse(dropout.keep_from_mixed_word(dropout.DROP_THRESHOLD - 1))
        self.assertTrue(dropout.keep_from_mixed_word(dropout.DROP_THRESHOLD))
        with self.assertRaises(TypeError):
            dropout.keep_from_mixed_word(float(dropout.DROP_THRESHOLD))  # type: ignore[arg-type]

    def test_lsb_packing_and_unused_high_bits(self) -> None:
        bits = (True, False, True, False, False, False, False, True, True)
        packed = dropout.pack_keep_bits_lsb_first(bits)
        self.assertEqual(packed, b"\x85\x01")
        self.assertEqual(packed[-1] & 0xFE, 0)
        with self.assertRaises(TypeError):
            dropout.pack_keep_bits_lsb_first([True])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            dropout.pack_keep_bits_lsb_first((1,))  # type: ignore[arg-type]

    def test_all_runtime_policy_mutants_are_rejected(self) -> None:
        self.assertEqual(
            dropout.validate_runtime_policy(dropout.RuntimePolicy()),
            "STATIC_RUNTIME_POLICY_VALIDATED_NONPRODUCTION",
        )
        mutants = (
            dict(world_size=2),
            dict(gradient_checkpointing=True),
            dict(hidden_recomputation=True),
            dict(fused_dropout=True),
            dict(framework_rng_calls=1),
            dict(framework_dropout_calls=1),
            dict(resolved_text_nonzero_dropout_sites=1),
            dict(eval_schedule_accesses=1),
        )
        for fields in mutants:
            with self.subTest(fields=fields):
                with self.assertRaises(dropout.RuntimePolicyError):
                    dropout.validate_runtime_policy(dropout.RuntimePolicy(**fields))
        self.assertIsNone(dropout.eval_mask())

    def test_object_forged_site_policy_and_golden_are_rejected(self) -> None:
        canonical = dropout.site_inventory("mime")
        forged_site = object.__new__(dropout.DropoutSite)
        object.__setattr__(forged_site, "ordinal", False)
        object.__setattr__(forged_site, "name", canonical[0].name)
        object.__setattr__(forged_site, "axes", canonical[0].axes)
        with self.assertRaises(TypeError):
            dropout.validate_canonical_inventory(
                "mime", (forged_site,) + canonical[1:]
            )

        forged_policy = object.__new__(dropout.RuntimePolicy)
        object.__setattr__(forged_policy, "world_size", True)
        object.__setattr__(forged_policy, "gradient_checkpointing", 0)
        object.__setattr__(forged_policy, "hidden_recomputation", 0)
        object.__setattr__(forged_policy, "fused_dropout", 0)
        object.__setattr__(forged_policy, "framework_rng_calls", False)
        object.__setattr__(forged_policy, "framework_dropout_calls", False)
        object.__setattr__(forged_policy, "resolved_text_nonzero_dropout_sites", False)
        object.__setattr__(forged_policy, "eval_schedule_accesses", False)
        with self.assertRaises(TypeError):
            dropout.validate_runtime_policy(forged_policy)

        fixture = dropout.GOLDEN_FIXTURES[0]
        forged_fixture = object.__new__(dropout.GoldenFixture)
        for field_name in fixture.__slots__:
            object.__setattr__(forged_fixture, field_name, getattr(fixture, field_name))
        object.__setattr__(forged_fixture, "drop_count", float(fixture.drop_count))
        with self.assertRaises(TypeError):
            dropout.verify_golden(forged_fixture)

    def test_module_has_no_framework_or_random_import(self) -> None:
        source = inspect.getsource(dropout)
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue({"random", "secrets", "numpy", "torch"}.isdisjoint(imported))


if __name__ == "__main__":
    unittest.main()
