from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from phasepair_core import batching, readiness


ROOT = Path(__file__).resolve().parents[1]


def valid_batch() -> batching.PreparedMotionBatch:
    actor_a = np.zeros((2, 4, 262), dtype=np.float32)
    actor_b = np.zeros((2, 4, 262), dtype=np.float32)
    relation_ab = np.zeros((2, 4, 799), dtype=np.float32)
    relation_ba = np.zeros((2, 4, 799), dtype=np.float32)
    actor_a[0, :2] = np.float32(1.0)
    actor_b[0, :2] = np.float32(3.0)
    relation_ab[0, :2] = np.float32(5.0)
    relation_ba[0, :2] = np.float32(7.0)
    actor_a[1, :3] = np.float32(2.0)
    actor_b[1, :3] = np.float32(4.0)
    relation_ab[1, :3] = np.float32(6.0)
    relation_ba[1, :3] = np.float32(8.0)
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.uint8)
    return batching.PreparedMotionBatch(
        actor_a,
        actor_b,
        relation_ab,
        relation_ba,
        mask,
        (2, 3),
        (b"A" * 32, b"B" * 32),
    )


class BatchingTests(unittest.TestCase):
    def test_ordered_passes_and_identity_are_exact(self) -> None:
        batch = valid_batch()
        passes = batching.build_ordered_motion_passes(batch)
        self.assertEqual(passes.actor_role_a.shape, (2, 2, 4, 262))
        self.assertEqual(passes.actor_role_b.shape, (2, 2, 4, 262))
        self.assertEqual(passes.relation.shape, (2, 2, 4, 799))
        self.assertEqual(passes.early_fusion.shape, (2, 2, 4, 786))
        np.testing.assert_array_equal(passes.actor_role_a[0], batch.actor_a)
        np.testing.assert_array_equal(passes.actor_role_a[1], batch.actor_b)
        np.testing.assert_array_equal(passes.actor_role_b[0], batch.actor_b)
        np.testing.assert_array_equal(passes.actor_role_b[1], batch.actor_a)
        np.testing.assert_array_equal(passes.relation[0], batch.relation_ab)
        np.testing.assert_array_equal(passes.relation[1], batch.relation_ba)
        np.testing.assert_array_equal(
            passes.early_fusion[..., 524:],
            passes.actor_role_b - passes.actor_role_a,
        )
        for array in (
            batch.actor_a,
            batch.actor_b,
            batch.relation_ab,
            batch.relation_ba,
            batch.valid_mask,
            passes.actor_role_a,
            passes.early_fusion,
        ):
            self.assertFalse(array.flags.writeable)
        identity = batching.prepared_batch_identity_bytes(batch)
        self.assertTrue(identity.startswith(batching.IDENTITY_DOMAIN + b"\x00"))
        self.assertEqual(batching.prepared_batch_sha256(batch), hashlib.sha256(identity).hexdigest())

    def test_input_mutation_cannot_change_snapshot(self) -> None:
        source = np.zeros((1, 2, 262), dtype=np.float32)
        relation = np.zeros((1, 2, 799), dtype=np.float32)
        mask = np.array([[1, 0]], dtype=np.uint8)
        batch = batching.PreparedMotionBatch(
            source,
            source.copy(),
            relation,
            relation.copy(),
            mask,
            (1,),
            (b"A" * 32,),
        )
        source[0, 0, 0] = np.float32(99.0)
        mask[0, 0] = 0
        self.assertEqual(float(batch.actor_a[0, 0, 0]), 0.0)
        self.assertEqual(int(batch.valid_mask[0, 0]), 1)
        batch.__post_init__()
        self.assertFalse(batch.actor_a.flags.writeable)
        with self.assertRaises(TypeError):
            batch.__post_init__(lambda value: value)

    def test_snapshot_precedes_numeric_and_mask_content_validation(self) -> None:
        actor = np.zeros((1, 2, 262), dtype=np.float32)
        mask = np.array([[1, 0]], dtype=np.uint8)
        original_snapshot = batching._immutable_array

        def corrupt_actor_during_snapshot(value: np.ndarray) -> np.ndarray:
            if value is actor:
                value[0, 0, 0] = np.nan
            return original_snapshot(value)

        with self.assertRaisesRegex(batching.BatchContractError, "finite"):
            batching._require_float32_tensor(
                actor,
                "actor",
                feature_dim=262,
                _snapshot=corrupt_actor_during_snapshot,
            )

        actor.fill(np.float32(0.0))

        def corrupt_mask_during_snapshot(value: np.ndarray) -> np.ndarray:
            if value is mask:
                value[0, 0] = 2
            return original_snapshot(value)

        with self.assertRaisesRegex(batching.BatchContractError, "binary"):
            batching._require_mask(
                mask,
                batch_size=1,
                padded_time=2,
                lengths=(1,),
                _snapshot=corrupt_mask_during_snapshot,
            )

    def test_snapshot_and_identity_semantics_are_captured_against_rebind(self) -> None:
        source = np.zeros((1, 2, 262), dtype=np.float32)
        other_actor = source.copy()
        relation = np.zeros((1, 2, 799), dtype=np.float32)
        other_relation = relation.copy()
        mask = np.array([[1, 0]], dtype=np.uint8)

        def bomb(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("a rebound helper was read")

        with mock.patch.multiple(
            batching,
            _immutable_array=lambda value: value,
            _require_float32_tensor=bomb,
            _require_exact_int=bomb,
            _require_lengths=bomb,
            _require_mask=bomb,
            _require_commitments=bomb,
            _require_positive_zero_padding=bomb,
            validate_prepared_motion_batch=bomb,
            _freeze_float32=bomb,
            IDENTITY_DOMAIN=b"forged-domain",
            np=None,
            hashlib=None,
        ):
            batch = batching.PreparedMotionBatch(
                source,
                other_actor,
                relation,
                other_relation,
                mask,
                (1,),
                (b"A" * 32,),
            )
            passes = batching.build_ordered_motion_passes(batch)
            identity = batching.prepared_batch_identity_bytes(batch)
            digest = batching.prepared_batch_sha256(batch)

        source[0, 0, 0] = np.nan
        mask[0, 0] = 0
        self.assertEqual(float(batch.actor_a[0, 0, 0]), 0.0)
        self.assertEqual(int(batch.valid_mask[0, 0]), 1)
        self.assertFalse(passes.actor_role_a.flags.writeable)
        self.assertTrue(identity.startswith(b"phasepair-prepared-motion-batch-v1\x00"))
        self.assertEqual(digest, hashlib.sha256(identity).hexdigest())

    def test_rejects_dynamic_subclass_before_reads(self) -> None:
        class Bomb(np.ndarray):
            def __getattribute__(self, name: str):
                if name in {"dtype", "ndim", "shape", "flags"}:
                    raise RuntimeError("dynamic metadata read")
                return super().__getattribute__(name)

        source = np.zeros((1, 2, 262), dtype=np.float32).view(Bomb)
        with self.assertRaisesRegex(TypeError, "exact base numpy.ndarray"):
            batching.PreparedMotionBatch(
                source,
                np.zeros((1, 2, 262), dtype=np.float32),
                np.zeros((1, 2, 799), dtype=np.float32),
                np.zeros((1, 2, 799), dtype=np.float32),
                np.array([[1, 0]], dtype=np.uint8),
                (1,),
                (b"A" * 32,),
            )

    def test_rejects_dtype_layout_numeric_padding_and_mask_errors(self) -> None:
        base = valid_batch()
        cases: list[tuple[str, object]] = []
        cases.append(("dtype", base.actor_a.astype(np.float64)))
        cases.append(("layout", np.asfortranarray(np.array(base.actor_a, copy=True))))
        numeric = np.array(base.actor_a, copy=True)
        numeric[0, 0, 0] = np.nan
        cases.append(("numeric", numeric))
        negative_zero = np.array(base.actor_a, copy=True)
        negative_zero[0, 2, 0] = np.float32(-0.0)
        cases.append(("padding", negative_zero))
        for label, actor_a in cases:
            with self.subTest(label=label), self.assertRaises((TypeError, batching.BatchContractError)):
                batching.PreparedMotionBatch(
                    actor_a,
                    base.actor_b,
                    base.relation_ab,
                    base.relation_ba,
                    base.valid_mask,
                    base.valid_lengths,
                    base.pair_commitments,
                )
        bad_mask = np.array(base.valid_mask, copy=True)
        bad_mask[0] = (1, 0, 1, 0)
        with self.assertRaises(batching.BatchContractError):
            batching.PreparedMotionBatch(
                base.actor_a,
                base.actor_b,
                base.relation_ab,
                base.relation_ba,
                bad_mask,
                base.valid_lengths,
                base.pair_commitments,
            )

    def test_rejects_commitment_alias_and_preserves_frozen_row_order(self) -> None:
        base = valid_batch()
        with self.assertRaises(batching.BatchContractError):
            batching.PreparedMotionBatch(
                base.actor_a,
                base.actor_b,
                base.relation_ab,
                base.relation_ba,
                base.valid_mask,
                base.valid_lengths,
                (b"A" * 32, b"A" * 32),
            )
        reversed_batch = batching.PreparedMotionBatch(
            base.actor_a,
            base.actor_b,
            base.relation_ab,
            base.relation_ba,
            base.valid_mask,
            base.valid_lengths,
            (b"B" * 32, b"A" * 32),
        )
        self.assertEqual(reversed_batch.pair_commitments, (b"B" * 32, b"A" * 32))
        self.assertNotEqual(
            batching.prepared_batch_sha256(base),
            batching.prepared_batch_sha256(reversed_batch),
        )

    def test_public_shape_limit_rebinding_cannot_weaken_batch_contract(self) -> None:
        with mock.patch.multiple(
            batching,
            MAX_BATCH=999,
            MAX_PADDED_TIME=999,
            MAX_VALID_TIME=999,
            ACTOR_DIM=1,
            RELATION_DIM=1,
            _require_exact_int=lambda *_args, **_kwargs: 1,
        ):
            with self.assertRaises(batching.BatchContractError):
                batching.PreparedMotionBatch(
                    np.zeros((129, 1, 262), dtype=np.float32),
                    np.zeros((129, 1, 262), dtype=np.float32),
                    np.zeros((129, 1, 799), dtype=np.float32),
                    np.zeros((129, 1, 799), dtype=np.float32),
                    np.ones((129, 1), dtype=np.uint8),
                    (1,) * 129,
                    tuple(index.to_bytes(32, "big") for index in range(129)),
                )
            with self.assertRaises(batching.BatchContractError):
                batching.PreparedMotionBatch(
                    np.zeros((1, 301, 262), dtype=np.float32),
                    np.zeros((1, 301, 262), dtype=np.float32),
                    np.zeros((1, 301, 799), dtype=np.float32),
                    np.zeros((1, 301, 799), dtype=np.float32),
                    np.array([[1] + [0] * 300], dtype=np.uint8),
                    (1,),
                    (b"A" * 32,),
                )
            with self.assertRaises(batching.BatchContractError):
                batching.PreparedMotionBatch(
                    np.zeros((1, 2, 1), dtype=np.float32),
                    np.zeros((1, 2, 1), dtype=np.float32),
                    np.zeros((1, 2, 1), dtype=np.float32),
                    np.zeros((1, 2, 1), dtype=np.float32),
                    np.array([[1, 0]], dtype=np.uint8),
                    (1,),
                    (b"A" * 32,),
                )

    def test_ordered_passes_direct_construction_revalidates_and_snapshots(self) -> None:
        canonical = batching.build_ordered_motion_passes(valid_batch())
        role_a = np.array(canonical.actor_role_a, copy=True)
        role_b = np.array(canonical.actor_role_b, copy=True)
        relation = np.array(canonical.relation, copy=True)
        early = np.array(canonical.early_fusion, copy=True)
        mask = np.array(canonical.valid_mask, copy=True)
        checked = batching.OrderedMotionPasses(
            role_a,
            role_b,
            relation,
            early,
            mask,
            canonical.pair_commitments,
        )
        role_a[0, 0, 0, 0] = np.float32(99.0)
        mask[0, 0] = 0
        self.assertNotEqual(float(checked.actor_role_a[0, 0, 0, 0]), 99.0)
        self.assertEqual(int(checked.valid_mask[0, 0]), 1)
        for value in (
            checked.actor_role_a,
            checked.actor_role_b,
            checked.relation,
            checked.early_fusion,
            checked.valid_mask,
        ):
            self.assertFalse(value.flags.writeable)
        checked.__post_init__()
        self.assertFalse(checked.actor_role_a.flags.writeable)
        with self.assertRaises(TypeError):
            checked.__post_init__(lambda value: value)

        bad_early = np.array(canonical.early_fusion, copy=True)
        bad_early[0, 0, 0, 0] += np.float32(1.0)
        with self.assertRaisesRegex(batching.BatchContractError, "early_fusion"):
            batching.OrderedMotionPasses(
                canonical.actor_role_a,
                canonical.actor_role_b,
                canonical.relation,
                bad_early,
                canonical.valid_mask,
                canonical.pair_commitments,
            )
        with self.assertRaises(batching.BatchContractError):
            batching.OrderedMotionPasses(
                np.zeros((2, 1, 1, 1), dtype=np.float32),
                np.zeros((2, 1, 1, 1), dtype=np.float32),
                np.zeros((2, 1, 1, 1), dtype=np.float32),
                np.zeros((2, 1, 1, 1), dtype=np.float32),
                np.ones((1, 1), dtype=np.uint8),
                (b"A" * 32,),
            )

    def test_ordered_pass_builder_rejects_finite_input_difference_overflow(self) -> None:
        maximum = np.finfo(np.float32).max
        actor_a = np.full((1, 1, 262), maximum, dtype=np.float32)
        actor_b = np.full((1, 1, 262), -maximum, dtype=np.float32)
        relation = np.zeros((1, 1, 799), dtype=np.float32)
        batch = batching.PreparedMotionBatch(
            actor_a,
            actor_b,
            relation,
            relation.copy(),
            np.ones((1, 1), dtype=np.uint8),
            (1,),
            (b"A" * 32,),
        )
        with self.assertRaisesRegex(batching.BatchContractError, "difference.*finite"):
            batching.build_ordered_motion_passes(batch)

    def test_forged_dataclass_is_rebuilt(self) -> None:
        base = valid_batch()
        forged = object.__new__(batching.PreparedMotionBatch)
        object.__setattr__(forged, "actor_a", base.actor_a)
        object.__setattr__(forged, "actor_b", base.actor_b)
        object.__setattr__(forged, "relation_ab", base.relation_ab)
        object.__setattr__(forged, "relation_ba", base.relation_ba)
        object.__setattr__(forged, "valid_mask", base.valid_mask)
        object.__setattr__(forged, "valid_lengths", (True, 3))
        object.__setattr__(forged, "pair_commitments", base.pair_commitments)
        with self.assertRaises(TypeError):
            batching.validate_prepared_motion_batch(forged)


class ReadinessTests(unittest.TestCase):
    def test_signal_runtime_and_fake_torch_observation(self) -> None:
        fake = SimpleNamespace(
            __version__="2.12.0+cpu",
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        observed = readiness.observe_local_runtime(torch_importer=lambda: fake)
        self.assertEqual(observed.python_version, readiness.SIGNAL_PYTHON)
        self.assertEqual(observed.numpy_version, readiness.SIGNAL_NUMPY)
        self.assertEqual(
            readiness.validate_signal_oracle_runtime(observed),
            "SIGNAL_ORACLE_RUNTIME_MATCH_NONPRODUCTION",
        )
        forged = object.__new__(readiness.RuntimeObservation)
        object.__setattr__(forged, "python_version", readiness.SIGNAL_PYTHON)
        object.__setattr__(forged, "python_implementation", "CPython")
        object.__setattr__(forged, "numpy_version", readiness.SIGNAL_NUMPY)
        object.__setattr__(forged, "torch_version", "x")
        object.__setattr__(forged, "torch_cuda_version", "NONE")
        object.__setattr__(forged, "cuda_available", 0)
        object.__setattr__(forged, "platform_string", "x")
        with self.assertRaises(TypeError):
            readiness.validate_signal_oracle_runtime(forged)

    def test_missing_and_complete_references_never_authorize_training(self) -> None:
        missing = readiness.assess_training_readiness(
            readiness.TrainingGateReferences(b"C" * 32)
        )
        self.assertEqual(missing.status, readiness.STATUS_HOLD)
        self.assertEqual(missing.missing_gates, readiness.GATE_FIELDS)
        complete = readiness.TrainingGateReferences(
            b"C" * 32,
            *(bytes([index]) * 32 for index in range(1, 9)),
        )
        assessment = readiness.assess_training_readiness(complete)
        self.assertEqual(assessment.status, readiness.STATUS_REFERENCES_PRESENT)
        self.assertEqual(assessment.authority, 0)
        self.assertFalse(assessment.production)
        self.assertFalse(assessment.training_authorized)
        self.assertEqual(assessment.missing_gates, ())
        decoded = json.loads(readiness.canonical_assessment_bytes(assessment))
        self.assertEqual(decoded["authority"], 0)
        self.assertFalse(decoded["training_authorized"])

    def test_gate_references_reject_bool_aliases_and_forged_objects(self) -> None:
        with self.assertRaises(TypeError):
            readiness.TrainingGateReferences(b"A" * 31)
        with self.assertRaises(TypeError):
            readiness.TrainingGateReferences(b"A" * 32, resolved_clip_receipt_sha256=True)
        forged = object.__new__(readiness.TrainingGateReferences)
        object.__setattr__(forged, "scientific_contract_sha256", b"A" * 32)
        for field in readiness.GATE_FIELDS:
            object.__setattr__(forged, field, None)
        object.__setattr__(forged, readiness.GATE_FIELDS[0], bytearray(b"A" * 32))
        with self.assertRaises(TypeError):
            readiness.assess_training_readiness(forged)
        forged_assessment = object.__new__(readiness.TrainingReadinessAssessment)
        object.__setattr__(forged_assessment, "status", readiness.STATUS_HOLD)
        object.__setattr__(forged_assessment, "authority", 9)
        object.__setattr__(forged_assessment, "production", True)
        object.__setattr__(forged_assessment, "training_authorized", True)
        object.__setattr__(forged_assessment, "missing_gates", readiness.GATE_FIELDS)
        object.__setattr__(forged_assessment, "contract_family", readiness.CONTRACT_FAMILY)
        with self.assertRaises(readiness.ReadinessError):
            readiness.canonical_assessment_bytes(forged_assessment)

    def test_public_policy_rebinding_cannot_elevate_or_erase_gates(self) -> None:
        complete = readiness.TrainingGateReferences(
            b"C" * 32,
            *(bytes([index]) * 32 for index in range(1, 9)),
        )
        with mock.patch.multiple(
            readiness,
            AUTHORITY=7,
            GATE_FIELDS=(),
            STATUS_HOLD="GRANTED",
            STATUS_REFERENCES_PRESENT="AUTHORIZED",
            CONTRACT_FAMILY="ATTACKER",
        ):
            assessment = readiness.assess_training_readiness(complete)
            self.assertEqual(assessment.authority, 0)
            self.assertEqual(
                assessment.status,
                "REFERENCES_PRESENT_VERIFICATION_REQUIRED",
            )
            self.assertEqual(
                assessment.contract_family,
                "PHASEPAIR_SCIENTIFIC_CONTRACT_20260824_165840",
            )
            decoded = json.loads(readiness.canonical_assessment_bytes(assessment))
            self.assertEqual(decoded["authority"], 0)
            missing = readiness.assess_training_readiness(
                readiness.TrainingGateReferences(b"C" * 32)
            )
            self.assertEqual(missing.status, "HOLD")
            self.assertEqual(len(missing.missing_gates), 8)

    def test_module_cli_is_canonical_silent_hold_exit42(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "phasepair_core.readiness"],
            cwd=ROOT,
            env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 42)
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(
            completed.stdout,
            json.dumps(
                json.loads(completed.stdout),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8") + b"\n",
        )
        self.assertEqual(json.loads(completed.stdout)["status"], readiness.STATUS_HOLD)


if __name__ == "__main__":
    unittest.main()
