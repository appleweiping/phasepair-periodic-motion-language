from __future__ import annotations

import math
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from phasepair_core import objectives  # noqa: E402


def _canonical_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    motion = torch.zeros((2, 2, 512), dtype=torch.float32)
    motion[0, 0, 0] = 1.0
    motion[1, 0, 1] = 1.0
    motion[0, 1, 2] = 1.0
    motion[1, 1, 3] = 1.0
    text = torch.zeros((6, 512), dtype=torch.float32)
    text[:3, 0] = 1.0
    text[3:, 2] = 1.0
    positive = torch.zeros((2, 6), dtype=torch.bool)
    positive[0, :3] = True
    positive[1, 3:] = True
    return motion.contiguous(), text.contiguous(), positive.contiguous()


def test_ordered_pair_average_and_three_caption_objective() -> None:
    motion, text, positive = _canonical_inputs()
    logit_scale = torch.nn.Parameter(torch.tensor([math.log(1.0 / 0.07)], dtype=torch.float32))
    output = objectives.phasepair_multi_positive_objective(
        motion,
        text,
        logit_scale,
        positive,
    )
    assert output.logits.shape == (2, 6)
    assert output.logits.is_contiguous()
    assert bool(torch.isfinite(output.loss))
    assert output.status == "DATA_FREE_OBJECTIVE_COMPUTED_NONPRODUCTION_NO_RESULT"
    output.loss.backward()
    assert logit_scale.grad is not None
    assert bool(torch.isfinite(logit_scale.grad).all())


def test_high_logit_golden_is_stable_240() -> None:
    logits = torch.tensor([[120.0, -120.0], [-120.0, 120.0]], dtype=torch.float32)
    positive = torch.tensor([[False, True], [True, False]], dtype=torch.bool)
    motion_to_text, text_to_motion, total = objectives.symmetric_multi_positive_infonce(
        logits,
        positive,
    )
    assert float(motion_to_text) == pytest.approx(240.0, abs=1e-12)
    assert float(text_to_motion) == pytest.approx(240.0, abs=1e-12)
    assert float(total) == pytest.approx(240.0, abs=1e-12)


def test_multi_positive_numerator_uses_all_three_captions() -> None:
    logits = torch.tensor(
        [[3.0, 2.0, 1.0, -4.0], [-4.0, -4.0, -4.0, 3.0]],
        dtype=torch.float32,
    )
    positive = torch.tensor(
        [[True, True, True, False], [False, False, False, True]],
        dtype=torch.bool,
    )
    motion_to_text, text_to_motion, total = objectives.symmetric_multi_positive_infonce(
        logits,
        positive,
    )
    expected_rows = torch.stack(
        (
            -(
                torch.logsumexp(logits[0, :3], dim=0)
                - torch.logsumexp(logits[0], dim=0)
            ),
            -(logits[1, 3] - torch.logsumexp(logits[1], dim=0)),
        )
    )
    expected_motion = expected_rows.mean()
    assert torch.equal(motion_to_text, expected_motion)
    assert bool(torch.isfinite(text_to_motion))
    assert torch.equal(total, 0.5 * (motion_to_text + text_to_motion))


def test_temperature_clamps_to_one_and_one_hundred() -> None:
    motion, text, positive = _canonical_inputs()
    low = objectives.phasepair_multi_positive_objective(
        motion,
        text,
        torch.tensor([-100.0], dtype=torch.float32),
        positive,
    ).logits
    high = objectives.phasepair_multi_positive_objective(
        motion,
        text,
        torch.tensor([100.0], dtype=torch.float32),
        positive,
    ).logits
    base = objectives.ordered_pair_average_scores(motion, text)
    assert torch.equal(low, base)
    assert torch.equal(high, base * 100.0)


def test_ordered_residual_is_averaged_then_weighted_point_two() -> None:
    motion, text, positive = _canonical_inputs()
    residual = torch.zeros((2, 2, 6), dtype=torch.float32)
    residual[0].fill_(0.2)
    residual[1].fill_(0.4)
    output = objectives.phasepair_multi_positive_objective(
        motion,
        text,
        torch.tensor([0.0], dtype=torch.float32),
        positive,
        ordered_residual_scores=residual,
    )
    base = objectives.ordered_pair_average_scores(motion, text)
    assert torch.allclose(output.logits, base + 0.06, rtol=0.0, atol=1e-7)
    residual[0, 0, 0] = 1.0001
    with pytest.raises(objectives.ObjectiveContractError, match=r"\[-1,1\]"):
        objectives.phasepair_multi_positive_objective(
            motion,
            text,
            torch.tensor([0.0], dtype=torch.float32),
            positive,
            ordered_residual_scores=residual,
        )


@pytest.mark.parametrize(
    "mask",
    (
        torch.tensor([[False, False], [True, True]], dtype=torch.bool),
        torch.tensor([[True, False], [True, False]], dtype=torch.bool),
    ),
)
def test_low_level_loss_rejects_empty_positive_rows_or_columns(mask: torch.Tensor) -> None:
    with pytest.raises(objectives.ObjectiveContractError):
        objectives.symmetric_multi_positive_infonce(
            torch.zeros((2, 2), dtype=torch.float32),
            mask,
        )


def test_canonical_mask_requires_exact_three_to_one_mapping() -> None:
    motion, text, positive = _canonical_inputs()
    mutants = []
    missing = positive.clone()
    missing[0, 0] = False
    mutants.append(missing)
    duplicate = positive.clone()
    duplicate[1, 0] = True
    mutants.append(duplicate)
    for mutant in mutants:
        with pytest.raises(objectives.ObjectiveContractError):
            objectives.phasepair_multi_positive_objective(
                motion,
                text,
                torch.tensor([0.0], dtype=torch.float32),
                mutant,
            )


def test_exact_dtype_layout_finite_l2_and_tensor_types() -> None:
    motion, text, positive = _canonical_inputs()
    cases = (
        (motion.double(), text, positive),
        (motion.transpose(1, 2), text, positive),
        (motion, text.double(), positive),
        (motion, text, positive[:, ::2]),
    )
    for bad_motion, bad_text, bad_mask in cases:
        with pytest.raises((TypeError, objectives.ObjectiveContractError)):
            objectives.phasepair_multi_positive_objective(
                bad_motion,
                bad_text,
                torch.tensor([0.0], dtype=torch.float32),
                bad_mask,
            )
    nonfinite = motion.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(objectives.ObjectiveContractError):
        objectives.ordered_pair_average_scores(nonfinite, text)
    nonunit = motion.clone()
    nonunit[0, 0] *= 0.5
    with pytest.raises(objectives.ObjectiveContractError):
        objectives.ordered_pair_average_scores(nonunit, text)


def test_input_mutation_after_call_cannot_change_saved_logits() -> None:
    motion, text, positive = _canonical_inputs()
    output = objectives.phasepair_multi_positive_objective(
        motion,
        text,
        torch.tensor([0.0], dtype=torch.float32),
        positive,
    )
    saved = output.logits.detach().clone()
    motion.zero_()
    text.zero_()
    positive.zero_()
    assert torch.equal(output.logits, saved)


def test_snapshots_use_captured_tensor_primitives_not_instance_methods() -> None:
    motion, text, positive = _canonical_inputs()
    logit_scale = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    expected = objectives.phasepair_multi_positive_objective(
        motion,
        text,
        logit_scale,
        positive,
    ).logits

    def bomb(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("shadowed Tensor instance method was called")

    for value, names in (
        (motion, ("clone", "is_contiguous")),
        (text, ("clone", "is_contiguous")),
        (logit_scale, ("clone", "is_contiguous")),
        (positive, ("detach", "clone", "is_contiguous")),
    ):
        for name in names:
            setattr(value, name, bomb)

    output = objectives.phasepair_multi_positive_objective(
        motion,
        text,
        logit_scale,
        positive,
    )
    assert torch.equal(output.logits, expected)
    output.loss.backward()
    assert logit_scale.grad is not None


def test_public_constant_rebinding_cannot_weaken_canonical_dimensions() -> None:
    motion, text, positive = _canonical_inputs()
    with mock.patch.multiple(
        objectives,
        VECTOR_WIDTH=1,
        MAX_BATCH=999,
        CAPTIONS_PER_SOURCE=1,
    ):
        output = objectives.phasepair_multi_positive_objective(
            motion,
            text,
            torch.tensor([0.0], dtype=torch.float32),
            positive,
        )
        assert output.logits.shape == (2, 6)
        with pytest.raises(objectives.ObjectiveContractError):
            objectives.phasepair_multi_positive_objective(
                torch.ones((2, 1, 1), dtype=torch.float32),
                torch.ones((1, 1), dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.ones((1, 1), dtype=torch.bool),
            )


def test_output_constructor_is_closed_and_public_tensors_are_isolated() -> None:
    valid_logits = torch.zeros((2, 6), dtype=torch.float32)
    positive = torch.zeros((2, 6), dtype=torch.bool)
    positive[0, :3] = True
    positive[1, 3:] = True
    with pytest.raises(objectives.ObjectiveContractError):
        objectives.PhasePairObjectiveOutput(valid_logits, positive)
    with pytest.raises(objectives.ObjectiveContractError):
        objectives.PhasePairObjectiveOutput(
            valid_logits,
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

    motion, text, canonical_mask = _canonical_inputs()
    output = objectives.phasepair_multi_positive_objective(
        motion,
        text,
        torch.tensor([0.0], dtype=torch.float32),
        canonical_mask,
    )
    saved_logits = output.logits
    saved_mask = output.positive_mask
    saved_loss = output.loss
    output.logits.fill_(float("nan"))
    output.positive_mask.zero_()
    output.loss.fill_(123.0)
    assert torch.equal(output.logits, saved_logits)
    assert torch.equal(output.positive_mask, saved_mask)
    assert torch.equal(output.loss, saved_loss)
    with pytest.raises(AttributeError):
        output.status = "FORGED"  # type: ignore[misc]
