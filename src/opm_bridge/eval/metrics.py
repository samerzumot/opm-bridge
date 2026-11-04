"""Pre-registered statistical evaluation, subject-level bootstrap CIs, and Phase 1.5 signal gate (§4.1, §5.2, §9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

DecisionVerdict = Literal["pre-training helped", "equivalent", "inconclusive"]


@dataclass
class BootstrapResult:
    """Subject-level bootstrap summary with seed averaging inside each test subject."""

    mean: float
    ci_low: float
    ci_high: float
    n_subjects: int
    n_resamples: int


@dataclass
class PrimaryComparisonResult:
    """Result of the pre-registered primary comparison (§4.1)."""

    budget: int | float
    delta_mean: float
    delta_ci_low: float
    delta_ci_high: float
    pretrained_bacc: BootstrapResult
    control_a_bacc: BootstrapResult
    sesoi: float
    verdict: DecisionVerdict


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute primary (balanced accuracy) and secondary (macro-F1, AUROC) metrics."""
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics = {"balanced_accuracy": bacc, "macro_f1": macro_f1}
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                auroc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auroc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            metrics["auroc"] = auroc
        except ValueError:
            metrics["auroc"] = float("nan")
    return metrics


def subject_bootstrap_ci(
    subject_seed_scores: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> BootstrapResult:
    """Compute 95% bootstrap CI over test subjects, averaging seeds within subject first (§4.1).

    Args:
        subject_seed_scores: Array of shape (N_subjects, N_seeds) or (N_subjects,).
        n_resamples: Number of bootstrap resamples over test subjects (default 10,000).
        alpha: Significance level (default 0.05 for 95% CI).
        rng_seed: Deterministic random seed.
    """
    scores = np.asarray(subject_seed_scores, dtype=np.float64)
    if scores.ndim == 2:
        subj_means = np.mean(scores, axis=1)
    elif scores.ndim == 1:
        subj_means = scores
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {scores.shape}")

    n_subj = len(subj_means)
    assert n_subj >= 2, "Need at least 2 test subjects for subject-level bootstrap."

    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n_subj, size=(n_resamples, n_subj))
    boot_means = np.mean(subj_means[idx], axis=1)

    ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return BootstrapResult(
        mean=float(np.mean(subj_means)),
        ci_low=ci_low,
        ci_high=ci_high,
        n_subjects=n_subj,
        n_resamples=n_resamples,
    )


def evaluate_primary_comparison(
    pretrained_scores: np.ndarray,
    control_a_scores: np.ndarray,
    budget: int | float,
    sesoi: float = 0.03,
    n_resamples: int = 10_000,
    rng_seed: int = 42,
) -> PrimaryComparisonResult:
    """Evaluate the pre-registered primary hypothesis test (§4.1).

    Paired bootstrap over test subjects (with seeds averaged within subject first):
      Delta_i = BACC_pretrained(subject i) - BACC_control_A(subject i)

    Decision rule (§4.1):
      - 95% CI excludes 0 (ci_low > 0) AND delta_mean >= sesoi -> "pre-training helped"
      -Entire 95% CI within [-sesoi, +sesoi] -> "equivalent"
      - Everything else -> "inconclusive"
    """
    pt = np.asarray(pretrained_scores, dtype=np.float64)
    ctrl = np.asarray(control_a_scores, dtype=np.float64)
    assert pt.shape == ctrl.shape, f"Shape mismatch: {pt.shape} vs {ctrl.shape}"

    pt_subj = np.mean(pt, axis=1) if pt.ndim == 2 else pt
    ctrl_subj = np.mean(ctrl, axis=1) if ctrl.ndim == 2 else ctrl
    deltas = pt_subj - ctrl_subj

    delta_boot = subject_bootstrap_ci(deltas, n_resamples=n_resamples, rng_seed=rng_seed)
    pt_boot = subject_bootstrap_ci(pt_subj, n_resamples=n_resamples, rng_seed=rng_seed + 1)
    ctrl_boot = subject_bootstrap_ci(ctrl_subj, n_resamples=n_resamples, rng_seed=rng_seed + 2)

    if delta_boot.ci_low > 0.0 and delta_boot.mean >= sesoi:
        verdict: DecisionVerdict = "pre-training helped"
    elif delta_boot.ci_low >= -sesoi and delta_boot.ci_high <= sesoi:
        verdict = "equivalent"
    else:
        verdict = "inconclusive"

    return PrimaryComparisonResult(
        budget=budget,
        delta_mean=delta_boot.mean,
        delta_ci_low=delta_boot.ci_low,
        delta_ci_high=delta_boot.ci_high,
        pretrained_bacc=pt_boot,
        control_a_bacc=ctrl_boot,
        sesoi=sesoi,
        verdict=verdict,
    )


def check_phase15_signal_gate(
    specialist_scores: np.ndarray,
    classical_scores: np.ndarray,
    chance_level: float = 0.5,
    n_resamples: int = 10_000,
    rng_seed: int = 42,
) -> Tuple[bool, Dict[str, BootstrapResult]]:
    """Phase 1.5 signal gate (§5.2): at least one baseline must beat chance (95% CI > chance)."""
    spec_ci = subject_bootstrap_ci(specialist_scores, n_resamples=n_resamples, rng_seed=rng_seed)
    clas_ci = subject_bootstrap_ci(classical_scores, n_resamples=n_resamples, rng_seed=rng_seed + 1)
    passed = (spec_ci.ci_low > chance_level) or (clas_ci.ci_low > chance_level)
    return passed, {"simple_specialist": spec_ci, "classical_decoder": clas_ci}


def compute_curve_summaries(
    budgets: Sequence[float],
    pretrained_by_budget: Sequence[np.ndarray],
    control_a_by_budget: Sequence[np.ndarray],
    n_resamples: int = 10_000,
    rng_seed: int = 42,
) -> Dict[str, object]:
    """Compute descriptive secondary measures (§4.2): area under Delta-curve and convergence budget."""
    assert len(budgets) == len(pretrained_by_budget) == len(control_a_by_budget)
    delta_means: List[float] = []
    comparisons: List[PrimaryComparisonResult] = []
    convergence_budget: Optional[float] = None

    for i, b in enumerate(budgets):
        comp = evaluate_primary_comparison(
            pretrained_by_budget[i],
            control_a_by_budget[i],
            budget=b,
            n_resamples=n_resamples,
            rng_seed=rng_seed + 10 * i,
        )
        comparisons.append(comp)
        delta_means.append(comp.delta_mean)
        if convergence_budget is None and comp.delta_ci_low <= 0.0 <= comp.delta_ci_high:
            convergence_budget = float(b)

    x_arr = np.asarray(budgets, dtype=np.float64)
    y_arr = np.asarray(delta_means, dtype=np.float64)
    auc_delta = float(np.trapezoid(y_arr, x_arr)) if len(x_arr) >= 2 else 0.0

    return {
        "auc_delta": auc_delta,
        "convergence_budget_descriptive": convergence_budget,
        "per_budget": comparisons,
    }

