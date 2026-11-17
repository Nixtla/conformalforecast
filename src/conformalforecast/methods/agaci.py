"""Aggregated Adaptive Conformal Inference (AgACI) for time series (Zaffran et al. 2022)."""

from __future__ import annotations

import copy
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from .._quantiles import (
    ExpandingSortedScores,
    _aci_levels,
    _aci_saturation,
    _ceil,
    masked_quantile,
)
from ..aggregators import EWA
from ..base import (
    CalibrationResult,
    Exog,
    Forecast,
    InsufficientCalibrationSamples,
    Interval,
    ScoreFunction,
    as_forecast,
)
from ..nonconformity.absolute import AbsoluteResidual
from .base import ConformalForecaster, ConformalMethod


def _pinball_loss(
    truth: NDArray[np.floating],
    prediction: NDArray[np.floating],
    beta: float,
) -> NDArray[np.floating]:
    """Pinball loss ``max(beta * (y - yhat), (beta - 1) * (y - yhat))``."""
    diff = truth - prediction
    return np.maximum(beta * diff, (beta - 1.0) * diff)


def _expert_bounds(
    point: NDArray[np.floating],
    thresholds: NDArray[np.floating],
    degenerate: NDArray[np.bool_],
    clip_lower: float,
    clip_upper: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Per-expert ``(lower, upper)``, clipped into ``[clip_lower, clip_upper]``.

    ``point`` broadcasts against ``thresholds`` and ``degenerate``. Where
    ``degenerate`` is set -- an expert asking for the empty set
    (``alpha_t >= 1``), or a cell's first valid calibration step -- both bounds
    are the point itself, as in ACI. Inverting the ``-sentinel`` threshold
    instead would give ``lower > upper``, which after clipping becomes
    ``[clip_upper, clip_lower]``: a finite but inverted pair that pulls the
    aggregate away from the forecast and earns a pinball loss set by the clip
    width rather than by the expert. Both bounds are clipped on both sides so an
    infinite-set expert (``+sentinel`` threshold) lands on the band edges.
    """
    lower = point - thresholds
    upper = point + thresholds
    point_b = np.broadcast_to(point, lower.shape)
    lower = np.where(degenerate, point_b, lower)
    upper = np.where(degenerate, point_b, upper)
    return np.clip(lower, clip_lower, clip_upper), np.clip(upper, clip_lower, clip_upper)


class AggregatedAdaptiveConformalInference(ConformalMethod["AgACIForecaster"]):
    """
    Aggregated Adaptive Conformal Inference on panel time series.

    Runs ``K = len(gammas)`` parallel ACI experts with different learning rates
    and aggregates their interval bounds via online expert aggregation. Lower
    and upper bounds use independent aggregators driven by per-expert pinball
    loss at ``beta_lower = alpha / 2`` and ``beta_upper = 1 - alpha / 2``, so the
    interval is generally asymmetric.

    See :class:`~conformalforecast.methods.base.ConformalMethod` for the
    calibrate/predict/update lifecycle and the ``calibrate`` / ``calibrate_cv``
    split. ``fit`` returns an :class:`AgACIForecaster`, whose ``update`` advances
    the experts and both aggregators in place.

    Any adapter works — no special capabilities required. Bind one with
    ``calibrate(forecaster, histories, truths)`` or
    ``calibrate_cv(forecaster, n_windows)``. Construction is keyword-only:
    ``AggregatedAdaptiveConformalInference(alpha=0.1)``.

    Parameters
    ----------
    alpha : float
        Miscoverage in (0, 1). Target coverage is ``1 - alpha``.
    gammas : tuple of float, default (0.001, 0.005, 0.01, 0.05, 0.1, 0.2)
        Grid of ACI learning rates. Non-empty; each value > 0.
    aggregator : EWA, default ``EWA()``
        Prototype aggregator, deep-copied and initialized once per bound
        (lower, upper) at fit time. **Typed as the concrete** :class:`EWA`
        rather than the abstract
        :class:`~conformalforecast.aggregators.base.OnlineAggregator`, so that
        ``EWA.eta`` shows up as nested config in
        ``model_json_schema()`` — a field typed as an ABC produces an empty
        nested schema. An :class:`EWA` subclass is accepted; an
        ``OnlineAggregator`` subclass that does *not* inherit from
        :class:`EWA` is rejected with ``pydantic.ValidationError``. That
        narrowing is deliberate.
    interval_clip_lower, interval_clip_upper : float or None
        Clip thresholds. Both-or-neither; ``lower < upper``. When None, computed
        at fit as ``min(truths) -/+ 5 * std(truths)`` and held on the fitted
        object's ``interval_clip_lower_`` / ``interval_clip_upper_``.
    seed : int, default 0
        Unused (kept for a uniform constructor signature); AgACI is
        deterministic by construction.
    score : ScoreFunction or None, default None
        Defaults to :class:`AbsoluteResidual`.

    Attributes
    ----------
    n_experts : int
        ``len(gammas)``, read-only — derived from the grid rather than
        settable alongside it.

    Notes
    -----
    **Saturated experts.** Each expert is a plain ACI recursion, so an expert
    whose ``alpha_t`` leaves ``(0, 1)`` asks for the whole real line
    (``alpha_t <= 0``) or the empty set (``alpha_t >= 1``); see the Notes on
    :class:`~conformalforecast.methods.aci.AdaptiveConformalInference`. Here an
    empty-set expert contributes the point interval ``[prediction, prediction]``
    (and still counts every step as a miss), and an infinite-set expert's bounds
    are clipped to ``[interval_clip_lower_, interval_clip_upper_]``. Each
    expert's bounds then enter the aggregators like any other's: a saturated
    expert is weighted by its pinball loss, not excluded, so it can carry real
    weight -- an empty-set expert in particular when the target ``alpha`` is
    large, since a point interval at the forecast is then a good lower and upper
    quantile. The emitted interval is always finite and never has
    ``lower > upper``. The clip is the price: when it binds, the long-run
    coverage guarantee of the underlying experts no longer holds exactly.
    ``alpha_t_per_expert_`` shows which experts are saturated.
    """

    REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]] = ()
    IS_ONLINE: ClassVar[bool] = True

    gammas: tuple[Annotated[float, Field(gt=0.0)], ...] = Field(
        default=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2),
        min_length=1,
        description=(
            "Grid of ACI learning rates, one expert per value: expert k runs "
            "alpha_t <- alpha_t + gammas[k] * (alpha - 1{miss}) on its own copy "
            "of the state, and the aggregator learns online, per (series, "
            "horizon) cell, which rate to put weight on. No single value in the "
            "grid is special, but the ends bracket the behaviour: a rate near 0 "
            "barely adapts and stays close to unweighted split conformal, while "
            "a large rate chases recent misses and gives noisy widths. Widening "
            "the grid costs compute and memory — every expert carries its own "
            "alpha_t and cumulative loss, each shape (n_experts, n_series, "
            "horizon) — and slightly slows convergence, since weight has more "
            "losers to move away from; the opposite risk is worse, because a "
            "too-narrow or badly centred grid caps the method at the best rate "
            "you happened to include. The knob exists so you do not have to "
            "commit to ACI's single gamma up front."
        ),
    )
    aggregator: EWA = Field(
        default_factory=EWA,
        description=(
            "Online aggregator prototype used to combine the per-expert bounds. "
            "It is deep-copied twice at fit — once for the lower bound, once "
            "for the upper — and each copy is driven by per-expert pinball loss "
            "at beta = alpha / 2 and 1 - alpha / 2 respectively, which is why "
            "AgACI's intervals are generally asymmetric. The prototype itself "
            "is never mutated, so it stays plain config. The default EWA() "
            "(eta=1.0) is the aggregation rule of Zaffran et al.; its eta is "
            "nested config here, and trades adaptation speed against weight "
            "stability (see EWA.eta). This field is typed as the concrete EWA "
            "rather than the abstract OnlineAggregator precisely so eta stays "
            "visible in this schema: an ABC-typed field produces an empty "
            "nested schema. The deliberate consequence is that an EWA subclass "
            "is accepted but a non-EWA OnlineAggregator subclass is rejected."
        ),
    )
    interval_clip_lower: float | None = Field(
        default=None,
        description=(
            "Hard floor applied to every emitted lower bound, in the units of "
            "the target. Both clips must be given together or neither: when "
            "both are None, fit resolves them from the calibration truths as "
            "min(truths) - 5 * std(truths) and max(truths) + 5 * std(truths) "
            "and keeps them on the separate fitted attributes "
            "interval_clip_lower_ / interval_clip_upper_. The clip exists "
            "because an expert on a large gamma can drive its alpha_t towards 0 "
            "and emit an absurdly wide interval after a single miss. A tight "
            "clip keeps widths interpretable but silently truncates genuinely "
            "wide intervals and so loses coverage exactly when a regime shift "
            "makes the width real; a loose clip preserves coverage but lets one "
            "runaway expert dominate the reported width. Set it explicitly when "
            "the target is physically bounded (a non-negative count, a share)."
        ),
    )
    interval_clip_upper: float | None = Field(
        default=None,
        description=(
            "Hard ceiling applied to every emitted upper bound, in the units of "
            "the target. Must be strictly greater than interval_clip_lower, and "
            "obeys the same both-or-neither rule and the same fit-time "
            "resolution from the calibration truths (max(truths) + 5 * "
            "std(truths)). Same trade-off as interval_clip_lower: too tight "
            "truncates real intervals and costs coverage, too loose lets a "
            "single runaway expert set the width."
        ),
    )

    @property
    def n_experts(self) -> int:
        """Number of ACI experts, ``len(gammas)``."""
        return len(self.gammas)

    @model_validator(mode="after")
    def _validate_clips(self) -> AggregatedAdaptiveConformalInference:
        """Enforce both-or-neither clips with ``lower < upper``."""
        lower, upper = self.interval_clip_lower, self.interval_clip_upper
        if (lower is None) != (upper is None):
            raise ValueError(
                "interval_clip_lower and interval_clip_upper must both be provided or both be None."
            )
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(
                f"interval_clip_lower ({lower}) must be strictly less "
                f"than interval_clip_upper ({upper})."
            )
        return self

    def _default_score(self) -> ScoreFunction:
        return AbsoluteResidual()

    def _new_aggregator(self, n_experts: int, n_series: int, horizon: int) -> EWA:
        """A fresh, initialized clone of the configured prototype.

        deepcopy rather than ``model_copy`` so a subclass carrying state pydantic
        does not know about is cloned too. The prototype is never mutated.
        """
        agg = copy.deepcopy(self.aggregator)
        agg.initialize(n_experts, n_series, horizon)
        return agg

    def fit(
        self,
        predictions: Forecast,
        truths: Forecast,
        *,
        X: Exog | None = None,
    ) -> AgACIForecaster:
        """
        Run the online AgACI loop over the calibration windows.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_windows, horizon)
            NaN cells mark windows that do not exist for a series.
        truths : Forecast, shape (n_series, n_windows, horizon)
        X : Exog or None
            Ignored by this method.

        Returns
        -------
        AgACIForecaster
            ``calibration_.diagnostics`` carries ``"gammas"`` and a copy of
            ``alpha_t_per_expert_``.

        Raises
        ------
        ValueError
            If array shapes are invalid or mismatched.
        InsufficientCalibrationSamples
            If any series has fewer than ``ceil(1 / alpha)`` effective windows.
        """
        preds = as_forecast(predictions, "predictions")
        obs = as_forecast(truths, "truths")
        if preds.shape != obs.shape:
            raise ValueError(
                f"predictions and truths must have the same shape, got "
                f"{preds.shape} and {obs.shape}"
            )
        score_fn = self._bind_score()
        score_fn.fit(preds, obs)
        all_scores = score_fn.score(preds, obs)  # (n_series, n_cal, horizon)

        n_eff = np.sum(~np.isnan(all_scores), axis=1)
        min_samples = int(_ceil(1.0 / self.alpha))
        bad = np.flatnonzero(n_eff.min(axis=1) < min_samples)
        if bad.size:
            idx = [int(i) for i in bad]
            raise InsufficientCalibrationSamples(
                f"Series {idx} have fewer than ceil(1/alpha) = {min_samples} "
                f"effective calibration samples for alpha={self.alpha}.",
                series_indices=idx,
            )

        dtype = all_scores.dtype
        n_series, n_cal, horizon = preds.shape
        K = self.n_experts
        gammas_arr = np.asarray(self.gammas, dtype=dtype).reshape(K, 1, 1)

        if self.interval_clip_lower is None:
            truth_std = float(np.nanstd(obs))
            clip_lower = float(np.nanmin(obs) - 5 * truth_std)
            clip_upper = float(np.nanmax(obs) + 5 * truth_std)
        else:
            assert self.interval_clip_upper is not None  # validated by _validate_clips
            clip_lower = self.interval_clip_lower
            clip_upper = self.interval_clip_upper

        alpha_t = np.full((K, n_series, horizon), self.alpha, dtype=dtype)
        aggregator_lower = self._new_aggregator(K, n_series, horizon)
        aggregator_upper = self._new_aggregator(K, n_series, horizon)
        beta_lower = self.alpha / 2.0
        beta_upper = 1.0 - self.alpha / 2.0

        has_past = np.zeros((n_series, horizon), dtype=bool)
        # One shared sorted prefix; the expert axis is batched into the gather.
        prefix = ExpandingSortedScores(n_series, horizon, dtype, capacity=n_cal)
        for t in range(n_cal):
            point_t = preds[:, t, :]  # (n_series, horizon)
            truth_t = obs[:, t, :]
            valid = ~np.isnan(all_scores[:, t, :])  # (n_series, horizon)
            first_valid = valid & ~has_past

            if t > 0:
                levels = _aci_levels(alpha_t)  # (K, n_series, horizon)
                thresholds = prefix.quantile(levels)
            else:
                # No history yet: every valid cell takes the first_valid
                # degenerate path below, and invalid cells are masked out of both
                # aggregator updates and the alpha_t step, so the value is inert.
                thresholds = np.zeros((K, n_series, horizon), dtype=dtype)
            # An empty-set expert, and a cell's first valid step, get a point
            # interval at the forecast.
            empty = _aci_saturation(alpha_t) < 0  # (K, n_series, horizon)
            lower, upper = _expert_bounds(
                point_t[None, :, :],
                thresholds,
                empty | first_valid[None, :, :],
                clip_lower,
                clip_upper,
            )

            losses_lower = _pinball_loss(truth_t[None, :, :], lower, beta_lower)
            losses_upper = _pinball_loss(truth_t[None, :, :], upper, beta_upper)
            # NaN steps must not contribute to any cell's cumulative loss.
            keep = valid[None, :, :]
            aggregator_lower.update(np.where(keep, losses_lower, 0.0))
            aggregator_upper.update(np.where(keep, losses_upper, 0.0))

            missed = (truth_t[None, :, :] < lower) | (truth_t[None, :, :] > upper)
            # The empty set misses whatever the truth, even on a tie with the
            # point the interval is emitted at.
            missed = (missed | empty).astype(dtype)
            step = gammas_arr * (self.alpha - missed)
            alpha_t = np.where(keep, alpha_t + step, alpha_t)
            has_past = has_past | valid
            # Push AFTER the thresholds: the prefix must stay all_scores[:, :t, :].
            prefix.push(all_scores[:, t, :])

        n_calibration_samples = n_eff.min(axis=1)
        return AgACIForecaster(
            method_=self._snapshot(),
            score_fn_=score_fn,
            n_calibration_samples_=n_calibration_samples,
            calibration_=CalibrationResult(
                n_calibration_samples=n_calibration_samples.copy(),
                diagnostics={
                    "gammas": self.gammas,
                    "alpha_t_per_expert": alpha_t.copy(),
                },
            ),
            alpha_t_per_expert_=alpha_t,
            aggregator_lower_=aggregator_lower,
            aggregator_upper_=aggregator_upper,
            interval_clip_lower_=clip_lower,
            interval_clip_upper_=clip_upper,
            scores_=all_scores,
            n_observations_=n_cal,
        )


class AgACIForecaster(ConformalForecaster):
    """
    A fitted :class:`AggregatedAdaptiveConformalInference`. ``update`` mutates it in place.

    Attributes
    ----------
    method_ : AggregatedAdaptiveConformalInference
    alpha_t_per_expert_ : NDArray, shape (n_experts, n_series, horizon)
        Per-expert time-varying miscoverage rate.
    aggregator_lower_, aggregator_upper_ : EWA
        The two independent aggregators, deep-copied from the method's
        prototype at fit and never shared with it.
    interval_clip_lower_, interval_clip_upper_ : float
        The *resolved* clips actually applied to emitted bounds — equal to the
        config values when those were given, otherwise computed at fit from the
        calibration truths.
    n_observations_ : int
        Number of windows seen so far (grows by one per :meth:`update`).
    scores_ : NDArray, shape (n_series, n_windows, horizon)
        Nonconformity scores retained for the online ``update`` step.
    """

    method_: AggregatedAdaptiveConformalInference
    alpha_t_per_expert_: NDArray[np.floating]
    aggregator_lower_: EWA
    aggregator_upper_: EWA
    interval_clip_lower_: float
    interval_clip_upper_: float
    n_observations_: int
    scores_: NDArray[np.floating]

    def _per_expert_bounds(
        self,
        point_squeezed: NDArray[np.floating],
        scores: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return ``(lower, upper)`` per expert, both shape ``(K, n_series, horizon)``."""
        levels = _aci_levels(self.alpha_t_per_expert_)
        # Cast to the point dtype, as the pre-allocated array used to do.
        thresholds = masked_quantile(scores, levels).astype(point_squeezed.dtype, copy=False)
        return _expert_bounds(
            point_squeezed[None, :, :],
            thresholds,
            self._empty_experts(),
            self.interval_clip_lower_,
            self.interval_clip_upper_,
        )

    def _empty_experts(self) -> NDArray[np.bool_]:
        """Experts asking for the empty set (``alpha_t >= 1``), shape (K, n_series, horizon)."""
        empty: NDArray[np.bool_] = _aci_saturation(self.alpha_t_per_expert_) < 0
        return empty

    def interval(
        self,
        predictions: Forecast,
        *,
        X: Exog | None = None,
    ) -> Interval:
        """
        Aggregate per-expert bounds for new predictions.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_samples, horizon)
        X : Exog or None
            Ignored by this method.

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)

        Raises
        ------
        ValueError
            If predictions shape (n_series, horizon) does not match fitted state.
        """
        preds = as_forecast(predictions, "predictions")
        pred_shape = (preds.shape[0], preds.shape[2])
        state_shape = (self.alpha_t_per_expert_.shape[1], self.alpha_t_per_expert_.shape[2])
        if pred_shape != state_shape:
            raise ValueError(
                f"predictions shape (n_series, horizon) = {pred_shape} does not "
                f"match fitted state shape {state_shape}"
            )
        dtype = preds.dtype
        levels = _aci_levels(self.alpha_t_per_expert_)
        # (K, n_series, horizon) in one batched gather over the expert axis.
        thresholds = masked_quantile(self.scores_, levels).astype(dtype, copy=False)
        # (K, n_series, n_samples, horizon)
        lower, upper = _expert_bounds(
            preds[None, :, :, :],
            thresholds[:, :, None, :],
            self._empty_experts()[:, :, None, :],
            self.interval_clip_lower_,
            self.interval_clip_upper_,
        )
        # Aggregator weights are float64; cast to preserve the input dtype.
        w_lower = self.aggregator_lower_.weights().astype(dtype)
        w_upper = self.aggregator_upper_.weights().astype(dtype)
        agg_lower = (w_lower[:, :, None, :] * lower).sum(axis=0)
        agg_upper = (w_upper[:, :, None, :] * upper).sum(axis=0)
        # A point outside [clip_lower_, clip_upper_] can invert the clipped
        # per-expert bounds (e.g. point < clip_lower_ => lower = clip_lower_ >
        # point + thr = upper); the convex combination inherits it. Collapse to
        # a degenerate interval rather than emit lower > upper.
        agg_lower = np.minimum(agg_lower, agg_upper)
        result: Interval = np.stack([agg_lower, agg_upper], axis=-1)
        return result

    def _update(
        self,
        prediction: Forecast,
        truth: Forecast,
        *,
        X: Exog | None = None,
    ) -> None:
        """
        Roll per-expert ``alpha_t`` and both aggregators forward one step.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, 1, horizon)
        truth : Forecast, shape (n_series, 1, horizon)
        X : Exog or None
            Ignored by this method.

        Raises
        ------
        ValueError
            If ``prediction``/``truth`` do not have shape ``(n_series, 1, horizon)``.
        """
        method = self.method_
        n_series, _, horizon = self.scores_.shape
        p = as_forecast(prediction, "prediction")
        t = as_forecast(truth, "truth")
        expected = (n_series, 1, horizon)
        if p.shape != expected or t.shape != expected:
            raise ValueError(
                f"prediction and truth must have shape {expected}, got {p.shape} and {t.shape}"
            )
        dtype = self.scores_.dtype
        point_sq = p[:, 0, :]
        truth_sq = t[:, 0, :]
        empty = self._empty_experts()  # read before alpha_t moves
        lower, upper = self._per_expert_bounds(point_sq, self.scores_)
        beta_lower = method.alpha / 2.0
        beta_upper = 1.0 - method.alpha / 2.0
        self.aggregator_lower_.update(_pinball_loss(truth_sq[None, :, :], lower, beta_lower))
        self.aggregator_upper_.update(_pinball_loss(truth_sq[None, :, :], upper, beta_upper))

        missed = (truth_sq[None, :, :] < lower) | (truth_sq[None, :, :] > upper)
        missed = (missed | empty).astype(dtype)  # the empty set always misses
        gammas_arr = np.asarray(method.gammas, dtype=dtype).reshape(method.n_experts, 1, 1)
        self.alpha_t_per_expert_ = self.alpha_t_per_expert_ + gammas_arr * (method.alpha - missed)

        new_score = self.score_fn_.score(p, t)
        self.scores_ = np.concatenate([self.scores_, new_score], axis=1)
        self.n_observations_ += 1
