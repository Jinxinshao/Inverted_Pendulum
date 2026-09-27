"""Free-swing analysis of the REAL potentiometer signal (dead zone, wrap, glitches).

Why a dedicated pipeline
------------------------
On this rig the WDD35D4 (345 deg electrical of 360) is mounted so that the
hanging position lies INSIDE the 15 deg dead zone. The first real recording
(2026-09-27, release from 90.8 deg) showed what the raw stream looks like:

* one side of the swing reads 4092 -> ~3000, the other side 0 -> ~900
  (it wrapped around the resistive track);
* every pass through the bottom gives a plateau at ~4092 (end terminal), a
  plateau at 0..1 (other terminal) and 1-5 GLITCH samples anywhere in
  between (e.g. "1 1 1029 2129 4091") - the wiper bridging the gap;
* the rest reading 4091 is clipped: the true bottom is ~7 counts further.

Glitches in the mid range look like near-upright angles; any extremum search
on the raw angle then finds dozens of false extrema (the old method A1 gave
omega0 = 11.8 rad/s). The pipeline below:

1. unwraps the reading around the rest value ``u = h + ((adc - h) mod N)``,
2. rejects samples outside the valid track and glitches (Hampel test against
   a 9-sample running median, dilated by 2 samples),
3. finds ONE turning point per valid segment (local quartic fit),
4. calibrates the sensor FROM THE SWING ITSELF: in every half cycle the rod
   passes the bottom exactly half-way between the two turning points, and the
   exact pendulum solution  phi(tau) = 2 asin(k sn(omega0 tau | k^2)),
   k = sin(A/2), gives the angle of every sample. Linear least squares on
       adc = b + K phi - N w          (w = 1 for wrapped samples)
   returns the bottom reading b, the gain K [counts/rad] and the wrap length N
   (track + dead zone), i.e. the dead-zone position and width,
5. small-amplitude period from full cycles with the elliptic-integral
   amplitude correction  T(A) = T0 2K(sin^2(A/2))/pi,
6. damping from the half-cycle energy balance with three candidate loss
   mechanisms and model selection by AIC:
       viscous   c phi_d          -> loss c * S1(A),  S1 = int phi_d^2 dt
       Coulomb   gamma sgn(phi_d) -> loss gamma * (A_j + A_j+1)
       air drag  d phi_d|phi_d|   -> loss d * S3(A),  S3 = int |phi_d|^3 dt
                                                     = 4 alpha (sin A - A cos A)
   (all linear in the coefficients; non-negative least squares).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import nnls
from scipy.signal import find_peaks
from scipy.special import ellipj

from .identification import half_cycle_action, period_factor

N_TURN_NOMINAL = 4096 * 360.0 / 345.0     # counts of one full wiper turn (datasheet, 345 deg electrical)
K_NOMINAL = 4095.0 / math.radians(345.0)  # counts per rad along the track (680.1)


@dataclass
class SwingCalibration:
    """adc = b + K phi - N w  (phi from the BOTTOM, w = 1 on the wrapped side)."""

    b: float = 4092.0
    K: float = K_NOMINAL
    N: float = N_TURN_NOMINAL
    identified: bool = False

    def phi(self, u: np.ndarray, wrapped: np.ndarray) -> np.ndarray:
        return (np.asarray(u, float) - self.b + (self.N - N_TURN_NOMINAL) * np.asarray(wrapped, float)) / self.K

    def describe(self, adc_hi_end: float = 4092.0, adc_lo_end: float = 0.0) -> dict:
        """Dead zone and implied upright reading (valid when the dead zone is at the bottom)."""
        return {
            "adc_at_bottom": self.b,
            "counts_per_rad": self.K,
            "electrical_deg": math.degrees(4095.0 / self.K),
            "wrap_counts": self.N,
            "dead_zone_deg": math.degrees((self.N - 4095.0) / self.K),
            "high_end_deg_from_bottom": math.degrees((adc_hi_end - self.b) / self.K),
            "low_end_deg_from_bottom": math.degrees((adc_lo_end + self.N - self.b) / self.K),
            "implied_upright_adc": self.b - math.pi * self.K,
        }


@dataclass
class DampingModel:
    name: str
    c: float
    gamma: float
    d: float
    rms: float
    aic: float


@dataclass
class SwingAnalysis:
    omega0: float
    T0: float
    T0_std: float                 # scatter of the single-cycle estimates [s]
    omega0_se: float              # standard error of omega0 [rad/s]
    calibration: SwingCalibration
    damping: DampingModel         # selected by AIC
    candidates: list[DampingModel]
    t_ext: np.ndarray             # turning-point times
    amp: np.ndarray               # |amplitude| [rad]
    side: np.ndarray              # +1 / -1
    t: np.ndarray                 # swing samples (time)
    phi: np.ndarray               # angle from the bottom [rad], NaN = rejected
    stats: dict = field(default_factory=dict)
    per_side: dict = field(default_factory=dict)

    # the fit interface shared with DecayFit / LSFit (omega0 above)
    @property
    def viscous_c(self) -> float:
        return self.damping.c

    @property
    def coulomb_gamma(self) -> float:
        return self.damping.gamma

    @property
    def quad_d(self) -> float:
        return self.damping.d

    def summary(self) -> str:
        return (f"turning points + energy balance: omega0={self.omega0:.4f} rad/s (T0={self.T0:.4f} s), "
                f"damping '{self.damping.name}': c={self.viscous_c:.4f} 1/s, gamma={self.coulomb_gamma:.4f} rad/s^2, "
                f"d={self.quad_d:.5f} 1/rad")

    def lines(self) -> list[str]:
        s, c = self.stats, self.calibration.describe()
        out = []
        if s.get("rest_ignored"):
            out.append(f"注意：静止段读数 {s['rest_given']:.0f} 离摆动中心太远（录制时杆没有自然下垂？），"
                       f"已改用摆动中心 {s['rest_adc']:.0f} 作参考")
        out += [
            f"数据: {s['n_swing']} 个摆动样本, {s['rate_hz']:.1f} Hz, 剔除 {s['rejected_pct']:.1f} %"
            f"（死区/端点 {s['n_deadzone']}，毛刺 {s['n_glitch']}），{s['n_crossings']} 次过死区",
            f"释放角 {math.degrees(self.amp[0]):.1f}°，{len(self.amp)} 个转折点，末振幅 {math.degrees(self.amp[-1]):.1f}°",
            f"传感器（由摆动自标定）: 最低点读数 {c['adc_at_bottom']:.1f}, {c['counts_per_rad']:.1f}"
            + (f" ± {s['K_se']:.1f}" if s.get('K_se') == s.get('K_se') else "（固定）")
            + f" LSB/rad (电气角 {c['electrical_deg']:.1f}°), 有效死区宽 {c['dead_zone_deg']:.1f}°, "
            f"高端饱和点在最低点 {c['high_end_deg_from_bottom']:+.1f}°, 低端起点 {c['low_end_deg_from_bottom']:+.1f}°",
            f"  → 按线性外推的直立读数 {c['implied_upright_adc']:.1f}",
            f"ω0 = {self.omega0:.4f} ± {self.omega0_se:.4f} rad/s (T0 = {self.T0:.4f} s，单周期散布 {self.T0_std * 1e3:.2f} ms)",
        ]
        for k, v in self.per_side.items():
            out.append(f"  只用 {k} 侧整周期: ω0 = {v:.4f} rad/s")
        out.append("阻尼模型（半周期能量平衡，NNLS，AIC 越小越好）:")
        for m in sorted(self.candidates, key=lambda m: m.aic):
            mark = " ←选用" if m is self.damping else ""
            out.append(f"  {m.name:<18s} c={m.c:.4f} 1/s  γ={m.gamma:.4f} rad/s²  d={m.d:.5f} 1/rad  "
                       f"rms={m.rms:.4f}  AIC={m.aic:.1f}{mark}")
        return out


# ---------------------------------------------------------------- helpers
REST_TOL_COUNTS = 400.0   # ~34 deg: a rest reading further than this from the swing centre is not "hanging"


def _circ_dist(a: float, b: float, n_turn: float = N_TURN_NOMINAL) -> float:
    d = (a - b) % n_turn
    return float(min(d, n_turn - d))


def swing_centre(adc: np.ndarray, lo: float = 8, hi: float = 4087, n_turn: float = N_TURN_NOMINAL) -> float:
    """Reading around which the rod swings (circular mean of the on-track samples of the
    second half of the record; the hand-held start is excluded). Accurate to ~10 deg -
    only used as the unwrap reference, the bottom itself is fitted later."""
    a = np.asarray(adc, float)
    a = a[len(a) // 2:]
    a = a[(a >= lo) & (a <= hi)]
    if len(a) < 10:
        return float(np.median(adc))
    ang = 2 * np.pi * a / n_turn
    c = math.atan2(float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang)))) % (2 * math.pi)
    return c * n_turn / (2 * math.pi)
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def robust_rest(adc: np.ndarray) -> tuple[float, float, int]:
    """Median, MAD-based std and number of outliers (> 20 counts) of a rest record."""
    a = np.asarray(adc, float)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    return med, mad, int(np.sum(np.abs(a - med) > 20))


def unwrap_reading(adc: np.ndarray, rest: float, n_turn: float = N_TURN_NOMINAL) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap around the rest reading; returns (u, wrapped flag in {-1, 0, 1})."""
    a = np.asarray(adc, float)
    u = rest + np.mod(a - rest + n_turn / 2, n_turn) - n_turn / 2
    return u, np.round((u - a) / n_turn)


def clean_samples(adc: np.ndarray, u: np.ndarray, lo: float = 8, hi: float = 4087,
                  hampel: float = 40.0, dilate: int = 2) -> tuple[np.ndarray, int, int]:
    """Valid-sample mask: on the track, not a glitch, not next to an invalid sample."""
    on_track = (adc >= lo) & (adc <= hi)
    med = median_filter(u, size=9, mode="nearest")
    ok = on_track & (np.abs(u - med) < hampel)
    n_dead, n_glitch = int(np.sum(~on_track)), int(np.sum(on_track & ~ok))
    bad = ~ok
    grown = bad.copy()
    for s in range(1, dilate + 1):
        grown[s:] |= bad[:-s]
        grown[:-s] |= bad[s:]
    return ~grown, n_dead, n_glitch


def turning_points(t: np.ndarray, u: np.ndarray, valid: np.ndarray, min_prom: float = 60.0,
                   min_sep: float = 0.25, hold_s: float = 0.15, half_win: float = 0.12):
    """One refined extremum per valid segment (more if a segment has several).

    Returns arrays (t_p, u_p, kind) with kind = +1 for a maximum of u, -1 for a
    minimum. An extremum whose plateau (within 4 counts) lasts longer than
    ``hold_s`` is the hand holding the rod before release and is dropped.
    """
    fs = 1.0 / float(np.median(np.diff(t)))
    tp, up, kind = [], [], []
    for a, b in _runs(valid):
        if b - a < 20:
            continue
        seg, ts = u[a:b], t[a:b]
        for sgn in (1, -1):
            idx, _ = find_peaks(sgn * seg, prominence=min_prom, distance=max(1, int(min_sep * fs)))
            idx = idx[(idx >= 8) & (idx <= len(seg) - 9)]  # needs samples on both sides for the fit
            for j in idx:
                w = int(min(half_win * fs, j, len(seg) - 1 - j))
                if w < 4:
                    continue
                x = ts[j - w:j + w + 1] - ts[j]
                y = sgn * seg[j - w:j + w + 1]
                co = np.polyfit(x, y, 4)
                r = np.roots(np.polyder(co))
                r = r[np.isreal(r)].real
                r = r[np.abs(r) < 0.5 * w / fs]
                x0 = float(r[np.argmin(np.abs(r))]) if len(r) else 0.0
                peak = float(np.polyval(co, x0))
                if np.sum(np.abs(sgn * seg - peak) < 4.0) > hold_s * fs:
                    continue  # hand hold before the release
                tp.append(ts[j] + x0)
                up.append(sgn * peak)
                kind.append(sgn)
    o = np.argsort(tp)
    return np.array(tp)[o], np.array(up)[o], np.array(kind)[o]


def _amplitudes(up: np.ndarray, wp: np.ndarray, cal: SwingCalibration) -> np.ndarray:
    return np.abs(cal.phi(up, wp))


def _t0_full_cycles(tp: np.ndarray, A: np.ndarray, sel: np.ndarray | None = None):
    per = tp[2:] - tp[:-2]
    aeff = 0.25 * (A[:-2] + 2 * A[1:-1] + A[2:])
    t0 = per / period_factor(aeff)
    m = np.ones(len(t0), bool) if sel is None else sel[:-2]
    t0 = t0[m]
    if len(t0) < 3:
        raise ValueError("fewer than 3 full cycles")
    med = float(np.median(t0))
    good = np.abs(t0 / med - 1.0) < 0.03
    return float(np.median(t0[good])), float(np.std(t0[good])), int(good.sum())


def calibrate_from_swing(t, adc_raw, w, valid, tp, up, wp, omega0, cal: SwingCalibration, fit_n: bool) -> SwingCalibration:
    """One pass: exact pendulum angle of every sample -> LS for the offsets (b, N), K fixed.

    The slope K is NOT taken from this regression: near the bottom the reading
    changes at K * 2 omega0 sin(A/2) with A = (u_p - b)/K, so K cancels to first
    order and only the waveform's nonlinearity is left - K and omega0 then trade
    off along a ridge (on the twin: K 0.6 % low, omega0 0.2 % high). K comes from
    the period-amplitude relation instead (:func:`fit_period_amplitude`).
    """
    A = _amplitudes(up, wp, cal)
    rows, ys = [], []
    for j in range(len(tp) - 1):
        if np.sign(cal.phi(up[j], wp[j])) == np.sign(cal.phi(up[j + 1], wp[j + 1])):
            continue  # not a half cycle (missed turning point)
        m = valid & (t > tp[j] + 0.02) & (t < tp[j + 1] - 0.02)
        if m.sum() < 5:
            continue
        # damping: the energy falls from E_j to E_j+1 during the half cycle, so the two
        # quarter periods differ (T(A) grows with A) and the amplitude seen by each
        # sample is between A_j and A_j+1 (adiabatic interpolation of the energy)
        q0 = 0.25 * (2 * math.pi / omega0) * float(period_factor(A[j]))
        q1 = 0.25 * (2 * math.pi / omega0) * float(period_factor(A[j + 1]))
        t_bot = 0.5 * ((tp[j] + q0) + (tp[j + 1] - q1))
        tau = t[m] - t_bot
        e0, e1 = 1 - math.cos(A[j]), 1 - math.cos(A[j + 1])
        e = e0 + (e1 - e0) * np.clip((t[m] - tp[j]) / (tp[j + 1] - tp[j]), 0, 1)
        k = np.sqrt(e / 2)  # sin(A/2) = sqrt((1 - cos A)/2)
        sn = ellipj(omega0 * tau, k * k)[0]
        direction = 1.0 if cal.phi(up[j], wp[j]) < 0 else -1.0
        phi = direction * 2 * np.arcsin(np.clip(k * sn, -1, 1))
        rows.append(np.column_stack([np.ones(m.sum()), phi, -w[m]]))
        ys.append(adc_raw[m])
    if not rows:
        return cal
    X, y = np.vstack(rows), np.concatenate(ys)
    y = y - cal.K * X[:, 1]                     # adc - K phi = b - N w
    if not fit_n or not np.any(X[:, 2]):
        b = float(np.median(y - cal.N * X[:, 2]))
        return SwingCalibration(b, cal.K, cal.N, True)
    Z = X[:, [0, 2]]
    th, *_ = np.linalg.lstsq(Z, y, rcond=None)
    res = y - Z @ th
    keep = np.abs(res) < 4 * np.std(res)
    th, *_ = np.linalg.lstsq(Z[keep], y[keep], rcond=None)
    return SwingCalibration(float(th[0]), cal.K, float(th[1]), True)


def fit_period_amplitude(tp, up, wp, cal: SwingCalibration, fit_k: bool = True):
    """T_full,i = T0 * f(A_eff,i(K)): small-amplitude period and the sensor slope K.

    With the right K the single-cycle T0 estimates show no trend with amplitude;
    a wrong K tilts them (f(A) - 1 ~ A^2/16). Needs a decay over a wide amplitude
    range (e.g. 90 -> 20 deg); otherwise K is kept. Returns (T0, K, se_T0, se_K, n).
    """
    from scipy.optimize import least_squares

    per = tp[2:] - tp[:-2]

    def aeff(K):
        A = np.abs((up + (cal.N - N_TURN_NOMINAL) * wp - cal.b) / K)
        return 0.25 * (A[:-2] + 2 * A[1:-1] + A[2:])

    a0 = aeff(cal.K)
    T0s = per / period_factor(a0)
    keep = np.abs(T0s / np.median(T0s) - 1) < 0.03
    wide = np.ptp(a0[keep]) > math.radians(40)
    if not (fit_k and wide):
        T0 = float(np.median(T0s[keep]))
        return T0, cal.K, float(np.std(T0s[keep]) / math.sqrt(keep.sum())), float("nan"), int(keep.sum())
    for _ in range(2):
        def res(x):
            return (per[keep] - x[0] * period_factor(aeff(x[1])[keep])) * 1e3

        sol = least_squares(res, [float(np.median(T0s[keep])), cal.K])
        r = sol.fun
        keep_new = keep.copy()
        keep_new[keep] = np.abs(r) < 4 * max(np.std(r), 1e-6)
        keep = keep_new
    J = sol.jac
    cov = np.linalg.inv(J.T @ J) * np.mean(sol.fun**2)
    se = np.sqrt(np.diag(cov))
    return float(sol.x[0]), float(sol.x[1]), float(se[0]), float(se[1]), int(keep.sum())


def fit_damping(A: np.ndarray, alpha: float, pairs: np.ndarray) -> list[DampingModel]:
    """Energy balance per half cycle A_j -> A_j+1; all 7 subsets of {c, gamma, d}."""
    a0, a1 = A[:-1][pairs], A[1:][pairs]
    am = 0.5 * (a0 + a1)
    y = -alpha * (np.cos(a0) - np.cos(a1))          # = energy lost per unit inertia
    s3 = 4 * alpha * (np.sin(am) - am * np.cos(am))
    X = np.column_stack([half_cycle_action(am, alpha), a0 + a1, s3])
    out = []
    names = {(0,): "粘性 c", (1,): "库仑 γ", (2,): "空气阻力 d", (0, 1): "粘性+库仑",
             (0, 2): "粘性+空气阻力", (1, 2): "库仑+空气阻力", (0, 1, 2): "粘性+库仑+空气阻力"}
    n = len(y)
    for cols, name in names.items():
        th, _ = nnls(X[:, cols], y)
        res = y - X[:, cols] @ th
        full = np.zeros(3)
        full[list(cols)] = th
        rms = float(np.sqrt(np.mean(res**2)))
        out.append(DampingModel(name, float(full[0]), float(full[1]), float(full[2]), rms,
                                n * math.log(max(rms, 1e-12) ** 2) + 2 * len(cols)))
    return out


def select_model(cands: list[DampingModel]) -> DampingModel:
    """Lowest AIC; a model with more terms must beat a simpler one by > 2 (and use all its terms)."""
    def nterms(m):
        return sum(v > 0 for v in (m.c, m.gamma, m.d))

    best = None
    for m in sorted(cands, key=lambda m: (m.aic, nterms(m))):
        if best is None or (m.aic < best.aic - 2 and nterms(m) > nterms(best)):
            best = m
    # prefer the simplest model within 2 AIC units of the best
    for m in sorted(cands, key=nterms):
        if m.aic <= best.aic + 2 and nterms(m) < nterms(best):
            best = m
            break
    return best


# ---------------------------------------------------------------- main entry
def analyse_swing(t: np.ndarray, adc: np.ndarray, rest_adc: float | None = None, lo: float = 8, hi: float = 4087,
                  min_amp_deg: float = 5.0, calibrate: bool = True, K0: float | None = None) -> SwingAnalysis:
    """Full pipeline on the SWING part of a recording (the rest part only gives ``rest_adc``).

    ``K0``: fix the sensor slope [counts/rad] instead of identifying it from the
    period-amplitude relation (use it for short / small swings).
    """
    t = np.asarray(t, float)
    adc = np.asarray(adc, float)
    # stale samples read in one burst at the start (serial backlog) carry no timing information
    k0 = 0
    while k0 < len(t) - 1 and t[k0 + 1] - t[k0] < 1e-6:
        k0 += 1
    t, adc = t[k0:], adc[k0:]
    # the unwrap needs a reference near the bottom. The "rest" record is only trusted if it
    # agrees with the centre of the swing itself (2026-09-27, 2nd recording: the rod was held
    # at 90 deg during the rest phase -> reference 928 instead of ~4092 -> omega0 = 6.4)
    centre = swing_centre(adc, lo, hi)
    rest_given = rest_adc
    if rest_adc is None or _circ_dist(rest_adc, centre) > REST_TOL_COUNTS:
        rest_adc = centre
    u, w = unwrap_reading(adc, rest_adc)
    valid, n_dead, n_glitch = clean_samples(adc, u, lo, hi)
    tp, up, kind = turning_points(t, u, valid, min_prom=math.radians(min_amp_deg) * K_NOMINAL)
    if len(tp) < 6:
        raise ValueError(f"only {len(tp)} turning points found - swing too short or too small?")
    wp = np.round(np.interp(tp, t, w))
    cal = SwingCalibration(b=rest_adc, K=K_NOMINAL if K0 is None else K0)
    A_guess = np.abs(cal.phi(up, wp))
    T0, T0_std, n_cyc = _t0_full_cycles(tp, A_guess)
    omega0 = 2 * math.pi / T0
    crossings = int(np.sum(np.diff(w[valid]) != 0))  # bottom passes through the dead zone
    fit_n = crossings >= 4
    se_T0, se_K = float("nan"), float("nan")
    if calibrate:
        for _ in range(8):
            cal = calibrate_from_swing(t, adc, w, valid, tp, up, wp, omega0, cal, fit_n)
            T0, K, se_T0, se_K, n_cyc = fit_period_amplitude(tp, up, wp, cal, fit_k=K0 is None)
            cal = SwingCalibration(cal.b, K, cal.N, True)
            omega0 = 2 * math.pi / T0
    A = _amplitudes(up, wp, cal)
    _, T0_std, _ = _t0_full_cycles(tp, A)
    side = np.sign(cal.phi(up, wp)).astype(int)
    alt = side[1:] != side[:-1]
    cands = fit_damping(A, omega0**2, alt)
    per_side = {}
    for s, name in ((1, "+"), (-1, "-")):
        m = side == s
        if m.sum() >= 4:
            ts_, As = tp[m], A[m]
            # full cycles on one side: effective amplitude uses the opposite extremum in between
            per = np.diff(ts_)
            mid = np.interp(0.5 * (ts_[:-1] + ts_[1:]), tp, A)
            t0s = per / period_factor(0.25 * (As[:-1] + 2 * mid + As[1:]))
            g = np.abs(t0s / np.median(t0s) - 1) < 0.03
            per_side[name] = 2 * math.pi / float(np.median(t0s[g]))
    phi = np.where(valid, cal.phi(u, w), np.nan)
    stats = {
        "n_swing": int(len(t)), "rate_hz": float((len(t) - 1) / (t[-1] - t[0])),
        "n_deadzone": n_dead, "n_glitch": n_glitch, "rejected_pct": 100.0 * float(np.mean(~valid)),
        "n_crossings": crossings, "n_full_cycles": n_cyc, "rest_adc": rest_adc, "skipped_backlog": k0,
        "rest_ignored": rest_given is not None and rest_given != rest_adc, "rest_given": rest_given,
    }
    stats["K_se"] = se_K
    return SwingAnalysis(omega0, T0, T0_std, omega0 * (se_T0 if se_T0 == se_T0 else T0_std / math.sqrt(max(n_cyc, 1))) / T0, cal,
                         select_model(cands), cands, tp, A, side, t, phi, stats, per_side)
