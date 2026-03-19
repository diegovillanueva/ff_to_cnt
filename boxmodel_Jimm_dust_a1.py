#!/usr/bin/env python3
"""
Box model for Jimm_dust_a1: immersion freezing nucleation rate for fine dust (accumulation mode)
from hetfrz_classnuc.F90 in CESM2/CAM (original code at commit a156735c).

Implements both the single-alpha and alpha-PDF contact angle models from
Wang et al. (2014, ACP), doi:10.5194/acp-14-10411-2014.

  Eq. 1: J_het = A' * r_N^2 / sqrt(f) * exp((-dg# - f*dg0) / (kT))
  Eq. 2: f = (2+m)(1-m)^2 / 4,  m = cos(alpha)
  Eq. 3: p(alpha) = 1/(alpha*sigma*sqrt(2*pi)) * exp(-(ln(alpha)-ln(mu))^2/(2*sigma^2))
  Eq. 4: ff_pdf = 1 - integral_0^pi p(alpha) * exp(-J_imm(T,alpha)*dt) d_alpha
         ff_single = 1 - exp(-J_imm(T, alpha_0) * dt)

All internal constants are hardcoded from the Fortran source.
External inputs that the user must provide are clearly marked below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# =============================================================================
# PHYSICAL CONSTANTS (from hetfrz_classnuc.F90 local parameters)
# =============================================================================
PI = np.pi
KBOLTZ = 1.38e-23          # Boltzmann constant [J/K]
HPLANCK = 6.63e-34         # Planck constant [J s]
RHPLANCK = 1.0 / HPLANCK   # 1/h [s/J]
AMU = 1.66053886e-27        # atomic mass unit [kg]
N1 = 1.0e19                # water molecules per unit area of substrate [m-2]
MWH2O = 18.016              # molecular weight of water [g/mol]
TMELT = 273.15              # melting point of ice [K]

# =============================================================================
# FREEZING PARAMETERS (Wang et al., 2014) for dust immersion
# =============================================================================
THETA_IMM_DUST = 46.0       # contact angle [deg]
DGA_IMM_DUST = 14.75e-20    # activation energy [J]

# =============================================================================
# PDF-theta parameters (from hetfrz_classnuc_init_pdftheta)
# =============================================================================
PDF_N_THETA = 301
I1 = 53 - 1  # convert to 0-based indexing (Fortran i1=53)
I2 = 113 - 1  # convert to 0-based indexing (Fortran i2=113)
IMM_DUST_VAR_THETA = 0.01  # variance of log-normal PDF


def svp_water_goff_gratch(t: float) -> float:
    """Saturation vapor pressure over liquid water [Pa] — Goff & Gratch (1946)."""
    tboil = 373.16
    es = 10.0 ** (
        -7.90298 * (tboil / t - 1.0)
        + 5.02808 * np.log10(tboil / t)
        - 1.3816e-7 * (10.0 ** (11.344 * (1.0 - t / tboil)) - 1.0)
        + 8.1328e-3 * (10.0 ** (-3.49149 * (tboil / t - 1.0)) - 1.0)
        + np.log10(1013.246)
    ) * 100.0  # hPa -> Pa
    return es


def svp_ice_goff_gratch(t: float) -> float:
    """Saturation vapor pressure over ice [Pa] — Goff & Gratch (1946)."""
    h2otrip = 273.16
    es = 10.0 ** (
        -9.09718 * (h2otrip / t - 1.0)
        - 3.56654 * np.log10(h2otrip / t)
        + 0.876793 * (1.0 - t / h2otrip)
        + np.log10(6.1071)
    ) * 100.0  # hPa -> Pa
    return es


def init_pdf_theta() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Initialize PDF-theta arrays (replicates hetfrz_classnuc_init_pdftheta)."""
    theta_min = PI / 180.0
    theta_max = 179.0 / 180.0 * PI
    imm_dust_mean_theta = THETA_IMM_DUST / 180.0 * PI

    pdf_d_theta = (179.0 - 1.0) / 180.0 * PI / (PDF_N_THETA - 1)

    x1_imm = (np.log(theta_min) - np.log(imm_dust_mean_theta)) / (np.sqrt(2.0) * IMM_DUST_VAR_THETA)
    x2_imm = (np.log(theta_max) - np.log(imm_dust_mean_theta)) / (np.sqrt(2.0) * IMM_DUST_VAR_THETA)
    norm_theta_imm = (erf(x2_imm) - erf(x1_imm)) * 0.5

    dim_theta = np.zeros(PDF_N_THETA)
    pdf_imm_theta = np.zeros(PDF_N_THETA)
    dim_f_imm_dust_a1 = np.zeros(PDF_N_THETA)

    for i in range(I1, I2 + 1):
        # Fortran: dim_theta(i) = 1/180*pi + (i-1)*pdf_d_theta
        # Fortran (i-1) = Python i (since we shifted by -1)
        dim_theta[i] = 1.0 / 180.0 * PI + i * pdf_d_theta
        pdf_imm_theta[i] = (
            np.exp(
                -((np.log(dim_theta[i]) - np.log(imm_dust_mean_theta)) ** 2)
                / (2.0 * IMM_DUST_VAR_THETA**2)
            )
            / (dim_theta[i] * IMM_DUST_VAR_THETA * np.sqrt(2 * PI))
            / norm_theta_imm
        )

    for i in range(I1, I2 + 1):
        m = np.cos(dim_theta[i])
        dim_f_imm_dust_a1[i] = (2 + m) * (1 - m) ** 2 / 4.0

    return dim_theta, pdf_imm_theta, dim_f_imm_dust_a1, pdf_d_theta


def compute_thermo(
    t: float,
    supersatice: float,
    aw: float = 1.0,
) -> dict:
    """Compute thermodynamic properties needed for immersion freezing.

    Matches the original hetfrz_classnuc_calc (commit a156735c):
    - Uses t directly (no apparent-temperature ramp)
    - Uses supersatice directly (no ramp)
    - Water activity check: if aw*supersatice <= 1, freezing is suppressed
    """
    tc = t - TMELT

    # Ice properties (functions of tc)
    rhoice = 916.7 - 0.175 * tc - 5.0e-4 * tc**2
    vwice = MWH2O * AMU / rhoice
    sigma_iw = (28.5 + 0.25 * tc) * 1e-3  # [J/m2]

    # Critical germ radius — default (no solute effect)
    rgimm = 2 * vwice * sigma_iw / (KBOLTZ * t * np.log(supersatice))

    # Water activity check (solute effect / freezing point depression)
    do_dst1 = True
    if aw * supersatice > 1.0:
        rgimm = 2 * vwice * sigma_iw / (KBOLTZ * t * np.log(aw * supersatice))
    else:
        do_dst1 = False

    # Homogeneous energy of germ formation
    dg0 = 4 * PI / 3.0 * sigma_iw * rgimm**2

    # Prefactor A'
    A_prime = N1 * (
        (vwice * RHPLANCK) / (rgimm**3)
        * np.sqrt(3.0 / PI * KBOLTZ * t * dg0)
    )

    return {
        "tc": tc,
        "supersatice": supersatice,
        "do_dst1": do_dst1,
        "rgimm": rgimm,
        "dg0": dg0,
        "A_prime": A_prime,
    }


def compute_ff_single(
    t: float,
    deltat: float,
    r_dust: float = 0.15e-6,
    alpha_deg: float = THETA_IMM_DUST,
    aw: float = 1.0,
) -> float:
    """Frozen fraction using a single contact angle.

    ff_single = 1 - exp(-J_imm(T, alpha) * dt)
    Matches the .not. pdf_imm_in branch of the original Fortran.
    """
    result = compute_Jimm_dust_a1(t, r_dust_a1=r_dust, aw_dst1=aw)
    if not result["do_dst1"]:
        return 0.0

    # alpha_rad = alpha_deg / 180.0 * PI
    # m = np.cos(alpha_rad)
    # f = (2 + m) * (1 - m) ** 2 / 4.0
    #
    # J = (
    #     thermo["A_prime"]
    #     * r_dust**2
    #     / np.sqrt(f)
    #     * np.exp((-DGA_IMM_DUST - f * thermo["dg0"]) / (KBOLTZ * t))
    # )
    # J = max(J, 0.0)

    J = result["Jimm_at_pdf_peak [s-1]"]

    if t > 263.15:
        return 0.0
    return 1.0 - np.exp(-J * deltat)


def compute_ff_pdf(
    t: float,
    deltat: float,
    r_dust: float = 0.15e-6,
    mu_deg: float = THETA_IMM_DUST,
    sigma: float = IMM_DUST_VAR_THETA,
    aw: float = 1.0,
) -> float:
    """Frozen fraction using the alpha-PDF model.

    ff_pdf = 1 - integral p(alpha) * exp(-J_imm(T,alpha)*dt) d_alpha
    Matches the pdf_imm_in branch of the original Fortran.
    """
    result = compute_Jimm_dust_a1(t, r_dust_a1=r_dust, aw_dst1=aw)
    if not result["do_dst1"]:
        return 0.0

    dim_Jimm = result["dim_Jimm_dust_a1"]
    pdf_imm_theta = result["pdf_imm_theta"]
    pdf_d_theta = result["pdf_d_theta"]

    # # Compute J_imm at each theta bin
    # dim_Jimm = np.zeros(PDF_N_THETA)
    # for i in range(I1, I2 + 1):
    #     dim_Jimm[i] = (
    #         thermo["A_prime"]
    #         * r_dust**2
    #         / np.sqrt(dim_f[i])
    #         * np.exp((-DGA_IMM_DUST - dim_f[i] * thermo["dg0"]) / (KBOLTZ * t))
    #     )
    #     dim_Jimm[i] = max(dim_Jimm[i], 0.0)

    # Trapezoidal integration of survival fraction
    survival = 0.0
    for i in range(I1, I2):
        survival += 0.5 * (
            pdf_imm_theta[i] * np.exp(-dim_Jimm[i] * deltat)
            + pdf_imm_theta[i + 1] * np.exp(-dim_Jimm[i + 1] * deltat)
        ) * pdf_d_theta

    # Original Fortran clamps survival > 0.99 to 1.0
    if survival > 0.99:
        survival = 1.0

    if t > 263.15:
        return 0.0
    return max(0.0, 1.0 - survival)


def compute_Jimm_dust_a1(
    t: float,
    r_dust_a1: float = 0.15e-6,
    aw_dst1: float = 1.0,
) -> dict:
    """Compute Jimm_dust_a1 nucleation rate for dust accumulation mode.

    Matches hetfrz_classnuc_calc from commit a156735c.

    Parameters (EXTERNAL INPUTS):
    ----------
    t : float
        Temperature [K]
    r_dust_a1 : float, optional
        Mass-mean radius of dust accumulation mode [m]. Default 0.15e-6 (150 nm).
    aw_dst1 : float, optional
        Water activity for dust_a1 [ ]. Default 1.0.
    """
    supersatice = svp_water_goff_gratch(t) / svp_ice_goff_gratch(t)
    tc = t - TMELT
    rhoice = 916.7 - 0.175 * tc - 5.0e-4 * tc**2
    vwice = MWH2O * AMU / rhoice
    sigma_iw = (28.5 + 0.25 * tc) * 1e-3

    # Critical germ radius — default (no solute)
    rgimm = 2 * vwice * sigma_iw / (KBOLTZ * t * np.log(supersatice))
    rgimm_dust_a1 = rgimm

    # Water activity check
    do_dst1 = True
    if aw_dst1 * supersatice > 1.0:
        rgimm_dust_a1 = 2 * vwice * sigma_iw / (KBOLTZ * t * np.log(aw_dst1 * supersatice))
    else:
        do_dst1 = False

    # Homogeneous energy of germ formation
    dg0imm_dust_a1 = 4 * PI / 3.0 * sigma_iw * rgimm_dust_a1**2

    # Prefactor A
    Aimm_dust_a1 = N1 * (
        (vwice * RHPLANCK) / (rgimm_dust_a1**3)
        * np.sqrt(3.0 / PI * KBOLTZ * t * dg0imm_dust_a1)
    )

    # PDF-theta: compute J at each theta bin
    dim_theta, pdf_imm_theta, dim_f, pdf_d_theta = init_pdf_theta()

    dim_Jimm_dust_a1 = np.zeros(PDF_N_THETA)
    for i in range(I1, I2 + 1):
        dim_Jimm_dust_a1[i] = (
            Aimm_dust_a1
            * r_dust_a1**2
            / np.sqrt(dim_f[i])
            * np.exp((-DGA_IMM_DUST - dim_f[i] * dg0imm_dust_a1) / (KBOLTZ * t))
        )
        dim_Jimm_dust_a1[i] = max(dim_Jimm_dust_a1[i], 0.0)

    # Peak Jimm at PDF mean for reporting
    mean_theta_rad = THETA_IMM_DUST / 180.0 * PI
    i_peak = np.argmin(np.abs(dim_theta[I1:I2 + 1] - mean_theta_rad)) + I1
    Jimm_at_peak = dim_Jimm_dust_a1[i_peak]

    return {
        "tc": tc,
        "rhoice": rhoice,
        "vwice": vwice,
        "sigma_iw": sigma_iw,
        "rgimm_dust_a1": rgimm_dust_a1,
        "dg0imm_dust_a1": dg0imm_dust_a1,
        "Aimm_dust_a1": Aimm_dust_a1,
        "Jimm_at_pdf_peak [s-1]": Jimm_at_peak,
        "dim_Jimm_dust_a1": dim_Jimm_dust_a1,
        "pdf_imm_theta": pdf_imm_theta,
        "pdf_d_theta": pdf_d_theta,
        "do_dst1": do_dst1,
    }


# =============================================================================
# Temperature sweep: compare single-alpha vs alpha-PDF frozen fractions
# =============================================================================
if __name__ == "__main__":

    # =========================================================================
    # >>> EXTERNAL INPUTS — CHANGE THESE FOR YOUR CASE <<<
    # =========================================================================
    R_DUST_A1 = 0.15e-6    # 300 nm diameter monodisperse (as in Fig. 2 of Wang+2014)
    DELTAT = 1800.0         # timestep [s]  (30 min, CAM5 default)
    DELTAT_CFDC = 10.0      # CFDC residence time [s] (~5-20 s)
    AW = 1.0                # water activity (1.0 = no solute effect)

    # Single-alpha parameters (Table 1: dust immersion, DeMott et al. 2011)
    ALPHA_SINGLE_DEG = 46.0  # contact angle [deg]

    # Alpha-PDF parameters (Table 2: CSU106)
    MU_PDF_DEG = 46.0        # mean contact angle [deg]
    SIGMA_PDF = 0.01         # standard deviation of log-normal
    # =========================================================================

    # Temperature range (matching paper Fig. 1 / Fig. 2)
    tc_range = np.linspace(-40.0, 0.0, 200)
    t_range = tc_range + TMELT

    ff_single = np.zeros_like(t_range)
    ff_pdf = np.zeros_like(t_range)
    ff_single_10s = np.zeros_like(t_range)
    ff_pdf_10s = np.zeros_like(t_range)

    for i, t in enumerate(t_range):
        ff_single[i] = compute_ff_single(
            t, DELTAT, r_dust=R_DUST_A1, alpha_deg=ALPHA_SINGLE_DEG, aw=AW,
        )
        ff_pdf[i] = compute_ff_pdf(
            t, DELTAT, r_dust=R_DUST_A1, mu_deg=MU_PDF_DEG, sigma=SIGMA_PDF, aw=AW,
        )
        ff_single_10s[i] = compute_ff_single(
            t, DELTAT_CFDC, r_dust=R_DUST_A1, alpha_deg=ALPHA_SINGLE_DEG, aw=AW,
        )
        ff_pdf_10s[i] = compute_ff_pdf(
            t, DELTAT_CFDC, r_dust=R_DUST_A1, mu_deg=MU_PDF_DEG, sigma=SIGMA_PDF, aw=AW,
        )

    # --- Print summary at a few temperatures ---
    print("=" * 70)
    print("  Frozen fraction comparison: single-alpha vs alpha-PDF")
    print(f"  r_dust = {R_DUST_A1*1e6:.3f} um, dt = {DELTAT:.0f} s")
    print(f"  single-alpha: alpha = {ALPHA_SINGLE_DEG}")
    print(f"  alpha-PDF: mu = {MU_PDF_DEG}, sigma = {SIGMA_PDF}")
    print("=" * 70)
    print(f"  {'T [C]':>8s}  {'ff_single':>12s}  {'ff_pdf':>12s}")
    print("-" * 70)
    for tc_val in [-35, -30, -25, -20, -15, -10]:
        idx = np.argmin(np.abs(tc_range - tc_val))
        print(f"  {tc_range[idx]:8.1f}  {ff_single[idx]:12.4e}  {ff_pdf[idx]:12.4e}")
    print("=" * 70)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Replace zeros with NaN for clean log plotting
    ff_single_plot = np.where(ff_single > 0, ff_single, np.nan)
    ff_pdf_plot = np.where(ff_pdf > 0, ff_pdf, np.nan)
    ff_single_10s_plot = np.where(ff_single_10s > 0, ff_single_10s, np.nan)
    ff_pdf_10s_plot = np.where(ff_pdf_10s > 0, ff_pdf_10s, np.nan)

    # dt = 1800 s curves (CAM5 default)
    ax.semilogy(tc_range, ff_single_plot, "r-", linewidth=2,
                label=f"single-$\\alpha$ ($\\alpha$={ALPHA_SINGLE_DEG}, dt={DELTAT:.0f}s)")
    ax.semilogy(tc_range, ff_pdf_plot, "b--", linewidth=2,
                label=f"$\\alpha$-PDF ($\\mu$={MU_PDF_DEG}, $\\sigma$={SIGMA_PDF}, dt={DELTAT:.0f}s)")

    # dt = 10 s curves (CFDC residence time)
    ax.semilogy(tc_range, ff_single_10s_plot, "r-", linewidth=1.5, alpha=0.5,
                label=f"single-$\\alpha$ (dt={DELTAT_CFDC:.0f}s)")
    ax.semilogy(tc_range, ff_pdf_10s_plot, "b--", linewidth=1.5, alpha=0.5,
                label=f"$\\alpha$-PDF (dt={DELTAT_CFDC:.0f}s)")

    ax.set_xlabel("Temperature [C]", fontsize=13)
    ax.set_ylabel("Active Fraction", fontsize=13)
    ax.set_title(
        f"Wang et al. (2014) -- Immersion freezing of dust\n"
        f"$r_N$ = {R_DUST_A1*1e9:.0f} nm",
        fontsize=13,
    )
    ax.set_xlim(-40, 0)
    ax.set_ylim(1e-7, 1e-1)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    outpath = Path(__file__).parent / "ff_single_vs_pdf.png"
    fig.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")
