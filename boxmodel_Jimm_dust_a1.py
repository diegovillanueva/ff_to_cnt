#!/usr/bin/env python3
"""
Box model for Jimm_dust_a1: immersion freezing nucleation rate for fine dust (accumulation mode)
from hetfrz_classnuc.F90 in CESM2/CAM.

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
# PHYSICAL CONSTANTS (from CESM physconst / hetfrz_classnuc.F90)
# =============================================================================
PI = np.pi
KBOLTZ = 1.38e-23          # Boltzmann constant [J/K]
HPLANCK = 6.63e-34         # Planck constant [J s]
RHPLANCK = 1.0 / HPLANCK   # 1/h [s/J]
AMU = 1.66053886e-27        # atomic mass unit [kg]
N1 = 1.0e19                # water molecules per unit area of substrate [m-2]

# CESM standard physical constants
RAIR = 287.04               # dry air gas constant [J/(kg K)]
CPAIR = 1004.64             # specific heat of dry air [J/(kg K)]
RH2O = 461.505              # water vapor gas constant [J/(kg K)]
RHOH2O = 1000.0             # density of liquid water [kg/m3]
MWH2O = 18.016              # molecular weight of water [g/mol]
TMELT = 273.15              # melting point of ice [K]

# =============================================================================
# FREEZING PARAMETERS (Wang et al., 2014) for dust immersion
# =============================================================================
DGA_IMM_DUST = 14.75e-20    # activation energy [J]

# =============================================================================
# NAMELIST DEFAULTS (from micro_dv_nml.F90)
# =============================================================================
THETA_IMM_DUST = 46.0       # contact angle for dust_a1 immersion [deg]
LIMFACDU = 1.0              # max INP fraction (dust)
DU_BACKGROUND = 0.0         # background dust number [#/cm3]
DEPR_POINT_FRZ = True       # freezing point depression toggle
FRZ_ASSUME_RAMP_RHCE = False
TC_RAMP_TOP = -42.0         # cold T limit for apparent-T ramp [C]
TC_RAMP_BOT = -6.0          # warm T limit for apparent-T ramp [C]
TC_APPARENT_RAMP_TOP = -42.0
TC_APPARENT_RAMP_BOT = -6.0
HCLAS_MAX_T_IMM = 263.15    # warmest T for immersion freezing [K]
BACK_DROP_PERC_PER_HOUR = 0.0

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
    """Initialize PDF-theta arrays (replicates hetfrz_classnuc_init_pdftheta).

    Returns:
        dim_theta: contact angle array [rad]
        pdf_imm_theta: PDF values
        dim_f_imm_dust_a1: form factor f for each theta
        pdf_d_theta: theta spacing [rad]
    """
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
        # but i is 1-based in Fortran, so Fortran (i-1) = Python i (since we shifted by -1)
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
    aw: float = 1.0,
) -> dict:
    """Compute thermodynamic properties needed for J_het (Eq. 1).

    Factored out so both single-α and α-PDF models share the same calculation.
    """
    eswtr = svp_water_goff_gratch(t)
    esice = svp_ice_goff_gratch(t)
    supersatice = eswtr / esice

    tc = t - TMELT

    # Apparent temperature ramp
    tc_apparent = tc
    if tc >= TC_RAMP_TOP and tc <= TC_RAMP_BOT:
        tc_apparent = TC_APPARENT_RAMP_TOP + (tc - TC_RAMP_TOP) * (
            TC_APPARENT_RAMP_BOT - TC_APPARENT_RAMP_TOP
        ) / (TC_RAMP_BOT - TC_RAMP_TOP)
    tk_apparent = tc_apparent + TMELT

    # Ramped supersaturation
    supersatice_ramp = supersatice
    if FRZ_ASSUME_RAMP_RHCE:
        if tc_apparent >= -30.0 and tc_apparent <= -10.0:
            supersatice_ramp = 1.34 + (tc_apparent - (-30.0)) * (1.10 - 1.34) / ((-10.0) - (-30.0))
        elif tc_apparent < -30.0:
            supersatice_ramp = 1.34
        elif tc_apparent > -10.0:
            supersatice_ramp = 1.10

    # Ice properties
    rhoice = 916.7 - 0.175 * tc_apparent - 5.0e-4 * tc_apparent**2
    vwice = MWH2O * AMU / rhoice
    sigma_iw = (28.5 + 0.25 * tc_apparent) * 1e-3  # [J/m2]

    # Critical germ radius for immersion
    do_dst1 = True
    if DEPR_POINT_FRZ:
        if aw * supersatice_ramp > 1.0:
            rgimm = 2 * vwice * sigma_iw / (KBOLTZ * tk_apparent * np.log(aw * supersatice_ramp))
        else:
            do_dst1 = False
            rgimm = 2 * vwice * sigma_iw / (KBOLTZ * tk_apparent * np.log(supersatice_ramp))
    else:
        rgimm = 2 * vwice * sigma_iw / (KBOLTZ * tk_apparent * np.log(supersatice_ramp))

    # Homogeneous energy of germ formation
    dg0 = 4 * PI / 3.0 * sigma_iw * rgimm**2

    # Prefactor A' (Eq. 1)
    A_prime = N1 * (
        (vwice * RHPLANCK) / (rgimm**3)
        * np.sqrt(3.0 / PI * KBOLTZ * tk_apparent * dg0)
    )

    return {
        "tc": tc,
        "tk_apparent": tk_apparent,
        "supersatice": supersatice,
        "do_dst1": do_dst1,
        "rgimm": rgimm,
        "dg0": dg0,
        "A_prime": A_prime,
    }


def compute_Jimm_at_alpha(
    alpha_rad: float,
    A_prime: float,
    r_dust: float,
    dg0: float,
    tk_apparent: float,
) -> float:
    """Compute J_imm for a single contact angle (Eq. 1 + Eq. 2).

    J_het = A' * r_N^2 / sqrt(f) * exp((-dg# - f*dg0) / (kT))
    f = (2+m)(1-m)^2 / 4,  m = cos(alpha)
    """
    m = np.cos(alpha_rad)
    f = (2 + m) * (1 - m) ** 2 / 4.0  # Eq. 2
    J = (
        A_prime
        * r_dust**2
        / np.sqrt(f)
        * np.exp((-DGA_IMM_DUST - f * dg0) / (KBOLTZ * tk_apparent))
    )
    return max(J, 0.0)


def compute_ff_single(
    t: float,
    r_dust: float,
    deltat: float,
    alpha_deg: float = THETA_IMM_DUST,
    aw: float = 1.0,
) -> float:
    """Frozen fraction using a single contact angle (Eq. 4 with p(α)=δ(α-α₀)).

    ff_single = 1 - exp(-J_imm(T, alpha) * dt)
    """
    thermo = compute_thermo(t, aw=aw)
    if not thermo["do_dst1"]:
        return 0.0
    alpha_rad = alpha_deg / 180.0 * PI
    J = compute_Jimm_at_alpha(
        alpha_rad, thermo["A_prime"], r_dust, thermo["dg0"], thermo["tk_apparent"]
    )
    return 1.0 - np.exp(-J * deltat)


def compute_ff_pdf(
    t: float,
    r_dust: float,
    deltat: float,
    mu_deg: float = THETA_IMM_DUST,
    sigma: float = IMM_DUST_VAR_THETA,
    aw: float = 1.0,
) -> float:
    """Frozen fraction using the α-PDF model (Eq. 3 + Eq. 4).

    ff_pdf = 1 - integral_0^pi p(alpha) * exp(-J_imm(T,alpha)*dt) d_alpha
    """
    thermo = compute_thermo(t, aw=aw)
    if not thermo["do_dst1"]:
        return 0.0

    dim_theta, pdf_imm_theta, dim_f, pdf_d_theta = init_pdf_theta()

    # Compute J_imm at each theta bin (Eq. 1)
    dim_Jimm = np.zeros(PDF_N_THETA)
    for i in range(I1, I2 + 1):
        dim_Jimm[i] = (
            thermo["A_prime"]
            * r_dust**2
            / np.sqrt(dim_f[i])
            * np.exp((-DGA_IMM_DUST - dim_f[i] * thermo["dg0"]) / (KBOLTZ * thermo["tk_apparent"]))
        )
        dim_Jimm[i] = max(dim_Jimm[i], 0.0)

    # Trapezoidal integration of survival fraction (Eq. 4)
    # NOTE: no 0.99 clamping here — that's a CESM Fortran numerical hack,
    # not in the paper's Eq. 4. It kills the α-PDF signal at warm T.
    survival = 0.0
    for i in range(I1, I2):
        survival += 0.5 * (
            pdf_imm_theta[i] * np.exp(-dim_Jimm[i] * deltat)
            + pdf_imm_theta[i + 1] * np.exp(-dim_Jimm[i + 1] * deltat)
        ) * pdf_d_theta

    return max(0.0, 1.0 - survival)


def compute_Jimm_dust_a1(
    t: float,
    p: float,
    supersatice: float,
    r_dust_a1: float,
    total_cloudborne_dust_a1: float,
    deltat: float,
    aw_dst1: float = 1.0,
    icnlx: float = 0.0,
) -> dict:
    """
    Compute Jimm_dust_a1 and the resulting frzduimm contribution from dust_a1.

    Parameters (EXTERNAL INPUTS — user must provide):
    ----------
    t : float
        Temperature [K]
    p : float
        Pressure [Pa]
    supersatice : float
        Ice supersaturation ratio at 100% RH over water [ ]
        = eswtr / esice  (saturation vapor pressure ratio)
    r_dust_a1 : float
        Mass-mean radius of dust accumulation mode [m]
        (typical: ~0.25e-6 to 1.5e-6 m)
    total_cloudborne_dust_a1 : float
        Cloudborne dust_a1 number concentration [#/cm3]
        (typical: 0.01 to 10 #/cm3)
    deltat : float
        Model timestep [s]  (e.g. 1800 s for 30-min physics)
    aw_dst1 : float, optional
        Water activity for dust_a1 [ ]. Default 1.0 (pure water, no solute effect).
        To compute properly, need awcam, awfacm, r3lx, total_interstitial_aer_num.
    icnlx : float, optional
        In-cloud droplet concentration [cm-3]. Only needed for background drop term.

    Returns:
    -------
    dict with intermediate quantities and final results.
    """
    # --- Saturation vapor pressures ---
    eswtr = svp_water_goff_gratch(t)
    esice = svp_ice_goff_gratch(t)

    # --- Temperature conversions ---
    tc = t - TMELT

    # Apparent temperature ramp
    tc_apparent = tc
    if tc >= TC_RAMP_TOP and tc <= TC_RAMP_BOT:
        tc_apparent = TC_APPARENT_RAMP_TOP + (tc - TC_RAMP_TOP) * (
            TC_APPARENT_RAMP_BOT - TC_APPARENT_RAMP_TOP
        ) / (TC_RAMP_BOT - TC_RAMP_TOP)
    tk_apparent = tc_apparent + TMELT

    # Ramped supersaturation
    supersatice_ramp = supersatice
    if FRZ_ASSUME_RAMP_RHCE:
        if tc_apparent >= -30.0 and tc_apparent <= -10.0:
            supersatice_ramp = 1.34 + (tc_apparent - (-30.0)) * (1.10 - 1.34) / ((-10.0) - (-30.0))
        elif tc_apparent < -30.0:
            supersatice_ramp = 1.34
        elif tc_apparent > -10.0:
            supersatice_ramp = 1.10

    # --- Ice properties (functions of tc_apparent) ---
    rhoice = 916.7 - 0.175 * tc_apparent - 5.0e-4 * tc_apparent**2
    vwice = MWH2O * AMU / rhoice
    sigma_iw = (28.5 + 0.25 * tc_apparent) * 1e-3  # [J/m2]

    # --- Critical germ radius for immersion ---
    # Check freezing point depression
    do_dst1 = True
    if DEPR_POINT_FRZ:
        if aw_dst1 * supersatice_ramp > 1.0:
            do_dst1 = True
            rgimm_dust_a1 = 2 * vwice * sigma_iw / (KBOLTZ * tk_apparent * np.log(aw_dst1 * supersatice_ramp))
        else:
            do_dst1 = False
            rgimm_dust_a1 = 2 * vwice * sigma_iw / (KBOLTZ * tk_apparent * np.log(supersatice_ramp))
    else:
        rgimm_dust_a1 = 2 * vwice * sigma_iw / (KBOLTZ * tk_apparent * np.log(supersatice_ramp))

    # --- Homogeneous energy of germ formation ---
    dg0imm_dust_a1 = 4 * PI / 3.0 * sigma_iw * rgimm_dust_a1**2

    # --- Prefactor A ---
    Aimm_dust_a1 = N1 * (
        (vwice * RHPLANCK) / (rgimm_dust_a1**3)
        * np.sqrt(3.0 / PI * KBOLTZ * tk_apparent * dg0imm_dust_a1)
    )

    # --- PDF-theta integration (pdf_imm_in = True) ---
    dim_theta, pdf_imm_theta, dim_f, pdf_d_theta = init_pdf_theta()

    dim_Jimm_dust_a1 = np.zeros(PDF_N_THETA)
    for i in range(I1, I2 + 1):
        dim_Jimm_dust_a1[i] = (
            Aimm_dust_a1
            * r_dust_a1**2
            / np.sqrt(dim_f[i])
            * np.exp((-DGA_IMM_DUST - dim_f[i] * dg0imm_dust_a1) / (KBOLTZ * tk_apparent))
        )
        dim_Jimm_dust_a1[i] = max(dim_Jimm_dust_a1[i], 0.0)

    # --- Integrate PDF: survival fraction ---
    sum_imm_dust_a1 = 0.0
    for i in range(I1, I2):  # trapezoidal rule
        sum_imm_dust_a1 += 0.5 * (
            pdf_imm_theta[i] * np.exp(-dim_Jimm_dust_a1[i] * deltat)
            + pdf_imm_theta[i + 1] * np.exp(-dim_Jimm_dust_a1[i + 1] * deltat)
        ) * pdf_d_theta

    if sum_imm_dust_a1 > 0.99:
        sum_imm_dust_a1 = 1.0

    frozen_fraction = 1.0 - sum_imm_dust_a1

    # --- Freezing rate [#/cm3/s] ---
    if do_dst1 and t <= HCLAS_MAX_T_IMM:
        frzduimm_a1 = min(
            LIMFACDU * total_cloudborne_dust_a1 / deltat,
            (total_cloudborne_dust_a1 + DU_BACKGROUND) / deltat * frozen_fraction,
        )
    else:
        frzduimm_a1 = 0.0

    # Add background drop freezing if enabled
    if BACK_DROP_PERC_PER_HOUR > 0.0:
        frzduimm_a1 += BACK_DROP_PERC_PER_HOUR * 0.00028 * icnlx

    # --- Peak Jimm (the single-theta Jimm at PDF mean) ---
    # Find the theta bin closest to the mean for reporting
    mean_theta_rad = THETA_IMM_DUST / 180.0 * PI
    i_peak = np.argmin(np.abs(dim_theta[I1:I2 + 1] - mean_theta_rad)) + I1
    Jimm_at_peak = dim_Jimm_dust_a1[i_peak]

    return {
        # Intermediate quantities
        "tc": tc,
        "tc_apparent": tc_apparent,
        "tk_apparent": tk_apparent,
        "eswtr": eswtr,
        "esice": esice,
        "supersatice_ramp": supersatice_ramp,
        "rhoice": rhoice,
        "vwice": vwice,
        "sigma_iw": sigma_iw,
        "rgimm_dust_a1": rgimm_dust_a1,
        "dg0imm_dust_a1": dg0imm_dust_a1,
        "Aimm_dust_a1": Aimm_dust_a1,
        # PDF integration
        "sum_imm_dust_a1 (survival fraction)": sum_imm_dust_a1,
        "frozen_fraction (1 - survival)": frozen_fraction,
        "Jimm_at_pdf_peak [s-1]": Jimm_at_peak,
        # Final output
        "do_dst1": do_dst1,
        "frzduimm_dust_a1 [#/cm3/s]": frzduimm_a1,
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

    for i, t in enumerate(t_range):
        ff_single[i] = compute_ff_single(
            t, R_DUST_A1, DELTAT, alpha_deg=ALPHA_SINGLE_DEG, aw=AW,
        )
        ff_pdf[i] = compute_ff_pdf(
            t, R_DUST_A1, DELTAT, mu_deg=MU_PDF_DEG, sigma=SIGMA_PDF, aw=AW,
        )

    # --- Print summary at a few temperatures ---
    print("=" * 70)
    print("  Frozen fraction comparison: single-alpha vs alpha-PDF")
    print(f"  r_dust = {R_DUST_A1*1e6:.3f} um, dt = {DELTAT:.0f} s")
    print(f"  single-alpha: alpha = {ALPHA_SINGLE_DEG}°")
    print(f"  alpha-PDF: mu = {MU_PDF_DEG}°, sigma = {SIGMA_PDF}")
    print("=" * 70)
    print(f"  {'T [°C]':>8s}  {'ff_single':>12s}  {'ff_pdf':>12s}")
    print("-" * 70)
    for tc_val in [-35, -30, -25, -20, -15, -10]:
        idx = np.argmin(np.abs(tc_range - tc_val))
        print(f"  {tc_range[idx]:8.1f}  {ff_single[idx]:12.4e}  {ff_pdf[idx]:12.4e}")
    print("=" * 70)

    # --- Load digitized Wang et al. (2014) Fig. 1 data ---
    csv_path = Path(__file__).parent / "fig_1_wang_ff_vs_t.csv"
    wang_data = {}
    if csv_path.exists():
        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for col in reader.fieldnames:
                wang_data[col] = []
            for row in reader:
                for col in reader.fieldnames:
                    val = row[col].strip() if row[col] else ""
                    wang_data[col].append(float(val) if val else np.nan)
        for col in wang_data:
            wang_data[col] = np.array(wang_data[col])
        print(f"  Loaded {csv_path.name} with {len(wang_data['temperature_C'])} points")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Replace zeros with NaN for clean log plotting
    ff_single_plot = np.where(ff_single > 0, ff_single, np.nan)
    ff_pdf_plot = np.where(ff_pdf > 0, ff_pdf, np.nan)

    # Our model curves
    ax.semilogy(tc_range, ff_single_plot, "r-", linewidth=2, label=f"single-$\\alpha$ ($\\alpha$={ALPHA_SINGLE_DEG}°)")
    ax.semilogy(tc_range, ff_pdf_plot, "b--", linewidth=2, label=f"$\\alpha$-PDF ($\\mu$={MU_PDF_DEG}°, $\\sigma$={SIGMA_PDF})")

    # Wang et al. (2014) Fig. 1 digitized data
    if wang_data:
        tc_wang = wang_data["temperature_C"]

        # CSU106 observations
        mask = ~np.isnan(wang_data["CSU106_obs"])
        if mask.any():
            ax.semilogy(tc_wang[mask], wang_data["CSU106_obs"][mask], "ro", ms=7, mfc="none", label="CSU106 Obs (Wang+2014)")

        # ZINC106 observations
        mask = ~np.isnan(wang_data["ZINC106_obs"])
        if mask.any():
            ax.semilogy(tc_wang[mask], wang_data["ZINC106_obs"][mask], "bs", ms=7, mfc="none", label="ZINC106 Obs (Wang+2014)")

        # CSU106 alpha-PDF (paper's fit)
        mask = ~np.isnan(wang_data["CSU106_alpha_PDF"])
        if mask.any():
            ax.semilogy(tc_wang[mask], wang_data["CSU106_alpha_PDF"][mask], "r:", linewidth=1.5, alpha=0.6, label="CSU106 $\\alpha$-PDF (Wang+2014)")

        # CSU106 single-alpha (paper's fit)
        mask = ~np.isnan(wang_data["CSU106_single_alpha"])
        if mask.any():
            ax.semilogy(tc_wang[mask], wang_data["CSU106_single_alpha"][mask], "r-.", linewidth=1.5, alpha=0.6, label="CSU106 single-$\\alpha$ (Wang+2014)")

    ax.set_xlabel("Temperature [°C]", fontsize=13)
    ax.set_ylabel("Active Fraction", fontsize=13)
    ax.set_title(
        f"Wang et al. (2014) — Immersion freezing of dust\n"
        f"$r_N$ = {R_DUST_A1*1e9:.0f} nm, $\\Delta t$ = {DELTAT:.0f} s",
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
