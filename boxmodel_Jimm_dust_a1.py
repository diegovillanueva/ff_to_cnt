#!/usr/bin/env python3
"""
Box model for Jimm_dust_a1: immersion freezing nucleation rate for fine dust (accumulation mode)
from hetfrz_classnuc.F90 in CESM2/CAM.

Uses the PDF-theta approach (pdf_imm_in = True) with classical nucleation theory.

All internal constants are hardcoded from the Fortran source.
External inputs that the user must provide are clearly marked below.
"""
from __future__ import annotations

import numpy as np
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
# EXAMPLE: single-point, single-timestep calculation
# =============================================================================
if __name__ == "__main__":

    # =========================================================================
    # >>> EXTERNAL INPUTS — CHANGE THESE FOR YOUR CASE <<<
    # =========================================================================
    T = 253.15              # Temperature [K]  (-20 C)
    P = 50000.0             # Pressure [Pa]  (~500 hPa, mid-troposphere)
    # Ice supersaturation: eswtr/esice at this T (computed internally if not known)
    SUPERSATICE = svp_water_goff_gratch(T) / svp_ice_goff_gratch(T)
    R_DUST_A1 = 0.5e-6     # dust_a1 mass-mean radius [m]
    CLOUDBORNE_DST1 = 1.0   # cloudborne dust_a1 number [#/cm3]
    DELTAT = 1800.0         # timestep [s]  (30 min)
    AW = 1.0                # water activity (1.0 = no solute effect)
    ICNLX = 100.0           # in-cloud droplet concentration [cm-3]
    # =========================================================================

    results = compute_Jimm_dust_a1(
        t=T,
        p=P,
        supersatice=SUPERSATICE,
        r_dust_a1=R_DUST_A1,
        total_cloudborne_dust_a1=CLOUDBORNE_DST1,
        deltat=DELTAT,
        aw_dst1=AW,
        icnlx=ICNLX,
    )

    print("=" * 65)
    print("  Box model: Jimm_dust_a1  (immersion freezing, PDF-theta)")
    print("=" * 65)
    print(f"  Input T = {T:.2f} K  ({T - TMELT:.2f} C)")
    print(f"  Input P = {P:.0f} Pa")
    print(f"  Input Si = {SUPERSATICE:.4f}")
    print(f"  Input r_dust_a1 = {R_DUST_A1:.2e} m")
    print(f"  Input N_cloudborne = {CLOUDBORNE_DST1:.2f} #/cm3")
    print(f"  Input dt = {DELTAT:.0f} s")
    print("-" * 65)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:45s} = {v:.6e}")
        else:
            print(f"  {k:45s} = {v}")
    print("=" * 65)
