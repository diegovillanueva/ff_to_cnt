## Handoff — ff_to_cnt

### What was done
- Created `boxmodel_Jimm_dust_a1.py`: a standalone Python box model that replicates the immersion freezing nucleation rate (Jimm) for fine dust (accumulation mode, dust_a1) from CESM2/CAM's `hetfrz_classnuc.F90`.
- Uses the PDF-theta approach (log-normal distribution of contact angles) with classical nucleation theory.
- All physical constants were verified against the local CESM2.1.5 Fortran source (`hetfrz_classnuc.F90`, `physconst.F90`, `shr_const_mod.F90`, `micro_dv_nml.F90`). Every constant matches exactly.

### Current state
- The box model runs standalone (`python boxmodel_Jimm_dust_a1.py`) and produces Jimm, frozen fraction, and freezing rate for a single point/timestep.
- Includes: Goff-Gratch saturation vapor pressure, apparent-temperature ramp, freezing-point depression, background-drop freezing term.
- Constants verified: KBOLTZ, HPLANCK, AMU, N1, MWH2O, TMELT, DGA_IMM_DUST, PDF_N_THETA, i1/i2, IMM_DUST_VAR_THETA, THETA_IMM_DUST, rhoice/sigma_iw/vwice formulas.
- Unused constants (RAIR, CPAIR, RH2O, RHOH2O) are defined but not referenced in any calculation — candidates for removal.

### Next steps
- Extend to contact nucleation (cnt) pathway — the "ff_to_cnt" goal.
- Add other aerosol species (dust_a3, BC) if needed.
- Add sweep/sensitivity scripts for parameter exploration.
- Consider removing dead constants (RAIR, CPAIR, RH2O, RHOH2O).

### Key decisions
- Used the **local** constants from `hetfrz_classnuc.F90` (e.g. `kboltz = 1.38e-23`) rather than the higher-precision values from `shr_const_mod` (e.g. `1.38065e-23`), to match the actual Fortran code path.
- Fortran 1-based indices (i1=53, i2=113) converted to 0-based Python (52, 112).
