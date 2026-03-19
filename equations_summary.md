# Equations leading to Frozen Fraction (ff)

Reference: Wang et al. (2014, ACP), doi:10.5194/acp-14-10411-2014

## Step 1 — Ice properties (functions of T)

$$\rho_{ice} = 916.7 - 0.175\,T_c - 5 \times 10^{-4}\,T_c^2 \quad [\text{kg/m}^3]$$

$$v_{ice} = \frac{M_{H_2O} \cdot \text{AMU}}{\rho_{ice}} \quad [\text{m}^3]$$

$$\sigma_{iw} = (28.5 + 0.25\,T_c) \times 10^{-3} \quad [\text{J/m}^2]$$

where $T_c = T - 273.15$ is temperature in Celsius.

## Step 2 — Supersaturation

$$S_i = \frac{e_{sw}(T)}{e_{si}(T)}$$

Saturation vapor pressures $e_{sw}$ (over liquid water) and $e_{si}$ (over ice) from Goff & Gratch (1946).

## Step 3 — Critical germ radius

$$r_g = \frac{2\,v_{ice}\,\sigma_{iw}}{k\,T\,\ln(a_w \cdot S_i)}$$

where $k$ is the Boltzmann constant and $a_w$ is the water activity (1.0 for pure water).

## Step 4 — Homogeneous energy of germ formation

$$\Delta g_g^0 = \frac{4\pi}{3}\,\sigma_{iw}\,r_g^2$$

## Step 5 — Prefactor A'

$$A' = N_1 \cdot \frac{v_{ice}/h}{r_g^3} \cdot \sqrt{\frac{3\,k\,T\,\Delta g_g^0}{\pi}}$$

where $N_1 = 10^{19}$ m$^{-2}$ (water molecules per unit area of substrate) and $h$ is Planck's constant.

## Step 6 — Form factor f (Eq. 2)

$$f(\alpha) = \frac{1}{4}(2 + m)(1 - m)^2, \qquad m = \cos\alpha$$

where $\alpha$ is the contact angle.

## Step 7 — Nucleation rate J (Eq. 1)

$$J_{het}(T, \alpha) = \frac{A'\,r_N^2}{\sqrt{f}} \exp\!\left(\frac{-\Delta g^\# - f\,\Delta g_g^0}{k\,T}\right)$$

where $r_N$ is the aerosol particle radius and $\Delta g^\#$ is the activation energy.

## Step 8 — Frozen fraction (Eq. 4)

### Single-alpha model

All particles share one contact angle $\alpha_0$:

$$f_{act,\text{single}} = 1 - \exp\!\left(-J_{het}(T, \alpha_0)\,\Delta t\right)$$

### Alpha-PDF model

Contact angles follow a log-normal distribution (Eq. 3):

$$p(\alpha) = \frac{1}{\alpha\,\sigma\,\sqrt{2\pi}} \exp\!\left(-\frac{(\ln\alpha - \ln\mu)^2}{2\sigma^2}\right)$$

where $\mu$ is the mean contact angle and $\sigma$ is the standard deviation.

The frozen fraction integrates over all contact angles:

$$f_{act,\alpha\text{-PDF}} = 1 - \int_0^{\pi} p(\alpha) \cdot \exp\!\left(-J_{het}(T, \alpha)\,\Delta t\right) d\alpha$$

## Parameters for dust immersion freezing (Table 1 & 2, CSU106)

| Parameter | Single-$\alpha$ | $\alpha$-PDF |
|-----------|-----------------|--------------|
| $\alpha$ / $\mu$ | 46.0° | 46.0° |
| $\sigma$ | — | 0.01 |
| $\Delta g^\#$ | 14.75 $\times 10^{-20}$ J | 14.75 $\times 10^{-20}$ J |
| $r_N$ (Fig. 2) | 150 nm | 150 nm |
