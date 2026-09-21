"""
sail_technology.py — what lightness numbers have actually been flown.

Why this module exists
──────────────────────
The project's headline claim is that the tidal-parity threshold

    beta_crit = 1 - (1 - mu^(1/3))^2 = 0.028646     (Sun-Earth)

falls "inside the range of current sail technology (beta ~ 0.01-0.05)".  That
parenthetical was carried through README.md, critical_beta.py and atlas.py
without a source.  It does not survive checking.

Reducing every mission to beta from PRIMARY specifications (area, mass, and
where available a measured thrust) gives, for every solar sail ever flown:

    IKAROS        beta = 6.2e-4    (measured, not inferred)
    ACS3          beta = 4.8e-3
    LightSail-2   beta = 6.1e-3
    NEA Scout     beta = 5.9e-3    (sail never deployed; design value)

The best flown value is LightSail-2 at 6.1e-3, or 9.8e-3 if one credits it with
a perfect reflector.  Even the cancelled Solar Cruiser DESIGN reached only 0.020.
So beta_crit = 0.0286 is:

    2.9x  the best flown sail at its ideal-reflector limit (LightSail-2, 9.8e-3)
    4.7x  the best flown sail realistically  (LightSail-2, 6.1e-3)
    1.4x  the most ambitious funded design ever built to (Solar Cruiser, 0.020)
     46x  the only sail with a MEASURED lightness number (IKAROS, 6.2e-4)

The honest claim is therefore NOT "already achievable".  It is that tidal parity
sits a factor of a few beyond flown hardware and within ~40 % of a sail NASA had
already designed and begun building — which is a stronger and more interesting
statement than the unsupported one, because it is checkable.

The conversion
──────────────
beta is the ratio of solar radiation pressure acceleration to solar gravitational
acceleration, both at 1 AU and both scaling as 1/r^2, so the ratio is constant:

    beta = sigma* / sigma,        sigma* = 2 P / g_sun                     (1)

with sigma = m/A the sail loading, P = W/c the radiation pressure, and g_sun the
solar gravitational acceleration at 1 AU.  Evaluating (1) from CODATA/IAU
constants reproduces McInnes' standard critical loading to four figures:

    g_sun    = GM_sun / AU^2  = 5.9301 mm/s^2
    P        = W / c          = 4.5398 uN/m^2      (W = 1361 W/m^2)
    sigma*   = 2P / g_sun     = 1.5311 g/m^2       (McInnes gives 1.53)

Equation (1) assumes a perfect specular reflector.  Real sails fall short, and
IKAROS is the one mission that pins the shortfall by measurement: JAXA reported
an ideal thrust of 1.79 mN against a measured 1.12 mN, so

    optical efficiency  eta = 1.12 / 1.79 = 0.626                          (2)

`beta_ideal` below applies (1) alone; `beta_effective` applies (1) with eta from
(2).  For IKAROS neither is needed — beta comes straight from the measured
thrust and the spacecraft mass, and is the only entry here that is not inferred.

Sources
───────
[1] JAXA press release, 9 July 2010, "Small Solar Power Sail Demonstrator
    'IKAROS' Confirmation of Photon Acceleration".  Measured thrust 1.12 mN.
    https://www.jaxa.jp/press/2010/07/20100709_ikaros_e.html
[2] eoPortal, IKAROS mission summary.  307 kg wet mass (293 kg dry); sail 20 m
    diagonal, ~196 m^2.   https://www.eoportal.org/satellite-missions/ikaros
[3] JAXA IKAROS mission overview.  Sail "a huge square some 20 metres in a
    diagonal line, as thin as 0.0075 mm", polyimide.
    https://global.jaxa.jp/countdown/f17/overview/ikaros_e.html
[4] The Planetary Society, LightSail programme.  LightSail-2: 32 m^2 sail,
    5 kg spacecraft, sail deployed 23 July 2019, reentered November 2022.
    https://www.planetary.org/sci-tech/lightsail
[5] NASA, ACS3 (Advanced Composite Solar Sail System).  80 m^2 sail (9.0 m
    edge, four 20 m^2 aluminised PET quadrants), 16 kg total vehicle mass.
    https://ntrs.nasa.gov/citations/20210016824
[6] NASA Science, NEA Scout.  86 m^2 sail, 14 kg CubeSat; deployed from
    Artemis I on 16 Nov 2022, never contacted, declared lost December 2022.
    https://science.nasa.gov/mission/nea-scout/
[7] NASA Science, Solar Cruiser.  1653 m^2 sail, characteristic acceleration
    > 0.12 mm/s^2.  Not confirmed by NASA's Science Mission Directorate on
    28 June 2022; cancelled.
    https://science.nasa.gov/heliophysics/programs/technology/solar-cruiser/
[8] McInnes, C. R. (1999). "Solar Sailing: Technology, Dynamics and Mission
    Applications", Springer.  Source of sigma* = 1.53 g/m^2 and of the
    beta = sigma*/sigma convention used throughout this project.

CAVEAT.  Every beta here except IKAROS' is derived by the author from area and
mass, not quoted from the mission teams.  Where a mission publishes its own
lightness number it should be preferred; these are reproducible lower-bound
estimates built from the most defensible primary numbers available.
"""

from __future__ import annotations

# ── physical constants ────────────────────────────────────────────────────────
W_SOLAR   = 1361.0              # solar constant at 1 AU [W/m^2]
C_LIGHT   = 2.99792458e8        # [m/s]
GM_SUN    = 1.32712440018e20    # [m^3/s^2]
AU_M      = 1.495978707e11      # [m]

G_SUN_1AU = GM_SUN / AU_M**2    # 5.9301e-3 m/s^2
P_SRP     = W_SOLAR / C_LIGHT   # 4.5398e-6 N/m^2
SIGMA_CRIT = 2.0 * P_SRP / G_SUN_1AU     # 1.5311e-3 kg/m^2

# IKAROS optical efficiency: JAXA's ideal 1.79 mN against measured 1.12 mN [1].
ETA_IKAROS = 1.12 / 1.79        # 0.626


class Sail:
    """One mission, reduced to a lightness number from primary specifications."""

    def __init__(self, name, area_m2, mass_kg, status, source,
                 thrust_N=None, a_char_mm_s2=None):
        self.name = name
        self.area_m2 = area_m2
        self.mass_kg = mass_kg
        self.status = status
        self.source = source
        self.thrust_N = thrust_N
        self.a_char_mm_s2 = a_char_mm_s2

    @property
    def sigma_g_m2(self):
        """Sail loading m/A [g/m^2], or None when mass is unpublished."""
        if self.area_m2 is None or self.mass_kg is None:
            return None
        return 1e3 * self.mass_kg / self.area_m2

    @property
    def beta_measured(self):
        """beta from a MEASURED thrust — no optical assumption.  IKAROS only."""
        if self.thrust_N is None or self.mass_kg is None:
            return None
        return (self.thrust_N / self.mass_kg) / G_SUN_1AU

    @property
    def beta_ideal(self):
        """beta = sigma*/sigma, eq. (1): perfect specular reflector."""
        s = self.sigma_g_m2
        if s is None:
            if self.a_char_mm_s2 is not None:
                return (self.a_char_mm_s2 * 1e-3) / G_SUN_1AU
            return None
        return 1e3 * SIGMA_CRIT / s

    @property
    def beta_effective(self):
        """beta_ideal scaled by the IKAROS-measured optical efficiency, eq. (2)."""
        if self.beta_measured is not None:
            return self.beta_measured
        b = self.beta_ideal
        return None if b is None or self.sigma_g_m2 is None else b * ETA_IKAROS

    @property
    def flown(self):
        return self.status.startswith('flown')


# Ordered by launch date.  `flown` distinguishes hardware that actually operated
# in space from a design study — the distinction the unsourced "beta ~ 0.01-0.05"
# band elided.
MISSIONS = [
    Sail('IKAROS',        196.0,  307.0, 'flown 2010',   '[1][2][3]',
         thrust_N=1.12e-3),
    Sail('LightSail-2',    32.0,    5.0, 'flown 2019',   '[4]'),
    Sail('NEA Scout',      86.0,   14.0, 'lost 2022',    '[6]'),
    Sail('ACS3',           80.0,   16.0, 'flown 2024',   '[5]'),
    Sail('Solar Cruiser', 1653.0,  None, 'cancelled 2022', '[7]',
         a_char_mm_s2=0.12),
]


def flown_beta_range() -> tuple:
    """(min, max) effective beta over sails that actually flew and deployed."""
    b = [m.beta_effective for m in MISSIONS
         if m.flown and m.beta_effective is not None]
    return (min(b), max(b))


def table() -> str:
    """Citable table, formatted for the manuscript."""
    L = []
    L.append("Table N.  Lightness number of every flown solar sail, reduced from")
    L.append("primary specifications via beta = sigma*/sigma, "
             f"sigma* = {SIGMA_CRIT*1e3:.4f} g/m^2.")
    L.append("IKAROS' value is from its MEASURED 1.12 mN thrust and needs no")
    L.append("optical assumption; the others apply the IKAROS-derived efficiency")
    L.append(f"eta = {ETA_IKAROS:.3f} to the perfect-reflector limit.")
    L.append("")
    L.append(f"  {'mission':<15}{'area m2':>9}{'mass kg':>9}{'sigma g/m2':>12}"
             f"{'beta ideal':>12}{'beta eff':>10}  {'status':<16}src")
    L.append("  " + "-" * 96)
    for m in MISSIONS:
        s = m.sigma_g_m2
        bi, be = m.beta_ideal, m.beta_effective
        L.append(
            f"  {m.name:<15}"
            f"{m.area_m2:9.0f}"
            f"{(f'{m.mass_kg:.0f}' if m.mass_kg else '—'):>9}"
            f"{(f'{s:.1f}' if s else '—'):>12}"
            f"{(f'{bi:.5f}' if bi else '—'):>12}"
            f"{(f'{be:.5f}' if be else '—'):>10}"
            f"  {m.status:<16}{m.source}")
    lo, hi = flown_beta_range()
    L.append("")
    L.append(f"  Flown-and-deployed range:  beta in [{lo:.5f}, {hi:.5f}]")
    L.append(f"  IKAROS is the only MEASURED value: beta = "
             f"{MISSIONS[0].beta_measured:.5f}")
    return "\n".join(L)


def compare_to_threshold(beta_crit: float = None) -> str:
    """State the headline claim against the flown record, honestly."""
    if beta_crit is None:
        from src.critical_beta import critical_beta_tidal_exact, MU_SE
        beta_crit = critical_beta_tidal_exact(MU_SE)

    ls2 = next(m for m in MISSIONS if m.name == 'LightSail-2')
    ika = next(m for m in MISSIONS if m.name == 'IKAROS')
    sc = next(m for m in MISSIONS if m.name == 'Solar Cruiser')
    lo, hi = flown_beta_range()

    L = []
    L.append(f"  Tidal-parity threshold          beta_crit = {beta_crit:.6f}")
    L.append(f"  Best flown, ideal-reflector     LightSail-2 = "
             f"{ls2.beta_ideal:.6f}   ({beta_crit/ls2.beta_ideal:.1f}x short)")
    L.append(f"  Best flown, realistic           LightSail-2 = "
             f"{ls2.beta_effective:.6f}   ({beta_crit/ls2.beta_effective:.1f}x short)")
    L.append(f"  Only MEASURED value             IKAROS      = "
             f"{ika.beta_measured:.6f}   ({beta_crit/ika.beta_measured:.0f}x short)")
    L.append(f"  Most ambitious funded design    Solar Cruiser = "
             f"{sc.beta_ideal:.6f}   ({beta_crit/sc.beta_ideal:.2f}x short, cancelled)")
    L.append("")
    L.append(f"  Flown-and-deployed band: beta in [{lo:.4f}, {hi:.4f}].")
    L.append("  The unsourced claim 'beta ~ 0.01-0.05 is current technology' is")
    L.append(f"  NOT supported: no flown sail has exceeded {hi:.4f}, and the only")
    L.append(f"  measured value is {ika.beta_measured:.5f}.")
    L.append("")
    L.append("  Defensible wording for the abstract:")
    L.append("    'Tidal parity for Sun-Earth falls at beta = 0.0286, a factor of")
    L.append("     three beyond the best sail flown to date (LightSail-2) and")
    L.append("     within 40 % of NASA's Solar Cruiser design, placing it at the")
    L.append("     edge of near-term rather than current capability.'")
    return "\n".join(L)


if __name__ == '__main__':
    print()
    print(table())
    print()
    print("=" * 96)
    print("  Headline claim against the flown record")
    print("=" * 96)
    print(compare_to_threshold())
    print()
