# arXiv readiness audit — *Dissolution of the collinear structure in the solar-sail CR3BP*

**Source:** `/Volumes/Bishwa/SOLAR_SAIL`, main file `paper/main.tex` (894 lines, single file — no `sections/`).
**Date:** 17 September 2026.
**Backup:** `paper/backup_20260917-150302/` holds the untouched `main.tex`, `refs.bib`, `verify_numbers.py` (md5 of `main.tex` confirmed identical to the pre-edit original before any change was made).

---

## 0. Where the briefing was wrong

You asked me to check your hypotheses against the folder rather than act on them. Five of them do not survive contact with the evidence:

| Briefing assumption | What the folder actually contains |
|---|---|
| "Inspect any existing submission zip" | **No zip exists.** `find . -name '*.zip'` returns nothing. There is no prior submission package to diff against. |
| "check whether a `.bbl` is present and complete" | **No `.bbl` exists anywhere** in the tree. Nothing had been built. |
| "If I attach the submitted PDF…" | **No PDF exists** anywhere in the repo, and none was attached. The `[?]`-vs-rendered-citations test could not be run — and was not needed, because the source compiles with 0 undefined citations from scratch (§2 below). |
| "Remove any journal-submission line (`Preprint submitted to …`)" | **No such line exists.** The only matches for `submitted`/`preprint` are in a comment block at the top describing how to retarget the document class. Nothing to remove. |
| "Proofread the Acknowledgements (AI-assistance disclosure)" | **There is no Acknowledgements section**, and no AI-assistance disclosure anywhere in the document. See open question **Q5**. |

One hypothesis was half-right and the mechanism matters:

> "add `\usepackage{doi}` AFTER hyperref so **plainnat's** plain-text `\doi` becomes a real link"

The document did not use `plainnat`. It used `\bibliographystyle{unsrt}`, which is a **standard BibTeX style that discards the `doi` field entirely** — it never emits `\doi` at all. Adding `doi.sty` to an `unsrt` document would have produced exactly zero DOI links, and the verification you asked for would have silently reported 0. The fix therefore had to change the bibliography style as well; see edit **E1b**.

`hidelinks` *was* present, as you said. Confirmed and fixed.

---

## 1. Baseline

| Item | Evidence |
|---|---|
| Submission zip | none |
| `.bbl` | none |
| Submitted PDF | none |
| `.bib` | `paper/refs.bib`, 18 entries, 16 carrying a `doi` field |
| Figures shipped in repo | 16 image/media files |
| Figures actually `\includegraphics`'d | 6 |
| Git state | clean history, 10 modified figure/CSV files uncommitted at audit time |
| Verification harness | `paper/verify_numbers.py` — 75 machine checks, **all 75 passing** against the original `main.tex` |

The `.venv` in the repo is a macOS virtualenv; its interpreter is a Mach-O binary and cannot run in a Linux sandbox. All Python was re-run against a fresh `numpy 2.4.4` / `scipy 1.17.1` environment importing your `src/` modules unmodified.

---

## 2. Compiling the way arXiv does

`bibtex` once to generate `main.bbl`, then `pdflatex ×3` with **no** bibtex, on a **fresh unzip into an empty directory containing nothing but the package contents**.

| Metric | Before (original source) | After (final package) |
|---|---|---|
| Undefined citations | **0** | **0** |
| Undefined references | **0** | **0** |
| LaTeX errors | **0** | **0** |
| Missing figure files | **0** | **0** |
| "Rerun to get …" warnings | 0 | **0** |
| Page count | 15 | **15** |
| `\bibitem` in `.bbl` | 15 | **15** |
| `\doi{}` in `.bbl` | 0 | **14** |
| `/Link` annotations in PDF | 90 | **105** |
| `/URI` entries in PDF | **0** | 28 |
| `doi.org` URIs in PDF | **0** | **14** |
| PDF `Title`/`Author`/`Subject`/`Keywords` | all empty | all populated |

14 of the 15 cited works carry a DOI and all 14 now resolve as live links. The one without is `acs3` (a NASA technical report with no DOI) — nothing to fix.

---

## 3. Full numerical claim audit

I extracted all **108 distinct numeric literals** in `main.tex` and cross-checked every one. `verify_numbers.py` covers 75 of them; the remainder I verified by re-running your own `src/` modules, the shipped `halo_atlas.csv`, and `results.txt`.

### 3.1 Table `tab:collapse` (saddle-strength collapse) — §3.2

| Claim | Source value | Verdict |
|---|---|---|
| s(β=0) = 3.0303, λ_u = 2.53256 | 3.030306129 / 2.532557945 | ✅ |
| s(0.010) = 2.123, λ_u = 2.13831 | 2.12300455 / 2.138305215 | ✅ |
| s(0.028646) = 1.000, λ_u = 1.50542 | 1.000 / 1.505418321 | ✅ |
| s(0.050) = 0.4012, λ_u = 0.99449 | 0.4011928 / 0.9944916 | ✅ |
| s(0.100) = 0.06820, λ_u = 0.44400 | 0.06819685 / 0.4440012 | ✅ |
| s(0.500) = 3.419e-4, λ_u = 0.03578 | 3.419116e-4 / 0.03578133 | ✅ |
| "20.6 r_H standoff" note | 20.6254 | ✅ |

### 3.2 Table `tab:closed` (closed form vs root-finding) — §4 — **DRIFT FOUND**

| Cell | Claimed in `.tex` | Recomputed now | Verdict |
|---|---|---|---|
| β_crit closed form, all 4 systems | 0.010961524792 / 0.028646456169 / 0.187175530204 / 0.406934946280 | identical to 12 dp | ✅ |
| \|Δ\| Sun–Mercury | 1.7e-16 | **3.1e-16** | ❌ **fixed → 3.1e-16** |
| \|Δ\| Sun–Earth | 6.9e-18 | **2.1e-16** | ❌ **fixed → 2.1e-16** |
| \|Δ\| Sun–Jupiter | 5.6e-17 | **2.2e-16** | ❌ **fixed → 2.2e-16** |
| \|Δ\| Earth–Moon | 5.6e-17 | **5.0e-16** | ❌ **fixed → 5.0e-16** |
| r₂ Sun–Mercury | 0.005500 | **0.005496** = (1.660e-7)^⅓ | ❌ **fixed → 0.005496** |
| r₂ Sun–Earth | 0.014427 | 0.014427 | ✅ |
| r₂ Sun–Jupiter | 0.098438 | **0.098432** = (9.537e-4)^⅓ | ❌ **fixed → 0.098432** |
| r₂ Earth–Moon | 0.229900 | **0.229893** = (1.215e-2)^⅓ | ❌ **fixed → 0.229893** |

**Table-vs-generated-output drift, decisively established.** The Sun–Earth cell is the smoking gun: `results.txt`, your own machine-generated file, prints

```
brentq on s(beta)-1 (independent) = 0.02864645616860539
residual                          = 2.082e-16
```

i.e. **2.1e-16**, while the manuscript table said **6.9e-18** — a 30× disagreement between the paper and the repo's own output for the same quantity. The likely cause is visible in the docstring of `critical_beta_tidal()` in `src/critical_beta.py`, which records that the brentq **bracket was changed** ("the previous fixed upper bound of 0.5 silently raised…"); a different bracket converges to a different last-place value, and the table was never regenerated afterwards.

The r₂ column was separately inconsistent with its own μ column: at parity r₂ = μ^⅓ exactly, and 0.005500 is not the cube root of the 1.660e-7 printed beside it.

**All eight cells regenerated from your own `critical_beta_tidal_exact` / `critical_beta_tidal` / μ^⅓.** No value was invented; each is the output of the function the table's own caption names. See **Q2**.

### 3.3 The `2.1e-16` sweep claim (abstract + §4) — **CROSS-EXPERIMENT CONFLATION**

> Abstract: "verified against independent root-finding to 2.1e-16 over μ∈[10⁻⁷,10⁻²]"
> §4: "The worst discrepancy over a **61-point logarithmic sweep** of μ∈[10⁻⁷,10⁻²] is 2.1e-16"

I ran exactly that sweep — `np.logspace(-7, -2, 61)`, your two functions:

| Claim | Source value | Verdict |
|---|---|---|
| worst \|Δ\| over the 61-point sweep = 2.1e-16 | **1.887e-15**, at μ = 2.154e-4 | ❌ **9× understated** |
| 2.1e-16 | = 2.082e-16, the **Sun–Earth** residual alone (`results.txt`) | correct number, **wrong scope** |

This is the pattern you asked me to hunt: a single-system residual promoted to "the worst over a sweep." `verify_numbers.py` never caught it because it only ever checks Sun–Earth. **Both occurrences fixed** (E3, E4): the 2.1e-16 is re-attributed to Sun–Earth, and the sweep bound is stated as the value the sweep actually produces. The substantive claim — machine precision, a few ulp — survives intact. See **Q3**.

### 3.4 Table `tab:tech` (flown hardware) — §8

σ* = 1.531111 g/m² recomputed from your CODATA/IAU chain. η = 1.12/1.79 = 0.625698 → 0.626 ✅

| Mission | area / mass | β_ideal claimed | recomputed | β_eff claimed | recomputed | Verdict |
|---|---|---|---|---|---|---|
| IKAROS | 196 / 307 | 0.00098 | 0.000978 | 0.00062 | 0.000615 (measured) | ✅ |
| LightSail-2 | 32 / 5 | 0.00980 | 0.009799 | 0.00613 | 0.006131 | ✅ |
| NEA Scout | 86 / 14 | 0.00941 | 0.009405 | 0.00588 | 0.005885 | ✅ |
| ACS3 | 80 / 16 | 0.00766 | 0.007656 | 0.00479 | 0.004790 | ✅ |
| Solar Cruiser | 1653 / — | 0.02024 | 0.020236 | — | — | ⚠️ **provenance undisclosed** |

**Sums and counts check:** every mass/area ratio in the table reproduces its σ and β to the digits printed. ✅

**The Solar Cruiser row cannot be reproduced from the table's own columns.** Its β comes from `a_char_mm_s2=0.12` in `src/sail_technology.py` — the published characteristic acceleration — divided by g_☉(1 AU) = 5.9301 mm/s², *not* from β = σ*/σ as the caption states for the table as a whole. The mass cell is "—", so a reader who tries σ*/σ gets nothing. Worse, your own source note records the figure as "**> 0.12 mm/s²**" — a lower bound. **Caption amended (E10)** to disclose both facts. The number itself is untouched.

### 3.5 Shortfall table and §8 prose — **ENDPOINT MISMATCH**

| Claim | Source value | Verdict |
|---|---|---|
| LightSail-2 ideal 0.009799, 2.9× | 0.009799, 2.9234× | ✅ |
| LightSail-2 realistic 0.006131, 4.7× | 0.006131, 4.6722× | ✅ |
| IKAROS 0.000615, 47× | 0.000615, 46.58× | ✅ |
| Solar Cruiser 0.020236, 1.4× | 0.020236, 1.4156× | ✅ |
| §8: "roughly **three times** beyond the best sail flown" | 2.9× is the *perfect-reflector* figure | ❌ **conflicts with abstract & conclusions** |
| Abstract: "a factor of **4.7** beyond demonstrated capability" | 4.67× = realistic figure | ✅ |
| Conclusions: "a factor **4.7** beyond the best deployed sail" | 4.67× | ✅ |
| §8: "within **40 %** of the most ambitious design" | 1.4156× ⇒ β_crit is **41.6 %** above it | ❌ **understated** |

The §8 paragraph says in one sentence that no deployed sail exceeds β = 0.00613, then in the next quotes the shortfall (2.9×) belonging to the *other* column, 0.009799. Abstract and conclusions both use 4.7. Tracing it back, the wording came verbatim from a suggested-abstract block inside `src/sail_technology.py` ("a factor of three… within 40 %"), which itself predates the table.

**Fixed (E8)** to state both, each labelled, using only values already in the table: "a factor of 4.7 beyond the best sail flown and deployed — 2.9 if that sail is credited with a perfect reflector — and a factor of 1.4 beyond the most ambitious design funded to flight hardware." The bogus "within 40 %" is gone; 1.4 is the table's own cell. See **Q1**.

### 3.6 Frequency-bound claims — §5

| Claim | Source value | Verdict |
|---|---|---|
| c₂* = 8/5, ν/ω = 2√2/3 = 0.942809041582 | exact | ✅ |
| λ_u²:ν²:ω² = 7:8:9 | 1.4 : 1.6 : 1.8 | ✅ |
| band 5.72 % wide | 100(1−0.9428090) = 5.719096 | ✅ |
| "a 4×10⁵-point scan" | `np.linspace(1.0+1e-9, 4.2, 400001)` | ✅ |
| agreement 8.1e-10 (in c₂) and 1.1e-16 (in ratio) | 8.13e-10 / 1.11e-16 | ✅ |
| 14 low-order rationals tested, 0 reachable | 14 / 0 | ✅ |
| classical L1 ν/ω = 0.9659 | 0.96585 | ✅ |
| c₂(L1)∈[4.042, 8.000], c₂(L2)∈[1.570, 3.959], c₂(L3)∈[1.000, 1.570] | 4.041986–8.000 / 1.569787–3.958777 / 1.0000009–1.569787 | ✅ |
| μ = 0.480187660 attains c₂* at L2 | 0.48018766 | ✅ |
| L3 max c₂ = 1.569787, misses bound by 3.07e-5 | 1.569786512 / 3.0689e-5 | ✅ |
| Sun–Earth classical c₂ = 4.060819 / 3.940764 / 1.000003 | exact to 6 dp | ✅ |
| β = 0.040932 places Sun–Earth on the bound; 1.4289× parity; s = 0.5897 | 0.04093195 / 1.428866 / 0.5896802 | ✅ |

### 3.7 Claims with no supporting table — re-run from source

| Claim | How verified | Verdict |
|---|---|---|
| §2: non-orthonormal triad error **1.0 %** at z/r₁ = 0.02 | √1.02 − 1 = **0.995 %** | ✅ (worst case over α, δ — see **Q8**) |
| §2: **4.4 %** at z/r₁ = 0.09 | √1.09 − 1 = **4.403 %** | ✅ |
| §2.2: C_sail spread **1.8e-15** vs C_grav **5.8e-6** at β = 0.001 over one period | **Re-integrated** the first `halo_atlas.csv` member (β = 0.001, A_z = 5.0415e-4, T = 3.0992) at rtol/atol 1e-13: C_sail spread **1.33e-15**, C_grav spread **5.787e-6**; recovered C_sail mean = 2.9988020258, matching the CSV cell exactly | ✅ (C_grav exact; C_sail spread is integrator-tolerance-dependent and of the same order — left alone) |
| §3.2: O(μ^⅓) error "**0.73 %** in c₂" | (2.014635 − 2)/2.014635 = **0.7265 %** | ✅ |
| §3.2: λ_u = 1.505418 vs 5^¼ = 1.495349 | exact | ✅ |
| §8: "at the best flown β ≈ 0.006 the saddle strength is **s ≈ 2.5**" (off-table — no such row exists) | s(0.006) = **2.459**, s(0.006131) = **2.447** | ✅ |
| Fig. 1 caption: "Earth neighbourhood exaggerated by a factor of **18**" | `src/schematic.py`, `zoom: float = 18.0` | ✅ |
| §6: γ "doubles across the band (0.0101 at β=0.001 to 0.0196 at β=0.05)" | `halo_atlas.csv`: 0.0100830 → 0.0195614, ratio **1.94** | ✅ |
| §6: "A_z = 0.02 … is roughly 2γ" | 0.02 / 0.0100830 = 1.98 | ✅ |
| §6: "twelve values of β" | CSV holds exactly 12 β values spanning [0.001, 0.050], 714 rows | ✅ |
| §7: σ₂ = 4.5e-4 against σ₁ = 5.2e-2 at α₀ = 0.5°, "115:1" | `results.txt`: 4.539e-4 / 5.202e-2, ratio **114.6** | ✅ |
| §6: "Δ_z is O(0.1) for a genuine halo" | CSV: \|Δ_z\| ∈ [0.109, 0.329] at β = 0.001 but falls to **[0.0125, 0.0863] at β = 0.05** | ⚠️ **see Q6** |
| §6: "Without the cross-β guard one of the twelve families lands on a different branch" | no artefact in the repo records this | ⚠️ **unverifiable — see Q7** |

### 3.8 Rounding-direction nit

| Claim | Source value | Verdict |
|---|---|---|
| Abstract (original): "no solar sail yet flown exceeds β = **0.0061**" | max = 0.0061313, which *does* exceed 0.0061 | ❌ **fixed → 0.00613**, matching Table `tab:tech` and §8 |
| §8: "No sail flown and deployed exceeds β = 0.00613" | 0.0061313 still marginally exceeds 0.00613 | ⚠️ standard rounding; see **Q9** |

### 3.9 Caption vs table

I checked every caption against the table or figure it labels. No caption promises a percentile, interval or column that its table omits. One caption **misused a symbol**:

| Claim | Problem | Verdict |
|---|---|---|
| Fig. `fig:dissolution` (b): "The standoff crosses the Hill sphere almost immediately (**β_crit = 2.98e-4**)" | Throughout the paper β_crit ≡ 0.0286, the *tidal-parity* threshold. 2.98e-4 is the *Hill-exit* value (§4 names it correctly as "β = 2.9814e-4"). The caption silently reassigns the paper's headline symbol to a different quantity two orders of magnitude away. | ❌ **fixed (E6)** — now "at β = 2.98e-4, the Hill-exit value, not β_crit" |

### 3.10 Unsupported rhetoric

I searched for "exponentially", "orders of magnitude", "collapses", "dramatic", "vanishes". The paper is unusually disciplined here: "collapse"/"collapses" is used 6 times and every instance is backed by `tab:collapse` (four decades of s) or the CSV. §5.2 explicitly frames its negative result as a negative. Limitations §9 pre-empts the over-claims. **No softening required.** The one place the prose ran ahead of the data is the Δ_z discriminator (Q6), which is a caveat rather than rhetoric.

---

## 4. Edits applied

Every edit, with the value that justifies it. Original preserved in `paper/backup_20260917-150302/`.

| # | Location | Change | Justification |
|---|---|---|---|
| **E1** | preamble | `\usepackage[hidelinks]{hyperref}` → `natbib[numbers,sort&compress]` + `hyperref` + `doi` + full `\hypersetup` with muted print-safe colours (navy `0.12,0.22,0.45` links / forest `0.10,0.36,0.26` citations / plum `0.40,0.16,0.36` URLs) and `pdftitle`/`pdfauthor`/`pdfsubject`/`pdfkeywords` | your brief; verified by /Link 90→105, /URI 0→28, doi.org 0→**14**, and `pdfinfo` now populated |
| **E1b** | `\bibliographystyle` | `unsrt` → `unsrtnat` | `unsrt.bst` discards the `doi` field outright, so `doi.sty` alone yields 0 links. `unsrtnat` preserves unsrt's citation-order numbering and emits `\doi{}`: `.bbl` went from 0 to **14** `\doi{}`. Citations still render as `[2]`, reference order unchanged. |
| **E2** | abstract | `β=0.0061` → `β=0.00613` | Table `tab:tech` cell 0.00613; also makes abstract and §8 agree |
| **E3** | abstract | "verified … to 2.1e-16 over μ∈[10⁻⁷,10⁻²]" → "verified … at machine precision over μ∈[10⁻⁷,10⁻²] (2.1e-16 at Sun–Earth)" | 2.1e-16 is the Sun–Earth residual (`results.txt`), not the sweep worst case |
| **E4** | §4 | "worst discrepancy over a 61-point sweep … is 2.1e-16" → "the two never differ by more than 1.9e-15 … at Sun–Earth the discrepancy is 2.1e-16" | re-ran your own functions on `np.logspace(-7,-2,61)`: max = 1.887e-15 |
| **E5** | Table `tab:closed` | 8 cells regenerated (4 × \|Δ\|, 3 × r₂) | §3.2 above; each is the output of the function the caption names |
| **E6** | Fig. `fig:dissolution` caption | `β_crit = 2.98e-4` → `β = 2.98e-4, the Hill-exit value, not β_crit` | §4 text: "Hill-sphere exit criterion … at β = 2.9814e-4"; β_crit is 0.0286 everywhere else |
| **E7** | §8 | "reproducing McInnes' 1.53 to four figures" → "to the three figures he quotes" | 1.5311 vs 1.53 agree to three significant figures, not four |
| **E8** | §8 | "roughly three times beyond the best sail flown and within 40 % of the most ambitious design" → "a factor of 4.7 beyond the best sail flown and deployed — 2.9 if that sail is credited with a perfect reflector — and a factor of 1.4 beyond the most ambitious design" | shortfall table rows 4.7 / 2.9 / 1.4; "40 %" was arithmetically 41.6 % |
| **E10** | Table `tab:tech` caption | added: Solar Cruiser has no published mass, its β is reduced from a_char ≥ 0.12 mm/s² and is a lower bound, not obtainable from the area column | `src/sail_technology.py` `a_char_mm_s2=0.12`; source note reads "> 0.12 mm/s²" |
| **E11** | §3.2 | added `Figure~\ref{fig:stability}` to the sentence that already describes the λ_u collapse | orphan float fix — see §5 |
| **E12** | preamble | `\graphicspath{{../}}` → `{{../}{./}}` | lets the same source build both from `paper/` locally and flat in the arXiv package |

**No experimental number was invented or nudged.** Every replaced value is either (a) already present in another table in the same paper, (b) printed in `results.txt`/`halo_atlas.csv`, or (c) the direct output of the `src/` function the surrounding text names. `verify_numbers.py` still reports **75/75 passing** against the edited manuscript.

---

## 5. Orphans

### Figures shipped but never `\includegraphics`'d (10)

`fig1_beta_family.png`, `fig3_floquet.png`, `fig5_station_keeping.png`, `fig6_poincare_map.png`, `fig7_manifold_transfer.png`, `paper_figure.png`, `beta_sweep_animation.gif`, `beta_sweep_animation.mp4`, `manifold_deployment.gif`, `manifold_deployment.mp4`

**Nothing deleted.** These are repository assets, and several (the Poincaré map, the manifold transfer, the animations) are clearly products of real work — `src/heteroclinic.py`, `src/transfer.py`, `src/stationkeeping.py`. They simply do not belong in an arXiv package that ships only what the manuscript uses, so the zip excludes them and the repo keeps them untouched. If you want any of them *in* the paper, say which and I will place it. See **Q10**.

### Labels never `\ref`'d

**Floats (fixed):**

- `fig:stability` — Figure 6 (`fig2_stability.png`) was typeset but never referenced. Fixed by **E11**, adding one cross-reference to the §3.2 sentence that already describes exactly what the figure shows ("Table 1 records that rate **and Figure 6 the accompanying collapse of λ_u**"). No new content invented.

**Equations (reported, not changed):**

- `eq:Aparity` — the closed form for c₂ at parity. The prose immediately after it (" which for Sun–Earth is 2.014635") points at it by position.
- `eq:alphastar` — the optimal cone angle. Same situation.

Both are harmless; LaTeX emits no warning and arXiv does not care. See **Q11**.

**Programmatic verification, post-edit:**

```
39 labels, 37 distinct targets referenced, 2 unreferenced
   EQUATION  eq:Aparity
   EQUATION  eq:alphastar

UNREFERENCED FLOATS: NONE  <-- PASS
```

---

## 6. Package hygiene

**`solar_sail_arxiv.zip` — 8 files, 1,195,147 bytes:**

```
main.tex
main.bbl
fig1_schematic.png
fig2_stability.png
fig4_frequency_ratio.png
fig5_control_authority.png
fig8_structure_dissolution.png
fig10_halo_atlas.png
```

**Excluded, and verified absent:** `refs.bib`, `__MACOSX/`, `._*`, `.DS_Store`, `.aux`, `.log`, `.out`, `.blg`, all PDFs, all unused figures, all animations, `.venv/`, `src/`, `__pycache__/`. A grep of the archive listing for any of these returns nothing.

**Acceptance test** — unzipped into an empty directory containing nothing else, then `pdflatex ×3` with **no bibtex**:

```
undefined citations : 0
undefined references: 0
LaTeX errors        : 0
missing figures     : 0
rerun warnings      : 0
Output written on main.pdf (15 pages, 1324981 bytes)
/Link annotations: 105
doi.org URIs     : 14
```

---

## 7. Open questions

**Q1 — §8 shortfall wording (please confirm).** Your paper quoted two different factors for "beyond the best sail flown": 4.7 in the abstract and conclusions, 2.9 in §8. I resolved it by stating both with their conditions attached, because both numbers are real and the ambiguity is the optical-efficiency assumption, not an error. If you would rather commit to one, tell me which and I will make all three places agree.

**Q2 — Table `tab:closed`, 8 regenerated cells.** These are floating-point residuals, and the brentq bracket in `critical_beta_tidal()` changed at some point after the table was typed. My regenerated values agree with `results.txt` for Sun–Earth. But: **is `critical_beta_tidal()` still the "independent root-finding" the table's caption means?** If you had a different independent implementation in mind, these four |Δ| numbers should come from it instead. I also want to confirm the r₂ column is meant to be μ^⅓ evaluated at the *printed* μ (which is what I used) rather than at fuller-precision mass ratios — if the latter, the **μ column** is what should change, not r₂.

**Q3 — the 2.1e-16 claim.** I kept your number and re-scoped it to Sun–Earth, and stated the true sweep bound (1.9e-15) alongside. The alternative is to drop the sweep framing entirely and quote Sun–Earth only. Both are honest; this touches the abstract, so I would rather you chose.

**Q4 — Data and code availability has no link.** The section says "reproducible from a single entry point" but names no repository, URL or DOI, which is not actionable for a referee. Your git remote is `https://github.com/Bishwaswarup/SOLAR_SAIL.git`. **Is that repository public?** If yes I will add it, ideally with a Zenodo DOI and the commit hash the paper's numbers were generated from. I have not added anything, because announcing a private URL in a preprint is worse than announcing none.

**Q5 — there is no Acknowledgements section and no AI-assistance disclosure.** You asked me to proofread one; there is nothing to proofread. arXiv does not require a disclosure, but a growing number of journals do, and this work carries a substantial computational pipeline. I have deliberately not drafted a disclosure or an acknowledgement on your behalf — what assistance was used, and how you characterise it, is yours to state. Tell me the substance and I will place and format it.

**Q6 — the Δ_z discriminator weakens at the top of the band.** §6 states "Δ_z is O(0.1) for a genuine halo … which makes it an effective discriminator." From `halo_atlas.csv` that holds well up to β = 0.032 (|Δ_z| ≥ 0.105), but at β = 0.04 the minimum falls to 0.0898 and at β = 0.05 to **0.0125** — an order of magnitude below the stated scale, and much closer to the vertical-Lyapunov branch's identically-zero value. The guard still separates the branches, so the claim is not wrong, but a referee who opens the CSV will notice. Suggested caveat: "… is O(0.1) for a genuine halo over most of the band, falling to 0.013 at β = 0.05". I have **not** applied this — it is your characterisation of your own method.

**Q7 — one claim is unverifiable from the repo.** §6: "Without the cross-β guard one of the twelve families lands on a different branch." Nothing in `halo_atlas.csv`, `results.txt` or the `src/` modules records a guard-off run, so I cannot confirm which β it was or reproduce the failure. Options: (a) re-run the atlas with the guard disabled and record the result, (b) name the β in the text so it is at least specific, (c) delete the sentence. I did not touch it.

**Q8 — triad-error percentages are worst-case.** The 1.0 % and 4.4 % figures are exact, but they are the maximum over cone and clock angle (sin 2α · sin δ = 1). The text presents them unqualified. Inserting "at most" would be strictly more accurate — a two-word edit I held back pending your say-so.

**Q9 — "exceeds 0.00613".** The exact maximum flown β is 0.0061313, which marginally exceeds the rounded 0.00613 now quoted in both the abstract and §8. This is ordinary rounding and almost certainly fine; if you want it airtight, "no solar sail yet flown reaches β = 0.0062" is unimpeachable.

**Q10 — six unused figures and four animations.** Confirm you are happy for the arXiv package to omit all of them (they remain in the repo, untouched). If any belongs in the paper — the Poincaré map and the manifold transfer look like the strongest candidates — tell me which and where.

**Q11 — two unreferenced equation labels** (`eq:Aparity`, `eq:alphastar`). Harmless. Leave, or add a cross-reference each?

---

## 8. Deliverables

| File | What it is |
|---|---|
| `solar_sail_arxiv.zip` | the clean arXiv package — 8 files, verified by compiling from an empty directory |
| `main.bbl` | standalone bibliography, 15 `\bibitem`, 14 `\doi{}` |
| `main.pdf` | built with **no bibtex**, 15 pages, 105 `/Link`, 14 `doi.org` URIs, full metadata |
| `arxiv_audit_report.md` | this report |
| `paper/main.tex` | edited in place on your machine |
| `paper/backup_20260917-150302/` | the untouched originals |
