# CMDA version: what changed from ../main.tex

The original `paper/main.tex` and `paper/refs.bib` are untouched. Everything below is in `paper/springer/`.

## Template and author block
- Uses the Springer Nature `sn-jnl` class with `sn-mathphys-num` (same files as the JAS cislunar paper).
- Author block matches the JAS paper: Bishwaswarup Nayak, Department of Physics, IISc, ORCID 0009-0001-9926-5329.
- Declarations block added: Funding, Competing interests, Use of AI-assisted tools, Data and code availability, Author contributions, Ethics.
- The derivations companion is now Supplementary Information (Online Resource 1).
- **Build:** run `pdflatex main && bibtex main && pdflatex main && pdflatex main`. This needs `siunitx`, which the TeX in the desktop VM lacks. Overleaf or a full TeX Live works.

## References: 19 to 52 in the .bib, 41 cited
- All new entries were checked against Crossref.
- **Fixed a bad entry:** `ceccaroni2015` had the wrong author list. It is now `bucciarelli2016` (Bucciarelli, Ceccaroni, Celletti, Pucacco).
- **Fixed a citation:** the "almost equal values of the frequencies" quote comes from Ceccaroni et al. 2016, *Physica D* 317, not from the AstroNet-II chapter. It now cites `ceccaroni2016physd`.
- **Added:**
  - photogravitational problem (Simmons et al. 1985)
  - generalised and non-ideal sails (Aliasi et al., Mengali et al.)
  - low-thrust artificial equilibria (Morimoto et al.)
  - control (Bookless & McInnes, Biggs et al., Farrés & Jorba, Huang et al.)
  - missions (Sunjammer, Solar Cruiser)
  - review (Gong & Macdonald)
  - Hill scale (Hamilton & Burns, Murray & Dermott)
  - centre manifold (Jorba & Masdemont, Pucacco 2019)
  - halo orbits and continuation (Breakwell & Brown, Howell, Doedel et al.)
  - **Verrier, Waters & Sieber 2014**

## Wording changes from the novelty check
1. **Tidal parity:** removed the claim that the term "has recently appeared in translunar astrodynamics". No citable source exists; the only use found is an informal seminar abstract, used in a different sense.
2. **β_crit:** added that solving the balance for β is elementary (McInnes et al. 1994). The claim now rests on the closed form at s = 1.
3. **ν/ω bound:** replaced "the ratio is not formed anywhere" with the prior work on the detuning ω − ν (Pucacco 2019; Ceccaroni et al. 2016; Bucciarelli et al. 2016). The claim is now limited to the ratio, its sharp bound and where it is attained. Added the remark that the Lyapunov-centre non-resonance condition holds at every collinear equilibrium.
4. **Abstract:** no citations, as Springer requires.

## Scientific correction: control section (please review)
The original says a face-on sail "reaches the in-plane pair or the vertical mode, never both" and that the Kalman rank drops to 4/6. That holds only for δ0 = 0 or δ0 = π/2.

For any other clock angle, the single input excites both blocks. Their spectra are disjoint because ν ≠ ω, so the system is controllable (rank 6/6). This was checked numerically at c2 = 1.409194: rank 6 at δ0 = 10°, 45° and 80°.

The abstract, the control section, the Fig. 5 caption and the conclusions now say: the face-on sail is a single-input system, uncontrollable only for purely in-plane or purely vertical tilt. Huang et al. 2020 already noted that the clock angle loses authority at α = 0.

**Still to fix:** panel (b) of `fig5_control_authority.png` is labelled "α0 = 0: rank 4/6, uncontrollable". Change it in the plotting code (probably `src/sail_authority.py`) to something like "α0 = 0: one input (rank 4/6 only at δ0 = 0)", then regenerate the figure.

## Before submitting: things only you can do
- [ ] **Verrier et al. 2014** report a branch point that splits the L1 halo family at β ≈ 0.0387, inside your [0.001, 0.05] band. Compare this with the branch discriminator weakening at β = 0.040–0.050 in the halo section. A CMDA referee will ask.
- [ ] Confirm the AI-use paragraph is accurate (marked `>>> AUTHOR` in main.tex).
- [ ] Create the `cmda-v1` git tag (and a Zenodo release) that the Data availability section names.
- [ ] Shorten the abstract. It is about 460 words; CMDA asks for 150–250.
- [ ] Rerun `paper/verify_numbers.py`. Its cell-by-cell table parsing points at `../main.tex`, not this file.
- [ ] A Zenodo record of this paper already exists (10.5281/zenodo.22967014). That is already a preprint with a DOI. Decide whether you still want Research Square. Either way, declare the preprint in the cover letter.
