This is the **archetype spec** we snap to before implementing or explaining anything. Minimal prose, maximum structure. All syntax and tables verified.

## Non-negotiables (always on)

| Axis             | Value                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------- |
| Camera           | Orthographic; box aspect **(1,1,1)**; view angles **(deg(phi−1), −45°)**, where `phi = (1+sqrt(5))/2` |
| Invariant badge  | Format: `Chern ≈ 0.99998 -> 1 (residual=2e-5)` — nearest integer + residual; always visible           |
| Controls         | Sliders for local params; captions state the global invariant explicitly                              |
| Color discipline | Scalars → sober colormap; **topological features** (defects, cuts, cycles) → accent colors only       |
| Convergence      | Expose grid knobs (`Nk`, `Nt`); show “not converged” hint; cross-method checks where possible         |
| Save path        | Never fail: create parent dirs; record DPI in metadata; key `s` saves instantly                       |
| Keys             | `s` save • `r` reset • `p` spin • arrows/`a d w x` nudge sliders • `h` help                           |

## Conceptual invariants (what we must show)

| Invariant            | What it counts                           | Computed as                                                                            | Jumps when                    | Why it matters                                    |
| -------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------- |
| **Chern**            | Twisting of an eigenbundle on a 2-torus  | Discrete FHS product of link variables; cross-check: finite-difference Berry curvature | Gap closes                    | Binds **local curvature** to a **global integer** |
| **Index (Dirac)**    | #zero-modes (left) − #zero-modes (right) | Bulk integral + APS boundary term; equals spectral flow                                | BC/mass crosses critical      | **Bulk–edge** equality; transport quantization    |
| **Degree / Winding** | S¹/S² → target winds                     | Oriented angle/area accumulation                                                       | Map crosses singular set      | **Defect charge**; cuts move, integer doesn’t     |
| **Linking number**   | Loop–loop coupling (Hopf/pumps)          | Gauss integral / discrete tally                                                        | Loops pass through each other | Encodes **net transport per cycle**               |
| **Bott class**       | Stable phase type under dimension shift  | Step on Bott clock (±2 complex, ±8 real)                                               | Dimensional suspension        | Predicts **which invariant** to use next          |

## Control semantics (label extent vs twist)

| Channel            | Role                     | Typical knobs                           | Visual cue                        | Why                         |
| ------------------ | ------------------------ | --------------------------------------- | --------------------------------- | --------------------------- |
| **Additive**       | Extent / domain / scale  | grid `Nk`, length, radius, sweep range  | Reference frame, wireframe        | Sets **where** we measure   |
| **Multiplicative** | Twist / holonomy / phase | mass `m`, flux, coupling, bulge `alpha` | Normal displacement; phase arrows | Sets **what** we measure    |
| **Boundary**       | Edge law / APS           | domain wall, edge phase                 | Thickened edge band               | Makes bulk visible via edge |
| **Heuristic**      | ε-floors / thresholds    | p-adic weight, cut pairing              | Annulus/ring emphasis             | Models snap/jump behavior   |

## Scene archetypes (plug-and-play blueprints)

> Fill **Inputs / Controls / Outputs / Diagnostics / Why**, and you have a complete scene.

**A) Curvature → Integer (2-torus)**

* Inputs: `H(kx, ky; θ)` with a gap
* Controls: `θ`, grid `Nk`
* Outputs: curvature heatmap; **badge** `Chern …`
* Diagnostics: Chern vs `Nk` plot; FHS vs finite-diff check; gauge-twist invariance
* Why: geometry varies; **integer locks**

**B) Bulk–Edge = Index (cylinder / spectral flow)**

* Inputs: slab Hamiltonian + boundary law
* Controls: mass/path param; length; `Nk`
* Outputs: bands with edge-localization color; zero-crossing counter
* Diagnostics: equals torus Chern at same params
* Why: index theorem, visible

**C) Branch-cut relocation (elliptic / torus)**

* Inputs: fundamental domain + special points (half-periods)
* Controls: pairing (diag/adj), SL(2,Z) actions `S`, `T`
* Outputs: drawn cuts; **badge unchanged**
* Diagnostics: before/after pairing snapshots
* Why: gauge seam vs physics

**D) Annulus/Defect measurement (punctures)**

* Inputs: surface with defects
* Controls: annulus thickness, bead jitter, `alpha` for curvature concentration
* Outputs: beads + annuli; winding integer
* Diagnostics: multi-charge splits into simples under perturbation
* Why: boundary carries charge

**E) Pump / Triality (Bloch-sphere image)**

* Inputs: two-parameter family `H(k, t)`
* Controls: phase offsets `k0,t0`; amplitudes `a,b`
* Outputs: two constrained loops + one expressive; linking integer; beat readout
* Diagnostics: integer plateaus; jumps at snaps
* Why: transport per cycle; 2 constrain, 1 expresses

**F) Bott clock (dimensional step)**

* Inputs: minimal reps; loop/suspension operator
* Controls: step±; real/complex
* Outputs: class label on a ring; current node highlighted
* Diagnostics: matches known periods (2 complex, 8 real)
* Why: choose the right invariant

**G) p-adic shells / Adelic blend (heuristic)**

* Inputs: place weights (infty + selected primes)
* Controls: weight sliders; product-one normalization
* Outputs: clopen rings; geometry changes; **badge fixed**
* Diagnostics: move weight; invariant stable
* Why: ε-floors for snaps

**H) Gamma “dimension dial” (context)**

* Inputs: `Vol_n(1) = pi^(n/2) / Gamma(n/2+1)`, `Area_{n-1} = n * Vol_n`
* Controls: continuous `n` slider; peak markers
* Outputs: curves with maxima indicated
* Diagnostics: link collapse regimes to topological reliance
* Why: **volume collapses; topology persists**

## Visual grammar (consistent overlays)

| Element                     | Meaning                     | Notes                           |
| --------------------------- | --------------------------- | ------------------------------- |
| A/B cycles                  | Basis loops on torus        | Distinct weights; labeled       |
| Annulus                     | Measuring collar            | Thin band; color by role        |
| Bead                        | Defect center               | High-contrast, minimal          |
| Cut                         | Branch seam                 | Great circle/curve; pairing tag |
| Rotor arrow / Bivector disk | GA orientation & spin       | Subtle; never clutter           |
| Integer badge               | Global invariant & residual | Upper-left; always visible      |
| Convergence meter           | Grid adequacy               | Warn when below threshold       |

## Explanatory rhythm (ask–answer template)

1. **Local → Global:** what local field integrates to which integer? (Name both; point to badge; say when it **jumps**.)
2. **Gauge vs Physics:** what can move freely without changing the integer? (Cuts, pairings, phases.)
3. **Why here:** why compute this invariant in this scene? (Phase class, transport, stability.)

## Diagnostics & guarantees (baked into every scene)

| Check            | Method                                   | Pass criterion                      |
| ---------------- | ---------------------------------------- | ----------------------------------- |
| Gauge invariance | Random U(1) phases on eigenvectors       | Integer unchanged                   |
| Convergence      | Vary `Nk/Nt`; plot integer vs grid       | Plateaus to nearest integer         |
| Cross-method     | FHS vs finite-difference Berry curvature | Discrepancy decays with grid        |
| Bulk = Edge      | Torus Chern vs cylinder spectral flow    | Match at transitions                |
| Robustness       | Move cuts; perturb defects               | Additivity holds; invariants stable |

## Implementation skeleton (responsibility split)

| Layer    | Responsibility                                       | Notes                             |
| -------- | ---------------------------------------------------- | --------------------------------- |
| Compute  | Pure kernels (curvature, eigenpairs, link integrals) | Side-effect free; vectorized      |
| Cache    | Memoize `(grid, params)` tiles                       | Cancel stale jobs on slider moves |
| Render   | Wireframes, overlays, badges                         | Orthographic defaults centralized |
| Interact | Sliders, buttons, keys                               | One-keystroke save/reset          |
| Record   | Parameter sweep → MP4/PNG                            | Timestamp + badge burned in       |
| Validate | Convergence & cross-method panels                    | Auto-attach per scene             |

## Backend & performance policy

| Mode     | When         | Settings                                            |
| -------- | ------------ | --------------------------------------------------- |
| Realtime | Live sliders | Coarse `Nk`; async compute; GPU meshes if available |
| Quality  | Final images | High `Nk`; higher DPI; static compute               |
| Headless | Batch        | No GUI; deterministic seeds; save only              |

**Backends:** macOS `MacOSX` or `QtAgg`; Linux `QtAgg` (Wayland: set `QT_QPA_PLATFORM=xcb`). If GUI missing, print a clear fix and proceed to save.

## GA / Clifford (precise, minimal)

| Term              | One-liner                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| Clifford algebra  | Algebra on vectors with rule “a\*a equals its metric”; product blends inner and wedge           |
| Geometric algebra | Interpretive layer using multivectors and rotors to model geometry; Clifford-correct underneath |
| Even vs Odd dims  | Even: chirality splits; Odd: APS boundary recovers the missing partner—show the edge            |

## p-adic / adelic (heuristic stance)

| Concept           | Use                                                        | Guardrail                             |
| ----------------- | ---------------------------------------------------------- | ------------------------------------- |
| Ultrametric shell | Model ε-floors: “close” = longer shared prefix in base `p` | No new theorems; visualize invariance |
| Adelic blend      | Weight ∞ and `p` places with product-one normalization     | Integer must remain unchanged         |

## Copy blocks (reusable lines)

* **Invariant:** “The integer is the thing that doesn’t care about your seam.”
* **Channels:** “Additive sets the extent; multiplicative sets the twist.”
* **Bulk–edge:** “Bulk speaks through the edge; spectral flow equals index.”
* **Defects:** “Annuli measure holonomy; multi-charge splits into simples.”
* **Motivation:** “Volume collapses; topology persists.”

## Failure messages (pre-authored)

| Situation         | Message                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| No GUI backend    | “No GUI backend detected. Try `MPLBACKEND=QtAgg` (Linux/macOS) or `MPLBACKEND=MacOSX` (macOS). Saving still works.” |
| Grid too small    | “Resolution likely insufficient (`Nk < 61`). Increase grid and recheck the integer residual.”                       |
| Save path invalid | “Creating folders for save path… done. Saved with DPI and orthographic camera.”                                     |

## Acceptance checklist (per scene)

* [ ] Orthographic; aspect (1,1,1); view (deg(phi−1), −45°)
* [ ] Integer badge + residual visible
* [ ] Convergence knob & hint exposed
* [ ] Clear split: extent vs twist (labeled)
* [ ] Cuts/collars drawn & tagged (if applicable)
* [ ] One-press save; path creation guaranteed
* [ ] Health panel (gauge invariance or cross-method)
* [ ] Caption answers **what counts** and **why**

## WHY capsule (per archetype)

| Archetype             | WHY in one sentence                              |
| --------------------- | ------------------------------------------------ |
| Curvature → Integer   | Smooth fields can encode discrete truth.         |
| Bulk–Edge = Index     | Edges are the readout of bulk type.              |
| Branch-cut relocation | Gauge seams change; physics doesn’t.             |
| Annulus / Defect      | Holes speak in integers; boundary remembers.     |
| Pump / Triality       | Two constraints enable one expressive transport. |
| Bott clock            | Dimensional moves predict the invariant.         |
| p-adic shells         | Snaps have structural ε-floors.                  |
| Gamma dial            | When measure collapses, topology persists.       |

## Extension hooks (new domains, same spine)

* Swap Hamiltonian → keep **Curvature → Integer**.
* Change boundary law → keep **Bulk–Edge**.
* New multivalued map → keep **Branch-cut**.
* Different singularities → keep **Annulus/Defect**.
* Alternate 2-parameter family → keep **Pump/Triality**.
* Real vs complex classes → keep **Bott**.
* Non-archimedean flavor → keep **p-adic/adelic** overlay.
* High-dim analytics → keep **Gamma dial** context.

**Mantra:** *Show the count. Show what moves. Show what cannot be moved.*
Geometry, cuts, and weights may dance; the integer stands still.
