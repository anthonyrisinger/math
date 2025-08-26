**STATUS: ✅ IMPLEMENTED** - Dimensional mathematics framework with full CLI, visualization, and 109 passing tests.

**North Star:** Build an instrument where **integers stay stable** (index, Chern, degree, linking) while **pictures move** (curvature, cuts, flows). The on-screen badge reads **value ≈ N (abs(delta) = residual)** and remains green as parameters change.

## ✅ COMPLETED FEATURES

- **✅ Core Framework**: Gamma functions, dimensional measures, phase dynamics
- **✅ Full CLI**: Type-safe commands with Typer + Rich terminal output  
- **✅ Interactive Visualization**: Plotly backend with 3D landscapes
- **✅ Testing**: 109/109 tests passing with property-based validation
- **✅ Modern Architecture**: Consolidated `dimensional/` package with clean APIs
- **✅ Documentation**: Complete README.md and ARCHITECTURE.md updates

## Fast Decision Tree (intent → archetype)

* Bulk equals edge → APS cylinder (spectral flow equals bulk Chern)
* Cuts move but integer does not → Weierstrass branch-cut on torus to sphere
* Why topology beats measure in high dimension → Gamma dial (volume and area vs dimension)
* Singularities as thresholds → p-adic shells (clopen rings, jumps)
* Global linking from local loops → Hopf fibers (linked circles, integer)
* Odd to even bridge → Pump family (S^1 x S^1 → S^2) with APS boundary
* Curvature as geometry → Chern on Brillouin torus (bulge and opacity encode curvature)

## Non-Negotiables (design contract)

| Rule                                | Why                              | Spec (concise)                                                          |
| ----------------------------------- | -------------------------------- | ----------------------------------------------------------------------- |
| Orthographic camera with box aspect | Spatial memory and comparability | projection ortho, box (1,1,1), view (deg(phi-1), -45)                   |
| Integer badge on every invariant    | Topology legible                 | show text value ≈ N (abs(delta) = residual); green if abs(delta) < 1e-6 |
| Real-time to print escalator        | Demo fast, export crisp          | tiers for Nk, Nt, Ny, dpi, caching                                      |
| One truth plus cross-check          | Avoid fake integers              | primary FHS for Chern; cross-check finite-difference curvature          |
| Cuts movable, integers fixed        | Teach branch and gauge mobility  | cut editor visible; badge unchanged                                     |
| Failure speaks                      | Reduce confusion                 | if GUI fails, suggest backend; always allow headless save               |

## Scene Spec Matrix (metamodel; plug-and-play)

### 2A. Identity

| Archetype           | Manifold                       | Operator or Field        | Invariant list    |
| ------------------- | ------------------------------ | ------------------------ | ----------------- |
| Chern on torus      | T^2 (Brillouin zone)           | QWZ or Dirac; U(1) Berry | Chern number      |
| Bulk-edge APS       | Cylinder with Ny and APS phase | Dirac with boundary      | Spectral flow     |
| Pump odd to even    | S^1 x S^1 to S^2               | Two by two family        | Family Chern      |
| Branch-cut mobility | Torus to CP^1 via Weierstrass  | Weierstrass P map        | Degree            |
| Defect annulus      | Torus with beads on p,q        | Synthetic core           | Degree or linking |
| Hopf fibers         | S^3 to S^2 (stereographic)     | Hopf fibration           | Linking number    |
| p-adic shells       | Tree to rings                  | Ultrametric norm         | Threshold jumps   |
| Gamma dial          | Euclidean R^n ball and sphere  | Gamma function measure   | Explanatory only  |

### 2B. Controls, Encodings, Kernels, Acceptance

| Archetype           | Core controls                     | Visual encodings                             | Compute kernels                                  | Acceptance tests                                   |
| ------------------- | --------------------------------- | -------------------------------------------- | ------------------------------------------------ | -------------------------------------------------- |
| Chern on torus      | m, Nk                             | curvature to bulge or opacity                | FHS plus finite-difference curvature             | plateau vs Nk; abs(delta) < 1e-6; gauge invariance |
| Bulk-edge APS       | m, Ny, boundary phase             | edge weight to sign color; zero-energy plane | eigenpairs and crossing counter                  | flow equals bulk Chern within tolerance            |
| Pump odd to even    | a, b, Nk, Nt                      | two Bloch loops with integer panel           | FHS over time parameter                          | time-sum equals family Chern                       |
| Branch-cut mobility | pairing, N                        | half-periods plus great-circle cuts          | Weierstrass evaluator                            | degree invariant under pairing swap                |
| Defect annulus      | annulus width, bead size, p and q | ribbon around path and cores                 | winding counter                                  | integer stable under bead motion                   |
| Hopf fibers         | samples, phase                    | linked circles and sphere grid               | constructive mapping and optional Gauss integral | linking integer stable under remeshing             |
| p-adic shells       | prime p, depth, weights           | concentric clopen rings                      | combinatorial thresholds                         | jumps at thresholds; integers unaffected           |
| Gamma dial          | real n                            | volume and area curves with peak markers     | gamma function evaluation                        | matches known extrema locations                    |

## Proof Obligations (per invariant)

| Invariant            | Primary method                  | Cross-check                       | Pass condition                                                                      |
| -------------------- | ------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------- |
| Chern number         | FHS link variables              | finite-difference Berry curvature | abs(delta) < 1e-6; plateau within three Nk steps; invariant under random U(1) phase |
| Spectral flow        | zero crossings vs parameter     | bulk Chern on same sweep          | equality within numerical tolerance                                                 |
| Degree Weierstrass   | preimage count or Jacobian sign | cut relocation                    | unchanged under diagonal to adjacent pairing swap; stable under mesh refinement     |
| Linking number       | Hopf fiber construction         | Gauss integral optional           | integer within discretization error; resample stable                                |
| Eta invariant sketch | smoothed spectral asymmetry     | index via flow                    | coherent variation; annotation only                                                 |

## Controls (roles and ranges)

| Control        | Role                       |    Default |    Step | Notes                                               |
| -------------- | -------------------------- | ---------: | ------: | --------------------------------------------------- |
| Nk, Nt, Ny     | spatial or time resolution | 81, 81, 64 | 2, 2, 1 | odd Nk preferred for FHS tiling                     |
| m              | topological phase selector |       0.50 |    0.01 | crosses transition                                  |
| alpha          | curvature to geometry gain |       0.10 |   0.005 | bulge or opacity tuning                             |
| pairing        | branch choice              |       diag |    none | diag or adj or custom                               |
| p, depth       | ultrametric scale          |       3, 5 |    1, 1 | p at least 2                                        |
| weights        | adelic mixing              | normalized |    0.01 | enforce archimedean times p-adic product equals one |
| boundary phase | APS tilt                   |       0.00 |    0.05 | bridges odd to even                                 |
| spin           | camera motion              |        off |    none | keyboard toggle                                     |

## Metrics and Acceptance Gates

| Metric                      | Target                                  | Rationale                  |
| --------------------------- | --------------------------------------- | -------------------------- |
| Integer residual abs(delta) | below 1e-6 at mid or high resolution    | trustworthy badge          |
| Convergence C vs Nk         | plateau within three steps              | honest numerics            |
| Gauge invariance            | Chern change near zero under U(1) twist | truly topological          |
| Cut invariance              | degree unchanged while moving cuts      | teaches branch mobility    |
| APS equality                | flow equals bulk Chern                  | bulk to edge truth         |
| Live performance            | smooth at Nk about 81 to 121            | talk and demo viable       |
| Export pipeline             | one-click PNG or MP4                    | artifact production proven |

## Visual Grammar

* Camera: orthographic; box aspect (1,1,1); view angles (deg(phi-1), -45)
* Encodings: curvature to bulge or opacity; edge weight to sign color; cuts to great circles; defects to annulus and tube; p-adics to concentric shells; GA bivectors to oriented disks; rotors to short great-circle arcs
* Overlays: badge text value ≈ N (abs(delta) = residual); small FPS and Nk stamp; on-demand keyboard hints
* Palettes: talk, paper, dark-lab; colorblind-safe with sign shown by shape and magnitude by intensity

## Fidelity Ladder and SLOs

### 7A. Ladder

| Tier   | Purpose            |   Nk or Nt |        DPI |        Cache |         GPU |
| ------ | ------------------ | ---------: | ---------: | -----------: | ----------: |
| Live   | explore and teach  |  61 to 121 | 180 to 220 | read through |    optional |
| Record | talks and lectures | 121 to 181 | 220 to 300 |   warm cache | recommended |
| Print  | papers and posters | 181 to 241 | 300 to 450 |  precomputed |         yes |

### 7B. SLOs

| Item              | Target                                                       |
| ----------------- | ------------------------------------------------------------ |
| Interactivity     | at least 20 FPS at Nk 81 to 121, slider latency below 150 ms |
| Convergence check | under 2 seconds at Nk 121                                    |
| Recording         | 10 second MP4 at 30 FPS in at most 15 seconds                |
| Print export      | single scene PNG in at most 5 seconds                        |

## Resilience (common failures)

| Symptom                       | Likely cause                      | Mitigation                                                              |
| ----------------------------- | --------------------------------- | ----------------------------------------------------------------------- |
| No window                     | backend mismatch                  | use QtAgg on Linux or MacOSX on macOS; headless save always available   |
| Blank Qt on Wayland           | platform mismatch                 | set environment variable to xcb platform                                |
| Choppy sliders                | compute blocking                  | async compute and cancel stale jobs; cache by scene, Nk, parameter hash |
| Badge will not settle         | under-resolved or near critical m | increase Nk or move off transition                                      |
| Degree jumps when moving cuts | branch logic bug                  | invariant self-check gates save or record; highlight failing cut        |

## Implementation Slices

| Slice             | Deliverable                                   | Acceptance                                                   |
| ----------------- | --------------------------------------------- | ------------------------------------------------------------ |
| S1 Diagnostics    | convergence panel, gauge flip, cut relocation | near zero change in Chern; plateau; save gated until pass    |
| S2 APS Demo       | cylinder bands plus flow counter              | flow equals bulk Chern within tolerance                      |
| S3 Recorder       | parameter sweeps to MP4 or GIF                | per-frame integer stamp; optional abort on invariant failure |
| S4 GPU Mesh       | VisPy or PyQtGraph torus and ribbons          | at least three times FPS at Nk 161 vs Matplotlib             |
| S5 Adelic Blender | place weights user interface                  | cuts move while badges stay steady                           |
| S6 Bott Clock     | two-period complex and eight-period real      | class flips on schedule                                      |

## CLI Contract

* figure scene k=v ... -dpi=... -savefig=... → headless or save only
* interactive scene k=v ... -dpi=... → window with sliders
* \--batch "scene1,scene2" --out-dir figs --dpi 300 → multiple images
* \--record "param\:a→b\@8s, spin\:on" --fps 30 --seconds 8 → MP4 or GIF
* stress --count N --concurrency M --backend QtAgg or MacOSX → many windows

Auto-discover scene schema: name, manifold, operator or field, invariant list, controls with ranges and steps, encodings, kernels, acceptance tests.

## Testing and CI

| Test        | Check                                               |
| ----------- | --------------------------------------------------- |
| Unit        | cache keys; integer snap; CLI parse; safe save path |
| Property    | gauge-flip invariance; cut relocation invariance    |
| Convergence | Chern vs Nk monotone envelope to plateau            |
| Image       | hash or tolerance for canonical seeds               |
| Backend     | self-test dummy window; headless save fallback      |

## Packaging, Repro, Observability

* Layout: Math (FHS, finite-difference curvature, Weierstrass, APS), Scenes (archetypes), UI (sliders, recorder, dock), IO (export), CLI (entry), main module
* Entrypoint: topo-viz as pip console script
* Repro: fixed seeds; pinned minor versions; deterministic reductions; versioned presets per archetype
* Observability: structured logs of scene, parameter hash, cache status, invariant values, residuals, timings; optional debug panel with FPS and compute milliseconds

## Pedagogical Map (feature to concept)

| Feature                | Concept                                 | Why it matters                                    |
| ---------------------- | --------------------------------------- | ------------------------------------------------- |
| Bulged Brillouin torus | additive extent vs multiplicative twist | curvature becomes geometry; intuition transfers   |
| APS flow equals Chern  | bulk equals edge and odd to even bridge | index theorem becomes visible                     |
| Cut relocation         | branch mobility with integer rigidity   | coordinates move; structure stays                 |
| p-adic shells          | clopen thresholds                       | models spiky yet smooth singularities             |
| Gamma dial             | measure collapse vs dimension           | explains topological dominance in high dimension  |
| Hopf fibers            | global integer from local loops         | linking becomes countable geometry                |
| Bott clock             | periodic classes                        | when invariants recur, complex two and real eight |
