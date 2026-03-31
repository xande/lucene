# TurboQuant Lucene Integration — Session Log & Debrief

> Date: 2026-03-30, ~20:00–20:55 UTC
> Participants: olexandb + AI assistant (multi-persona expert simulation)
> Artifacts produced: 3 documents in `/home/olexandb/Projects/TurboQuant/lucene/`

---

## Interaction Timeline

### 1. Initial Research & First Draft (20:00–20:06)

**Request:** Read the TurboQuant paper (arXiv:2504.19874v1) and GitHub repo (scos-lab/turboquant), then create a detailed plan for native Lucene integration.

**What happened:**
- Fetched and analyzed the full paper (algorithm, proofs, experiments)
- Read all core source files from the reference implementation (core.py, rotation.py, scalar_quantizer.py, qjl.py)
- Researched Lucene's codec architecture (KnnVectorsFormat, scalar quantization, int4, BBQ)
- Read the actual Lucene source for KnnVectorsFormat.java

**Output:** First draft of integration plan covering:
- Algorithm summary (MSE-optimal TurboQuant, why not Prod variant)
- New codec module structure with 8 Java files
- File format (.tqv, .tqn, .tqm, .tqg, .tqr)
- Index-time and search-time flows
- SIMD scoring strategy
- Key insight: rotation preserves distances → no inverse rotation at search time
- Hadamard rotation (seeded PRNG or Hadamard + diagonal)
- 5 implementation phases
- Principal engineer review identifying 4 open questions

**Key decision made:** MSE-only (not Prod) for NN search, based on reference impl's own finding that QJL residual hurts recall.

---

### 2. Decision Finalization (20:06–20:08)

**Request:** Answer the 4 open questions: (1) codecs module, (2) user-configurable, (3) Hadamard-only, (4) mixed-precision pros/cons. Save plan as .MD.

**Decisions:**
1. `lucene/codecs/` (experimental) ✓
2. User-configurable with default b=4 ✓
3. Hadamard-only mode as default ✓
4. No mixed-precision — rotation homogenizes distributions, per-field selection covers the useful case

**Output:** `TURBOQUANT_LUCENE_INTEGRATION_PLAN.md` saved to home directory.

---

### 3. File Move & Codebase Discovery (20:08–20:10)

**Request:** Move plan to `/home/olexandb/Projects/TurboQuant/lucene`.

**Discovery:** The target directory is a full Apache Lucene source checkout with `lucene/codecs/` module already present. Confirmed the exact target package path: `lucene/codecs/src/java/org/apache/lucene/codecs/turboquant/`.

---

### 4. Review Round 1 — Architecture, Performance, Compatibility (20:10–20:22)

**Request:** Review as community experts in Lucene, performance, and compatibility. Factor in 4K-dim embeddings.

**Simulated reviewers:** Lucene PMC (architecture), SIMD engineer (performance), production engineer (compatibility).

**Critical findings:**
- **BLOCKER:** Wrong abstraction layer — should be `FlatVectorsFormat`, not `KnnVectorsFormat`
- **BLOCKER:** Missing `FlatVectorsScorer` interface implementation
- **CRITICAL:** `getMaxDimensions()` must be raised above 1024 for d=4096
- **CRITICAL:** Off-heap storage mandatory at d=4096
- **HIGH:** Global rotation seed eliminates merge re-quantization (byte-copy merge)
- **HIGH:** d=4096 = 2^12 — perfect Hadamard, simplifies everything

**All findings incorporated.** Plan restructured around `FlatVectorsFormat` pattern matching `Lucene104ScalarQuantizedVectorsFormat` exactly.

---

### 5. d=768 Hadamard Support (20:22–20:25)

**Request:** We still need d=768 support, not just d=4096.

**Resolution:** Block-diagonal Hadamard with pre-permutation. d=768 decomposes to blocks (512, 256) — zero padding overhead. Any dimension works via binary decomposition. Updated plan with block decomposition table for all common dimensions.

---

### 6. Review Round 2 — API Reuse & Extensibility (20:25–20:31)

**Request:** Review as open-source Lucene principal engineers focused on extensibility and backward compatibility. Reuse existing APIs where possible.

**Simulated reviewers:** Lucene committer (API design), Lucene PMC (backward compat).

**Critical findings:**
- Keep `TurboQuantEncoding` separate from `ScalarEncoding` (different quantization model)
- Explicit `rawVectorDelegate` lifecycle matching Lucene104 writer exactly
- Must implement `mergeOneFieldToIndex()` returning `CloseableRandomVectorScorerSupplier`
- Remove standalone `TurboQuantCodec.java` — use `PerFieldKnnVectorsFormat` composition
- Extend `BaseKnnVectorsFormatTestCase` for ~50 free tests
- Keep everything in `lucene/codecs/`, nothing in `lucene/core`

**All findings incorporated.** Class hierarchy, writer lifecycle, and test strategy updated.

---

### 7. Review Round 3 — Testing Strategy (20:31–20:34)

**Request:** Review as PMC Lucene testers for proper testing strategy.

**Simulated reviewers:** Test framework maintainer, randomized testing expert.

**Critical findings:**
- `BaseKnnVectorsFormatTestCase` has hidden assumptions: `randomVectorEncoding()` returns BYTE randomly (TurboQuant is float-only), `getQuantizationBits()` defaults to 8
- `assertOffHeapByteSize()` hard-checks for `"vec"`, `"vex"`, `"veq"` keys
- Missing test categories: algorithm correctness, scoring correctness, merge-specific, adversarial inputs, CheckIndex, JMH benchmarks
- Randomized dimension and encoding testing needed

**Testing expanded** from 7 bullet points to 8 sub-sections with 40+ specific test items.

---

### 8. File Extension Resolution (20:34–20:38)

**Request:** How do file extensions work? TurboQuant uses different extensions than existing formats.

**Deep investigation:** Traced extension usage through Lucene104ScalarQuantizedVectorsReader, Lucene99HnswVectorsReader, and the test assertions. Found:
- Convention: different format types use different extensions (raw=`.vec`, scalar=`.veq`, binary=`.veb`)
- Extensions ARE reused across versions of same type, but NOT across different types
- `assertOffHeapByteSize()` checks for specific extension keys

**Resolution:** TurboQuant uses unique extensions `.vetq` / `.vemtq`. Override `assertOffHeapByteSize()` in test. Implement `QuantizedVectorsReader` interface for `hasQuantized()` detection.

---

### 9. Mike McCandless Gap Analysis (20:38–20:40)

**Request:** As Mike McCandless, identify top 6 gaps and questions for the community.

**Gaps identified:**
1. Global rotation seed fragility across AddIndexes/schema changes
2. Float32 numerical stability of Hadamard at d=4096
3. `.veq` extension reuse creates silent compatibility trap (later resolved with unique extensions)
4. No plan for when quantized search quality is unacceptable / graph building scorer
5. Memory accounting during indexing at d=4096
6. Block-diagonal Hadamard has no published theoretical backing

---

### 10. Expert Panel Resolves Gaps (20:40–20:46)

**Request:** Iterate on the 6 gaps as community of experts and improve the plan.

**Resolutions:**
1. **Rotation seed:** Field name is stable across AddIndexes. Added optional `rotationSeed` constructor parameter. Seed stored in metadata, verified during merge with fallback to re-quantization.
2. **Float32 stability:** WHT rounding error ~2e-7, quantization boundary spacing ~0.002. Five orders of magnitude margin. Float32 is fine.
3. **Extensions:** Reversed to unique `.vetq`/`.vemtq` (confirmed convention from codebase analysis).
4. **Graph building:** Confirmed scalar quant builds HNSW with quantized distances. TurboQuant follows same pattern. No two-pass mode needed.
5. **Memory accounting:** Confirmed scalar quant does NOT buffer quantized vectors in heap. Quantization streams at flush time. TurboQuant follows same pattern.
6. **Block-diagonal:** Theoretically sound (cross-block correlation = 0). Added empirical validation test: must be within 5% of full QR rotation.

---

### 11. Comprehensive Plan Review (20:46–20:51)

**Request:** Review implementation plan for well-defined tasks with testing gates between phases.

**Restructured** from flat checklists to:
- 5 phases with explicit entry criteria and gate conditions
- Each phase has numbered subtasks (1.1, 1.2, 2.1, etc.)
- Every subtask has inline tests
- Phase gates are pass/fail checklists that must clear before next phase starts
- Phase 2 uses naive scorer (correctness first), Phase 3 swaps in SIMD (performance second)

---

### 12. Document Split (20:51–20:55)

**Request:** Move implementation plan to separate doc.

**Result:** Three clean documents:
- `TURBOQUANT_LUCENE_INTEGRATION_PLAN.md` (589 lines) — design & architecture
- `TURBOQUANT_IMPLEMENTATION_PLAN.md` (323 lines) — phased execution plan
- `REVIEW_FEEDBACK.md` (562 lines) — 4 rounds of expert review audit trail

---

## Key Decisions Log

| # | Decision | Rationale | Interaction |
|---|----------|-----------|-------------|
| 1 | MSE-only, not Prod | Reference impl shows MSE beats Prod for NN search recall | 1 |
| 2 | `FlatVectorsFormat` not `KnnVectorsFormat` | Matches Lucene104 pattern; HNSW is orthogonal | 4 |
| 3 | Hadamard rotation, not QR | O(d log d) vs O(d²); d=4096 is perfect power of 2 | 2 |
| 4 | Block-diagonal Hadamard for d=768 | Zero padding overhead; binary decomposition | 5 |
| 5 | Global rotation seed from field name | Enables byte-copy merge (no re-quantization) | 4 |
| 6 | Optional explicit `rotationSeed` parameter | Safety for AddIndexes across indices | 10 |
| 7 | Unique extensions `.vetq`/`.vemtq` | Lucene convention: different format types use different extensions | 8 |
| 8 | No mixed-precision | Rotation homogenizes distributions; per-field selection suffices | 2 |
| 9 | Naive scorer first, SIMD second | Correctness before performance; Phase 2 vs Phase 3 | 11 |
| 10 | No standalone TurboQuantCodec | Users compose via PerFieldKnnVectorsFormat | 6 |
| 11 | Flush-time quantization (no heap buffering) | Matches Lucene104 pattern; streams through raw vectors | 10 |
| 12 | Implement `QuantizedVectorsReader` | Enables `hasQuantized()` detection in base test case | 8 |

## Artifacts

```
/home/olexandb/Projects/TurboQuant/lucene/
├── TURBOQUANT_LUCENE_INTEGRATION_PLAN.md   — Design & architecture (589 lines)
├── TURBOQUANT_IMPLEMENTATION_PLAN.md       — Phased execution plan (323 lines)
├── REVIEW_FEEDBACK.md                      — Expert review audit trail (562 lines)
└── SESSION_LOG.md                          — This document
```

---

## Session 2: Implementation (2026-03-30 ~21:09 – 2026-03-31 ~13:18 UTC)

> Participants: olexandb + AI assistant (Kiro CLI, Team Lead role)

### Execution Summary

All 5 phases of the TurboQuant implementation plan were executed, tested, debugged, and validated.

### Phase 1: Core Algorithm (21:09–21:25)

Implemented 4 source files + 4 test files. All 32 unit tests pass.

| Deliverable | Status |
|---|---|
| `TurboQuantEncoding.java` — enum BITS_2/3/4/8 | ✅ |
| `BetaCodebook.java` — precomputed Lloyd-Max centroids for N(0,1) | ✅ |
| `HadamardRotation.java` — block-diagonal FWHT + permutation + sign flip | ✅ |
| `TurboQuantBitPacker.java` — optimized packing for b=2,3,4,8 | ✅ |

Centroid values computed by running Lloyd's algorithm via scipy on the reference implementation's scalar_quantizer.py. MSE distortion at d=4096 b=4 = 0.0095, matching paper's 0.009.

### Phase 2: Codec Integration (21:25–21:50)

Implemented 6 source files. 53/53 inherited `BaseKnnVectorsFormatTestCase` tests pass.

**Bugs found and fixed:**
1. **HNSW writer assertion** — `FieldWriter.isFinished()` didn't match Lucene104 pattern. Fix: `finish()` asserts delegate finished, then sets own flag.
2. **File handle leak during merge** — opened `.vetq` for reading while still writing. Fix: use temp file for scorer supplier.
3. **Byte vector UnsupportedOperationException** — reader threw instead of delegating. Fix: delegate to raw reader.

### Phase 3: SIMD Scoring (21:50–22:00)

Created `TurboQuantScoringUtil.java` with LUT-based scoring that operates directly on packed bytes. Replaced naive scorer. All 89 tests pass, no regression.

### Phase 4: Quality Validation (22:00–22:10)

Created `TestTurboQuantQuality.java` with recall, edge case, merge stress, and similarity×encoding matrix tests. 97 tests pass.

### Phase 5: Documentation (22:10–22:15)

Created `package-info.java`, added `CHANGES.txt` entry, verified ASF license headers on all 21 Java files.

### Completeness Audit (2026-03-31 12:35–12:55)

Re-read the full implementation plan and identified gaps:
- Added block-diagonal MSE quality test (Phase 1 gate)
- Added `TestTurboQuantHnswVectorsFormatParams` — testLimits, testToString (Phase 2.6a)
- Added 10-segment merge stress test (Phase 4.4)
- Added recall test at d=768 (Phase 4.1)
- Added all similarity × all encoding test (Phase 4.2)
- Created JMH benchmark `TurboQuantBenchmark.java` (Phase 4.6)
- Added `CHANGES.txt` entry (Phase 5.2)
- Exported turboquant package from codecs module-info
- Added codecs dependency to benchmark-jmh module

107 tests pass after audit.

### Full Test Suite Integration (12:55–13:05)

Added `TurboQuantHnswVectorsFormat` to `RandomCodec`'s knn format pool in `lucene/test-framework`. This means any Lucene test using the random codec may randomly select TurboQuant.

**Bug found:** DOT_PRODUCT scorer multiplied by `docNorm`, producing scores > 1.0. Fix: DOT_PRODUCT uses `(1 + dot) / 2` without docNorm; MAXIMUM_INNER_PRODUCT uses `scaleMaxInnerProductScore(dot * docNorm)`.

504 core vector tests pass with TurboQuant in the random rotation.

### JMH Benchmarks (13:05)

```
Benchmark                              (bits)  (dim)   Mode       Score   Units
TurboQuantBenchmark.dotProductScoring       4   4096  thrpt  313,617   ops/s
TurboQuantBenchmark.hadamardRotation        4   4096  thrpt   32,125   ops/s
TurboQuantBenchmark.quantize                4   4096  thrpt    8,169   ops/s
```

### Recall Validation (13:09–13:15)

Initial recall tests used small dimensions. Proper validation at plan-specified dimensions revealed:

**Brute-force quantization quality (no HNSW):**
- d=768 b=4: 0.856 recall@10 — quantization quality is good
- d=768 b=8: 0.980 recall@10 — near-lossless

**HNSW search recall (with over-retrieval searchK=50 for top-10):**
- d=4096 b=4: 0.905 recall@10
- d=768 b=4: 0.850 recall@10
- d=768 b=8: 0.980 recall@10
- d=768 b=3: 0.810 recall@10
- d=768 b=2: 0.680 recall@10

**Key finding:** HNSW greedy traversal with quantized distances needs over-retrieval (searchK > k) to compensate for approximation error. This is the same behavior as scalar quantization.

### Implementation Report (13:05–13:09)

Created `TURBOQUANT_IMPLEMENTATION_REPORT.md` (516 lines) covering architecture decisions, implementation details, test results, benchmarks, bugs found, and deferred items. Updated with real recall data after validation.

### Quip Publish Attempt (13:16–13:19)

Attempted to publish report to `quip-amazon.com/RFA5AFoM2ikW/Turboquant`. Failed due to expired Midway credentials. User ran `mwinit --aea` but token propagation may need more time.

---

### Final Artifact Summary

```
Source (12 files, 2,090 lines):
  TurboQuantEncoding.java, BetaCodebook.java, HadamardRotation.java,
  TurboQuantBitPacker.java, TurboQuantScoringUtil.java,
  TurboQuantFlatVectorsFormat.java, TurboQuantFlatVectorsWriter.java,
  TurboQuantFlatVectorsReader.java, OffHeapTurboQuantVectorValues.java,
  TurboQuantVectorsScorer.java, TurboQuantHnswVectorsFormat.java,
  package-info.java

Tests (10 files, 1,290 lines):
  TestTurboQuantEncoding, TestBetaCodebook, TestHadamardRotation,
  TestTurboQuantBitPacker, TestTurboQuantScoringUtil,
  TestTurboQuantHnswVectorsFormat, TestTurboQuantHnswVectorsFormatParams,
  TestTurboQuantHighDim, TestTurboQuantQuality,
  TestTurboQuantBruteForceRecall, TestTurboQuantRecall

Benchmarks (1 file):
  TurboQuantBenchmark.java (JMH)

Config changes:
  META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat (SPI)
  codecs module-info.java (export)
  benchmark-jmh build.gradle + module-info.java (dependency)
  test-framework RandomCodec.java (random rotation)
  CHANGES.txt (new feature entry)

Docs:
  TURBOQUANT_IMPLEMENTATION_REPORT.md
  TURBOQUANT_IMPLEMENTATION_PLAN.md (27/27 gates checked)
```

### Git Commits (11 total)

```
2f5ead3 docs: Update implementation report with real recall data
19cd595 test: Proper recall validation at plan-specified dimensions
e06ed0c feat: All plan gates complete — zero unchecked items
c4f073b docs: Mark randomized codec gate as complete
4dd51c4 fix: Fix scorer formulas and add to RandomCodec
1a757b8 fix: Complete all remaining plan items
4cce13b docs: Complete Phase 5 — package-info.java
d89bc82 feat: Complete Phase 4 — quality validation
48d000c feat: Complete Phase 3 — LUT-based scoring
97be63d feat: Complete Phase 2 gate — d=4096 and d=768 verified
64091e4 fix: Fix all Phase 2 test failures — 53/53 pass
5c4ebe9 feat: Implement Phase 1 + Phase 2 scaffold
```
