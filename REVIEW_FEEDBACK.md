# Community Expert Review: TurboQuant Lucene Integration Plan

## Review Rounds

- **Round 1** — Architecture, Performance, Compatibility (incorporated)
- **Round 2** — API Reuse, Extensibility, Backward Compatibility (below)

## Round 1 Reviewers

- **Reviewer A** — Lucene Codec Architecture (PMC-level)
- **Reviewer B** — SIMD / Performance Engineering
- **Reviewer C** — Compatibility & Production Readiness

---

## Reviewer A: Lucene Codec Architecture

### BLOCKER: Wrong abstraction layer

The plan proposes `TurboQuantVectorsFormat extends KnnVectorsFormat` with a "delegate for HNSW graph." This is backwards. Looking at the actual Lucene 10.4 codebase:

- `FlatVectorsFormat` is the abstraction for how vectors are stored and scored (quantized or raw)
- `Lucene99HnswVectorsWriter` takes a `FlatVectorsWriter` as a constructor parameter
- `Lucene104ScalarQuantizedVectorsFormat extends FlatVectorsFormat` — this is the pattern

**TurboQuant should be a `FlatVectorsFormat`, not a `KnnVectorsFormat`.** The HNSW graph is orthogonal. Users compose them:

```java
new Lucene104HnswScalarQuantizedVectorsFormat(...)  // HNSW + scalar quant
// becomes:
new SomeHnswTurboQuantVectorsFormat(...)             // HNSW + turboquant
```

Or more cleanly, TurboQuant is just a `FlatVectorsFormat` that plugs into the existing `Lucene99HnswVectorsWriter`. This is exactly how `Lucene104ScalarQuantizedVectorsFormat` works — it provides a `FlatVectorsWriter` and `FlatVectorsReader`, and the HNSW format wraps it.

**Impact:** The entire module structure, class hierarchy, and file format sections need revision.

### BLOCKER: Must implement `FlatVectorsScorer`

The plan mentions a `TurboQuantScorer` but doesn't address the `FlatVectorsScorer` interface, which is how Lucene's HNSW graph builder and searcher get scoring functions. You need:

```java
public class TurboQuantVectorsScorer implements FlatVectorsScorer {
    RandomVectorScorerSupplier getRandomVectorScorerSupplier(...);
    RandomVectorScorer getRandomVectorScorer(..., float[] target);
    RandomVectorScorer getRandomVectorScorer(..., byte[] target);
}
```

This is the hot path. The scorer must handle the query rotation and LUT-based distance computation.

### ISSUE: `getMaxDimensions()` hardcoded to 1024

Every existing Lucene vector format returns 1024 from `getMaxDimensions()`. The plan targets d=4096 embeddings. This requires either:
1. Overriding `getMaxDimensions()` to return a higher value (e.g., 4096 or 16384)
2. Ensuring the upstream `Lucene99HnswVectorsWriter` respects the flat format's max dimensions

This is actually a TurboQuant advantage — the algorithm works better at higher dimensions (Gaussian approximation improves). Advertise this.

### ISSUE: File extensions conflict risk

Custom extensions `.tqv`, `.tqn`, `.tqm`, `.tqg` are fine for the experimental codec. But the plan also uses `.vec` for raw vectors — this conflicts with `Lucene99FlatVectorsFormat` which uses `.vec`. Since TurboQuant should delegate raw vector storage to `Lucene99FlatVectorsFormat` (like scalar quant does), this resolves itself.

### SUGGESTION: Follow the Lucene104 pattern exactly

The cleanest integration:
- `TurboQuantFlatVectorsFormat extends FlatVectorsFormat` — stores quantized + delegates raw to `Lucene99FlatVectorsFormat`
- `TurboQuantFlatVectorsWriter extends FlatVectorsWriter` — quantizes on write
- `TurboQuantFlatVectorsReader extends FlatVectorsReader` — reads quantized, provides scorer
- Companion `TurboQuantHnswVectorsFormat extends KnnVectorsFormat` — composes HNSW + TurboQuant flat format (optional convenience class)

---

## Reviewer B: SIMD / Performance Engineering

### CRITICAL: d=4096 changes everything for Hadamard

The plan was written around d=768. At d=4096:

1. **Hadamard is perfect** — 4096 = 2^12, exact power of 2. No padding, no block-diagonal hacks. The entire §9 risk about "d=768 not power of 2" and the block-diagonal mitigation become irrelevant for the primary use case.

2. **Rotation cost scales:** O(d log d) = 4096 × 12 = 49,152 FLOPs per query per segment. Still small vs HNSW traversal at d=4096, but worth noting.

3. **Quantized vector size at b=4:** 4096 × 4/8 = 2048 bytes per vector. For 1M vectors: ~1.95 GB quantized vs ~15.6 GB float32. Still a 8x win.

4. **Memory bandwidth is the real bottleneck at d=4096.** Each HNSW hop reads 2048 bytes. With ~100 hops per query, that's ~200 KB per query. The LUT-based scoring becomes critical — avoid dequantizing to float32 (which would be 16 KB per vector).

### CRITICAL: LUT scoring strategy needs rethinking for d=4096

The plan's scoring approach (per-dimension gather + fma) is O(d) per candidate. At d=4096, that's 4096 multiply-adds. The better approach for b=4:

**Precompute a 16-entry LUT per query:** `lut[j] = 0` initially, then for each candidate, accumulate `lut[idx[i]] += q_rot[i]`. Wait — that's per-candidate, not per-query.

Actually, the correct optimization is **ADC (Asymmetric Distance Computation)**:
1. Per query, precompute `q_rot` (one Hadamard transform)
2. Per candidate, the dot product is `sum(q_rot[i] * centroids[idx[i]])` for i in [0, d)
3. Since there are only 16 centroid values, precompute `partial_sums[j] = sum of q_rot[i] where idx[i] == j` — but this requires knowing idx first, so it doesn't help.

The gather+fma approach is actually correct. But at d=4096 with b=4, each vector is 2048 bytes (nibble-packed). The inner loop processes 2 indices per byte. With AVX-512, we process 64 bytes (128 indices) per iteration → 32 iterations for d=4096. This is fast.

### ISSUE: Off-heap storage is mandatory at d=4096

At d=4096, b=4, 1M vectors = 1.95 GB of quantized data. This MUST be off-heap (mmap'd `IndexInput`), not loaded into Java heap. The plan doesn't discuss off-heap vs on-heap. The existing `OffHeapScalarQuantizedVectorValues` pattern must be followed.

### SUGGESTION: Add d=4096 to all storage/performance calculations

The plan's examples use d=768. The primary use case is d=4096. Update all tables.

---

## Reviewer C: Compatibility & Production Readiness

### ISSUE: Segment merge is always O(n·d·log d) — no skip optimization

Lucene's scalar quantization can skip re-quantization during merge when quantiles haven't shifted significantly. TurboQuant always re-quantizes because each segment has a different rotation matrix.

**Mitigation option:** Use a global rotation seed (e.g., derived from field name hash) so all segments share the same rotation. Then merge never needs re-quantization — just copy quantized bytes. This is safe because TurboQuant is data-oblivious; the rotation doesn't depend on data.

**This is a major performance win for merge-heavy workloads.** The plan should make this the default.

### ISSUE: Codec versioning

The plan names the format `TurboQuant10` but doesn't define version constants (`VERSION_START`, `VERSION_CURRENT`) or use `CodecUtil.writeIndexHeader`/`checkIndexHeader`. Every Lucene format must do this for forward/backward compatibility detection.

### ISSUE: CheckIndex support

`CheckIndex` must be able to validate TurboQuant segments. This means:
- Checksums on all files (via `CodecUtil`)
- Ability to verify quantized vectors round-trip correctly against raw vectors
- Report quantization statistics (mean MSE, max MSE)

### ISSUE: `toString()` for diagnostics

Every format must have a meaningful `toString()` for debugging. Include bit-width, rotation type, max dimensions.

### SUGGESTION: Merge worker support

`Lucene104HnswScalarQuantizedVectorsFormat` supports `numMergeWorkers` and `TaskExecutor` for parallel merge. The TurboQuant companion HNSW format should too.

### SUGGESTION: `ScalarEncoding`-like enum for bit-width

Instead of raw `int bitsPerCoordinate`, consider an enum:
```java
public enum TurboQuantEncoding {
    BITS_2(2), BITS_3(3), BITS_4(4), BITS_8(8);
}
```
This prevents invalid values and makes the API self-documenting. Skip b=1 (too lossy for NN search) and b=5,6,7 (odd bit-packing, marginal benefit over b=4 or b=8).

---

## Consolidated Action Items

| # | Priority | Item | Reviewer |
|---|----------|------|----------|
| 1 | BLOCKER | Restructure as `FlatVectorsFormat`, not `KnnVectorsFormat` | A |
| 2 | BLOCKER | Implement `FlatVectorsScorer` interface | A |
| 3 | CRITICAL | Raise `getMaxDimensions()` to support d=4096 | A |
| 4 | CRITICAL | All examples/calculations must use d=4096 as primary | B |
| 5 | CRITICAL | Off-heap (mmap) storage for quantized vectors | B |
| 6 | HIGH | Global rotation seed to avoid merge re-quantization | C |
| 7 | HIGH | Codec versioning with `CodecUtil` headers/checksums | C |
| 8 | HIGH | d=4096 is power of 2 — simplify Hadamard section; block-diagonal for d=768 | B |
| 9 | MEDIUM | Delegate raw vector storage to `Lucene99FlatVectorsFormat` | A |
| 10 | MEDIUM | CheckIndex support | C |
| 11 | MEDIUM | Merge worker / TaskExecutor support | C |
| 12 | LOW | Enum for bit-width instead of raw int | C |
| 13 | LOW | Meaningful `toString()` | C |


---

## Round 2: API Reuse, Extensibility, Backward Compatibility

### Reviewers

- **Reviewer D** — Lucene Committer, API design & extensibility
- **Reviewer E** — Lucene PMC, backward compatibility & release process

---

### Reviewer D: API Reuse & Extensibility

#### D1. CRITICAL: Don't invent `TurboQuantEncoding` — extend `ScalarEncoding`

Lucene already has `QuantizedByteVectorValues.ScalarEncoding` with a wire format, bits-per-dim, packing logic, and `getDocPackedLength()` / `getDiscreteDimensions()`. It already supports 1-bit, 2-bit, 4-bit, 7-bit, and 8-bit encodings.

TurboQuant's b=2,3,4,8 maps directly onto this. Rather than a parallel enum, **add new entries to `ScalarEncoding`** or, if that's too invasive for an experimental codec, create `TurboQuantEncoding` that delegates to `ScalarEncoding` for packing math. At minimum, reuse `ScalarEncoding.getDocPackedLength()` and `getDiscreteDimensions()` rather than reimplementing bit-packing arithmetic.

However — `ScalarEncoding` is tightly coupled to `OptimizedScalarQuantizer` and its corrective terms (centroid, quantized component sum). TurboQuant doesn't use centroids or corrective terms in the same way. So extending `ScalarEncoding` directly would pollute it.

**Recommendation:** Keep `TurboQuantEncoding` as a separate enum but reuse the packing math patterns from `ScalarEncoding`. Don't extend `ScalarEncoding` itself. This is the right trade-off between reuse and clean separation.

#### D2. HIGH: Reuse `Lucene99FlatVectorsFormat` as raw vector delegate — exactly like Lucene104 does

The plan says "delegate raw vector storage to Lucene99FlatVectorsFormat." Good — but be explicit: the writer must hold a `FlatVectorsWriter rawVectorDelegate` field and call `rawVectorDelegate.addField()`, `rawVectorDelegate.flush()`, `rawVectorDelegate.mergeOneField()`, and `rawVectorDelegate.finish()` at the right lifecycle points. This is exactly what `Lucene104ScalarQuantizedVectorsWriter` does (line 77, 103, 128, 141, 319, 331, 333, 341).

The reader must hold a `FlatVectorsReader rawVectorsReader` for rescore and `getFloatVectorValues()`.

#### D3. HIGH: Implement `mergeOneFieldToIndex()` properly

This is the method `Lucene99HnswVectorsWriter` calls during merge to get a scorer over the newly merged flat vectors. The scalar quant writer does complex work here: re-quantizes vectors, writes to temp files, returns a `CloseableRandomVectorScorerSupplier`.

For TurboQuant with global rotation: merge is simpler (byte copy), but you still need to return a valid `CloseableRandomVectorScorerSupplier` over the merged quantized data so the HNSW graph can be rebuilt. Don't skip this — it's how the HNSW merge works.

#### D4. MEDIUM: Reuse `VectorUtil` for SIMD primitives

`VectorUtil` already has Panama Vector API-optimized `dotProduct()`, `squareDistance()`, `int4DotProduct()`, etc. For TurboQuant scoring, you'll need a new primitive (LUT-gather-fma), but the pattern should follow `VectorUtil` conventions:
- Static method in `VectorUtil` or a new `TurboQuantVectorUtil`
- Let the JVM's auto-vectorization and Panama API handle SIMD
- Register with `VectorizationProvider` if using platform-specific intrinsics

#### D5. MEDIUM: `getFloatVectorValues()` and `getByteVectorValues()` contracts

`FlatVectorsReader` inherits from `KnnVectorsReader` which requires `getFloatVectorValues()` and `getByteVectorValues()`. For TurboQuant:
- `getFloatVectorValues()` → delegate to `rawVectorsReader.getFloatVectorValues()` (for rescore, scripts, etc.)
- `getByteVectorValues()` → throw `UnsupportedOperationException` (TurboQuant only handles float32 input)

This is the same pattern as `Lucene104ScalarQuantizedVectorsReader`.

#### D6. LOW: Consider `Accountable` / `ramBytesUsed()` carefully

`FlatVectorsReader` implements `Accountable`. Your reader must report:
- Shallow size of the reader object
- Size of cached rotation matrix (d × 4 bytes for signs, d × 4 bytes for permutation)
- Size of field metadata map
- Delegate to `rawVectorsReader.ramBytesUsed()`

And `getOffHeapByteSize()` must report the mmap'd quantized data size per field.

---

### Reviewer E: Backward Compatibility & Release Process

#### E1. HIGH: Module placement — `lucene/codecs/` is correct but has implications

Experimental codecs in `lucene/codecs/` module:
- No backward compatibility guarantee (format can change every release)
- Not included in the default `Codec` — users must explicitly select it
- SPI registration in `META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat`
- Must NOT be registered in `META-INF/services/org.apache.lucene.codecs.Codec` (don't create a full Codec, just the format)

The plan's `TurboQuantCodec.java` should be removed. Users compose via `PerFieldKnnVectorsFormat` or a custom `FilterCodec`. A standalone Codec is unnecessary and creates a maintenance burden.

#### E2. HIGH: Version constants and file format stability

Even for experimental codecs, define:
```java
static final int VERSION_START = 0;
static final int VERSION_CURRENT = VERSION_START;
```
And use `CodecUtil.writeIndexHeader` / `checkIndexHeader` on every file. This lets us detect format changes and fail fast rather than silently corrupt.

When the format changes, bump `VERSION_CURRENT` and add read-path handling for old versions (or reject them with a clear error).

#### E3. MEDIUM: Don't add to `lucene/core` — keep in `lucene/codecs`

The plan correctly places code in `lucene/codecs/`. Do NOT add anything to `lucene/core` (no new `VectorUtil` methods, no new `ScalarEncoding` entries). The experimental codec should be self-contained. If it graduates to default, then we move things to core.

Exception: if the Hadamard transform proves generally useful, it could eventually go to `lucene/core/src/.../util/`, but not in the initial contribution.

#### E4. MEDIUM: Test infrastructure

Extend `BaseKnnVectorsFormatTestCase` for the HNSW+TurboQuant format. This gives you dozens of pre-existing tests for free (indexing, searching, merging, filtering, sorting, multi-segment, etc.). This is how all vector formats are tested.

```java
public class TestTurboQuantHnswVectorsFormat extends BaseKnnVectorsFormatTestCase {
    @Override
    protected KnnVectorsFormat getKnnVectorsFormat() {
        return new TurboQuantHnswVectorsFormat();
    }
}
```

#### E5. LOW: Gradle build file

Add `lucene/codecs/build.gradle` dependency on `lucene/core` (already exists). No new external dependencies — TurboQuant is pure math (Hadamard, Lloyd-Max centroids are precomputed constants). This is a strength.

---

## Round 2 Consolidated Action Items

| # | Priority | Item | Reviewer |
|---|----------|------|----------|
| 14 | CRITICAL | Keep `TurboQuantEncoding` separate from `ScalarEncoding`, but reuse packing math patterns | D |
| 15 | HIGH | Explicit `rawVectorDelegate` lifecycle (addField/flush/mergeOneField/finish) | D |
| 16 | HIGH | Implement `mergeOneFieldToIndex()` returning `CloseableRandomVectorScorerSupplier` | D |
| 17 | HIGH | Remove standalone `TurboQuantCodec.java` — use `PerFieldKnnVectorsFormat` composition | E |
| 18 | HIGH | Extend `BaseKnnVectorsFormatTestCase` for free test coverage | E |
| 19 | MEDIUM | `getFloatVectorValues()` delegates to raw reader; `getByteVectorValues()` throws | D |
| 20 | MEDIUM | Proper `ramBytesUsed()` and `getOffHeapByteSize()` | D |
| 21 | MEDIUM | Keep everything in `lucene/codecs/`, nothing in `lucene/core` | E |
| 22 | MEDIUM | Follow `VectorUtil` patterns for SIMD scoring primitives | D |
| 23 | LOW | No external dependencies — pure precomputed math | E |


---

## Round 3: Testing Strategy Review

### Reviewers

- **Reviewer F** — Lucene PMC, test framework maintainer
- **Reviewer G** — Lucene committer, randomized testing & edge cases

---

### Reviewer F: Test Framework Integration

#### F1. CRITICAL: `BaseKnnVectorsFormatTestCase` gives you ~50 tests but has assumptions

Extending `BaseKnnVectorsFormatTestCase` is mandatory. It provides tests for:
- Basic indexing, field construction, illegal args
- Multi-segment merging with different fields
- Sorted index support
- Sparse vectors, deleted docs
- Random stress tests (float + byte)
- Recall validation across all 4 similarity functions
- CheckIndex integrity
- Off-heap byte size reporting
- Writer RAM estimation
- AddIndexes from different codecs

**But it has assumptions you must handle:**

1. **`assertOffHeapByteSize()`** checks for keys `"vec"`, `"vex"`, `"veq"` in the off-heap map. TurboQuant uses different extensions. You must either:
   - Return `"vec"` key for raw vectors (delegated to Lucene99FlatVectorsFormat — this happens automatically)
   - Return `"vex"` key for HNSW graph (delegated to Lucene99HnswVectorsReader — automatic)
   - Return your quantized data under a key like `"tqvec"` — the test checks `totalByteSize > 0` which will pass, but the `hasQuantized()` check uses class name heuristic (`name.contains("quantized")`). Your reader class name should contain "quantized" or "turboquant" — or override `assertOffHeapByteSize()`.

2. **`getQuantizationBits()`** defaults to 8. Override to return your actual bit-width (e.g., 4) so epsilon tolerances in float comparison tests are correct.

3. **`supportsFloatVectorFallback()`** — return `false` (TurboQuant doesn't support reading raw floats from quantized-only storage).

4. **`testIllegalDimensionTooLarge()`** — this test uses `getMaxDimensions()`. Since TurboQuant returns 16384 instead of 1024, the test will try to create vectors with dim > 16384. This should work fine.

5. **`randomVectorEncoding()`** returns BYTE or FLOAT32 randomly. TurboQuant only supports FLOAT32. Override to always return FLOAT32, or handle BYTE by delegating to the raw format.

#### F2. HIGH: Override `testRecall()` with TurboQuant-appropriate thresholds

The base `testRecall()` asserts recall ≥ 0.5. For b=4 TurboQuant this should easily pass. But you should add a TurboQuant-specific recall test that:
- Tests at d=768 AND d=4096 (your two primary dimensions)
- Tests at b=2, b=4, b=8 to validate quality degrades gracefully
- Compares against exact brute-force search
- Asserts recall@10 ≥ 0.9 for b=4 at d=4096 (the sweet spot)

#### F3. HIGH: `BaseIndexFileFormatTestCase` provides critical infrastructure

This parent class provides:
- `testMergeStability()` — suppressed for kNN (graph non-determinism), but still runs
- `testMultiClose()` — verifies reader/writer close is idempotent
- `testRandomExceptions()` — injects random IOExceptions during indexing/searching to verify graceful failure
- `testCheckIntegrityReadsAllBytes()` — verifies `checkIntegrity()` reads every byte of every file

These are critical for production readiness. The `testRandomExceptions()` test is particularly brutal — it will find resource leaks and missing try-finally blocks.

---

### Reviewer G: Randomized Testing & Edge Cases

#### G1. CRITICAL: Missing test categories

The plan's Phase 4 lists tests but misses several critical categories:

**Algorithm correctness tests (unit level):**
- Hadamard rotation: verify `H·D·x` preserves norm for random x at d=4096, d=768, d=384
- Hadamard rotation: verify `inverseRotate(rotate(x)) == x` within float32 epsilon
- Hadamard block decomposition: verify `decomposeBlocks(768) == [512, 256]`
- Hadamard block decomposition: verify `decomposeBlocks(4096) == [4096]`
- Hadamard block decomposition: verify for all d in [32..8192]
- Codebook: verify precomputed centroids match Lloyd-Max algorithm output
- Codebook: verify MSE distortion at d=4096 matches paper's theoretical values
- Bit-packing: round-trip for all encodings (b=2,3,4,8) at various dimensions
- Bit-packing: edge cases — d=1 (below minimum), d=32 (minimum), d=16384 (maximum)

**Codec integration tests (beyond BaseKnnVectorsFormatTestCase):**
- Single vector per segment (degenerate case)
- Empty segment (zero vectors)
- Very large segment (100K+ vectors at d=4096 if CI resources allow)
- Mixed fields: one field with TurboQuant, another with scalar quant, in same index
- Force merge from N segments to 1: verify byte-copy merge path
- Index sorting with vector fields
- Concurrent indexing + searching
- AddIndexes from a directory using a different codec

**Scoring correctness tests:**
- For each similarity function: quantized score vs exact score, verify error within theoretical MSE bound
- Verify rotation preserves distances: `dist(a, b) == dist(rotate(a), rotate(b))` within epsilon
- Verify query rotation is applied correctly: search results should be identical whether we rotate query or inverse-rotate all docs
- Score monotonicity: if `exact_score(a) > exact_score(b)`, then `quantized_score(a) > quantized_score(b)` with high probability

**Merge-specific tests:**
- Verify byte-copy merge produces identical quantized vectors as fresh quantization (since global rotation)
- Verify merge with deleted docs correctly excludes them
- Verify merge from segments with different vector dimensions fails gracefully
- Verify `mergeOneFieldToIndex()` returns a working `CloseableRandomVectorScorerSupplier`

#### G2. HIGH: Randomized dimension testing

Don't just test d=768 and d=4096. The randomized test framework should pick random dimensions:
```java
int dim = random().nextInt(32, 4097); // covers power-of-2 and non-power-of-2
```
This will exercise the block-diagonal Hadamard path for non-power-of-2 dims.

#### G3. HIGH: Randomized encoding testing

Like `TestLucene104HnswScalarQuantizedVectorsFormat` randomly picks a `ScalarEncoding` in `setUp()`, your test should randomly pick a `TurboQuantEncoding`:
```java
@Before
public void setUp() throws Exception {
    var encodings = TurboQuantEncoding.values();
    encoding = encodings[random().nextInt(encodings.length)];
    format = new TurboQuantHnswVectorsFormat(encoding, 16, 100);
    super.setUp();
}
```

#### G4. MEDIUM: Stress test the Hadamard transform with adversarial inputs

- All-zeros vector (should be handled — norm=0 edge case)
- One-hot vectors (e_i for each i) — worst case for rotation quality
- Vectors with extreme values (Float.MAX_VALUE / d)
- Vectors with subnormal floats
- Vectors where all coordinates are identical

#### G5. MEDIUM: Test `CheckIndex` integration

`CheckIndex` should:
- Verify CodecUtil checksums on .tqvec and .tqmeta
- Verify vector count in metadata matches actual stored vectors
- Verify quantized vectors can be dequantized and compared against raw vectors
- Report per-field quantization statistics (mean MSE, max MSE, encoding, dimension)

#### G6. LOW: Performance regression test (JMH)

Add a JMH benchmark in `lucene/benchmark-jmh/` that measures:
- Quantization throughput (vectors/sec) at d=4096, b=4
- Hadamard rotation throughput at d=4096
- Quantized dot product throughput at d=4096
- Compare against scalar quantization at same bit-width

This isn't a correctness test but prevents performance regressions across releases.

---

## Round 3 Consolidated Action Items

| # | Priority | Item | Reviewer |
|---|----------|------|----------|
| 24 | CRITICAL | Override `randomVectorEncoding()` → FLOAT32 only | F |
| 25 | CRITICAL | Override `getQuantizationBits()` → return actual bit-width | F |
| 26 | CRITICAL | Add algorithm correctness unit tests (rotation, codebook, bit-packing) | G |
| 27 | HIGH | Override `testRecall()` with d=4096 and d=768 specific thresholds | F |
| 28 | HIGH | Randomized dimension testing (d ∈ [32, 4097]) | G |
| 29 | HIGH | Randomized encoding testing (random TurboQuantEncoding in setUp) | G |
| 30 | HIGH | Merge-specific tests (byte-copy correctness, deleted docs, scorer supplier) | G |
| 31 | HIGH | Scoring correctness: quantized vs exact within theoretical MSE bound | G |
| 32 | MEDIUM | Adversarial input tests (zero vector, one-hot, extreme values) | G |
| 33 | MEDIUM | CheckIndex integration with quantization statistics | G |
| 34 | ~~MEDIUM~~ RESOLVED | `assertOffHeapByteSize()` compatibility — reuse `"veq"` extension key + implement `QuantizedVectorsReader` | F |
| 35 | LOW | JMH performance benchmark in `lucene/benchmark-jmh/` | G |


---

## Round 4: Addressing Mike McCandless's 6 Gaps

### Expert Panel Responses

---

#### Gap 1: Global rotation seed fragility across AddIndexes / schema changes

**Expert consensus:** The field name is stable across `AddIndexes` — Lucene remaps field *numbers* but preserves field *names*. So `MurmurHash3(fieldName)` is safe for `AddIndexes`.

However, the real risk is **user confusion**: if someone reindexes data from field "embedding_v1" to "embedding_v2", the rotations differ and the quantized representations are incompatible. This isn't a bug — it's expected behavior — but it should be documented.

**Resolution:** Use field name as seed (confirmed safe), but add a constructor parameter `rotationSeed` for advanced users who need explicit control. Default = derive from field name. Store the actual seed used in `.vemtq` metadata so it can be verified during `AddIndexes`.

```java
// Default: seed from field name
new TurboQuantFlatVectorsFormat(TurboQuantEncoding.BITS_4)

// Advanced: explicit seed for cross-field compatibility
new TurboQuantFlatVectorsFormat(TurboQuantEncoding.BITS_4, 42L)
```

During merge, verify source segments' rotation seeds match the target. If they don't (e.g., `AddIndexes` from an index with a different explicit seed), fall back to re-quantization from raw vectors.

---

#### Gap 2: Float32 numerical stability of Hadamard at d=4096

**Expert consensus (numerical methods):** The Walsh-Hadamard transform is a sequence of additions and subtractions of same-magnitude values. Unlike FFT, there are no multiplications by twiddle factors that could amplify error. The worst-case rounding error for a d-point WHT in float32 is O(√(log d) · ε_mach) per coordinate, where ε_mach ≈ 6e-8.

For d=4096 (12 levels): worst-case per-coordinate error ≈ √12 × 6e-8 ≈ 2e-7. The quantization boundary spacing at b=4 for d=4096 is approximately `2/(16·√4096)` ≈ 0.002. The rounding error is 5 orders of magnitude smaller than the boundary spacing.

**Resolution:** Float32 is fine. No need for double. Add a unit test that verifies `||rotate(x)||² == ||x||²` within 1e-5 relative error at d=4096 over 10K random vectors. This is sufficient validation.

---

#### Gap 3: File extension reuse (.veq) creates confusion

**Expert consensus (codec maintainers):** Looking at the actual codebase, the convention is clear — **different format types use different extensions**: raw=`vec`, scalar quant=`veq`, binary quant=`veb`. Extensions ARE reused across *versions* of the same type (Lucene99 and Lucene104 both use `veq` for scalar quant), but never across different format types.

TurboQuant is a fundamentally different format type. It should have its own extensions.

**Resolution:** Use unique extensions:
- `.vetq` — TurboQuant quantized vector data
- `.vemtq` — TurboQuant metadata

Override `assertOffHeapByteSize()` in the test class to check for `"vetq"` instead of `"veq"`. The `hasQuantized()` detection works via `QuantizedVectorsReader` interface (instanceof check), not extension names.

---

#### Gap 4: No plan for when quantized search quality is unacceptable / graph building scorer

**Expert consensus (vector search maintainers):** Looking at `Lucene104ScalarQuantizedVectorsWriter.mergeOneFieldToIndex()`, the HNSW graph IS built using quantized distances during merge. The `CloseableRandomVectorScorerSupplier` returned is over quantized data. This is the standard pattern — the graph quality depends on quantized distance quality.

For TurboQuant, this means the HNSW graph quality at b=2 may be poor. The mitigation is the same as scalar quant: users can over-retrieve (higher `k` in kNN search) and rescore with raw vectors.

**Resolution:** TurboQuant's `mergeOneFieldToIndex()` returns a scorer over quantized data (same as scalar quant). Document that b=2 may require over-retrieval + rescoring. Add a recall test at b=2 that validates this: recall@10 with efSearch=25 should be ≥ 0.7.

No "two-pass" mode needed — this would be a departure from Lucene's architecture and isn't justified by the data.

---

#### Gap 5: Memory accounting during indexing at d=4096

**Expert consensus (IndexWriter experts):** Looking at `Lucene104ScalarQuantizedVectorsWriter.FieldWriter`, it does NOT buffer quantized vectors in heap. It only buffers:
- Raw vectors via the delegate `flatFieldVectorsWriter` (this is the big cost)
- Per-vector metadata (magnitudes, dimension sums) — small

Quantization happens at flush time, streaming through the buffered raw vectors.

TurboQuant should follow the same pattern:
- Buffer raw vectors via delegate (16 KB × N vectors — same cost as any format)
- At flush time, iterate through buffered vectors, rotate + quantize, write to .vetq
- The rotation itself needs a temporary d-float buffer (16 KB at d=4096) — reused per vector, not per document

**Resolution:** No additional heap buffering needed beyond the raw delegate. `ramBytesUsed()` reports:
- `flatFieldVectorsWriter.ramBytesUsed()` (the raw vectors — dominant cost)
- Shallow size of the TurboQuant field writer
- The rotation scratch buffer (16 KB, shared)

This is actually *less* heap than scalar quant, which also buffers magnitudes and dimension sums.

---

#### Gap 6: Block-diagonal Hadamard theoretical backing

**Expert consensus (randomized linear algebra):** The concern is valid. A block-diagonal Hadamard with random permutation is NOT equivalent to a full random rotation. The coordinates within each block are well-mixed, but cross-block mixing relies solely on the permutation.

However, for quantization purposes, what matters is that each coordinate's marginal distribution is close to N(0, 1/d). A random permutation + sign flip + block-Hadamard achieves this: each output coordinate is a sum of ±1 weighted input coordinates (within its block), and the permutation ensures the input coordinates are randomly selected.

The key question is: are the coordinates sufficiently *independent*? For a full random rotation, any pair of output coordinates has correlation O(1/d). For block-diagonal, coordinates within the same block have correlation O(1/block_size), and coordinates across blocks have correlation 0 (exactly independent). For d=768 with blocks (512, 256), the worst case is O(1/256) ≈ 0.004 — negligible.

**Resolution:** Block-diagonal is theoretically sound for quantization purposes. But add empirical validation:
- Phase 1 unit test: compare MSE distortion of block-diagonal vs full QR rotation at d=768 over 10K random vectors
- If distortion differs by > 5%, fall back to seeded QR for non-power-of-2 dims
- Document the empirical results in the codec's package-info.java
