# TurboQuant Lucene Implementation Plan

> Detailed phased implementation plan for the TurboQuant codec.
> See [TURBOQUANT_LUCENE_INTEGRATION_PLAN.md](./TURBOQUANT_LUCENE_INTEGRATION_PLAN.md) for design, architecture, and decisions.

Each phase has explicit entry criteria, deliverables, and gate tests that must pass before proceeding.


### Phase 1: Core Algorithm (2–3 weeks)

**Entry criteria:** None (first phase).

#### 1.1 `TurboQuantEncoding.java`
- Enum with BITS_2(2), BITS_3(3), BITS_4(4), BITS_8(8)
- `bitsPerCoordinate`, `getPackedByteLength(int d)`, `getDiscreteDimensions(int d)` methods
- Wire number for serialization
- **Test:** round-trip wire number serialization for all values; `getPackedByteLength(4096)` returns 2048 for BITS_4

#### 1.2 `BetaCodebook.java`
- Static precomputed canonical Gaussian centroids for b=2,3,4,8 (N(0,1) distribution)
- `centroids(int d, int b)` → returns 2^b float values scaled by 1/√d
- `boundaries(int d, int b)` → returns 2^b + 1 boundary values (midpoints between adjacent centroids)
- **Tests:**
  - Centroids are symmetric around 0 for all bit-widths
  - Centroids match reference implementation values within 1e-4
  - MSE distortion at d=4096 matches paper: 0.117±0.01 (b=2), 0.030±0.005 (b=3), 0.009±0.002 (b=4)
  - MSE distortion computed by: generate 10K random unit vectors, quantize each coordinate, measure mean squared reconstruction error

#### 1.3 `HadamardRotation.java`
- `decomposeBlocks(int d)` → power-of-2 block sizes (binary representation of d)
- `create(int d, long seed)` → constructs rotation with random permutation + sign flip + block-Hadamard
- `rotate(float[] x, float[] out)` → apply rotation in-place, O(d · log(maxBlock))
- `inverseRotate(float[] y, float[] out)` → apply inverse rotation
- Fast Walsh-Hadamard transform implementation for a single power-of-2 block
- **Tests:**
  - `decomposeBlocks(4096) == [4096]`, `decomposeBlocks(768) == [512, 256]`, `decomposeBlocks(384) == [256, 128]`
  - `decomposeBlocks(d)` sums to d for all d in [32..8192]
  - Round-trip: `inverseRotate(rotate(x)) == x` within 1e-5 at d=4096, 768, 384, 100, 33
  - Norm preservation: `||rotate(x)||² == ||x||²` within 1e-5 relative error, 10K random vectors at d=4096
  - Inner product preservation: `rotate(a)·rotate(b) == a·b` within 1e-5, 1K random pairs at d=4096
  - Determinism: same seed produces identical rotation
  - Different seeds produce different rotations
  - Adversarial inputs: zero vector (norm=0 → handle gracefully), one-hot vectors, Float.MAX_VALUE/d, subnormals, all-identical coordinates
  - Block-diagonal quality: MSE distortion of block-diagonal (512+256) vs full QR rotation at d=768 over 10K random vectors — within 5%
  - Float32 stability: `||rotate(x)||²` relative error < 1e-5 at d=4096 over 10K vectors

#### 1.4 `TurboQuantBitPacker.java`
- `pack(byte[] indices, int b, byte[] out)` → pack b-bit indices into bytes
- `unpack(byte[] packed, int b, int d, byte[] out)` → unpack bytes into b-bit indices
- Optimized paths for b=2 (4 per byte), b=3 (8 indices per 3 bytes), b=4 (2 per byte / nibble), b=8 (1 per byte / no-op)
- **Tests:**
  - Round-trip: `unpack(pack(indices)) == indices` for all encodings at d=32, 768, 4096, 16384
  - Boundary values: all-zeros, all-max (2^b - 1), alternating patterns
  - Output length matches `TurboQuantEncoding.getPackedByteLength(d)`
  - Edge case: d=32 (minimum), d=16384 (maximum)

#### Phase 1 Gate

**All of the following must pass before starting Phase 2:**
- [x] All unit tests in `TestHadamardRotation`, `TestBetaCodebook`, `TestTurboQuantBitPacker` pass
- [x] MSE distortion at d=4096 b=4 is within [0.007, 0.011] (paper says 0.009)
- [x] Block-diagonal MSE at d=768 is within 5% of full QR rotation MSE
- [x] Hadamard round-trip error < 1e-5 at d=4096
- [x] No external dependencies (pure Java + precomputed constants)

---

### Phase 2: Codec Integration (3–4 weeks)

**Entry criteria:** Phase 1 gate passed.

#### 2.1 `TurboQuantFlatVectorsFormat.java`
- Extends `FlatVectorsFormat`
- Constructor: `(TurboQuantEncoding encoding)`, `(TurboQuantEncoding encoding, Long rotationSeed)`
- `fieldsWriter(state)` → returns `TurboQuantFlatVectorsWriter`
- `fieldsReader(state)` → returns `TurboQuantFlatVectorsReader`
- `getMaxDimensions(fieldName)` → returns 16384
- `toString()` with encoding, rotation info
- SPI registration in `META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat`

#### 2.2 `TurboQuantFlatVectorsWriter.java`
- Extends `FlatVectorsWriter`
- Holds `FlatVectorsWriter rawVectorDelegate` (Lucene99FlatVectorsFormat)
- Opens `.vemtq` and `.vetq` with `CodecUtil.writeIndexHeader`
- `addField(fieldInfo)` → delegates to raw writer, wraps in `TurboQuantFieldWriter`
- `TurboQuantFieldWriter` inner class:
  - `addValue(docID, vector)` → delegates to raw field writer (buffering)
  - `getVectors()` → delegates to raw field writer
  - `getDocsWithFieldSet()` → delegates
  - `ramBytesUsed()` → raw writer RAM + shallow size (no quantized buffering)
- `flush(maxDoc, sortMap)`:
  - Delegates `rawVectorDelegate.flush()`
  - Streams through buffered raw vectors: rotate, quantize, write to `.vetq`
  - Writes metadata to `.vemtq`
- `mergeOneField(fieldInfo, mergeState)` → delegates to raw writer
- `mergeOneFieldToIndex(fieldInfo, mergeState)`:
  - Delegates `rawVectorDelegate.mergeOneField()`
  - Reads source segment metadata to check rotation seeds
  - If seeds match: byte-copy quantized data
  - If seeds differ: re-quantize from merged raw vectors
  - Writes to temp file, copies to `.vetq`
  - Returns `CloseableRandomVectorScorerSupplier` over merged quantized data
- `finish()` → delegates to raw writer, writes `CodecUtil.writeFooter` on both files
- **Tests (2.2a):**
  - Write 100 vectors at d=768, read back, verify quantized data matches expected
  - Write + flush + read-back round-trip
  - `ramBytesUsed()` is non-zero and doesn't include quantized buffer

#### 2.3 `TurboQuantFlatVectorsReader.java`
- Extends `FlatVectorsReader`, implements `QuantizedVectorsReader`
- Holds `FlatVectorsReader rawVectorsReader`
- Opens `.vemtq` with `CodecUtil.checkIndexHeader`, reads field metadata
- Opens `.vetq` as mmap'd `IndexInput`
- `getFloatVectorValues(field)` → delegates to raw reader
- `getByteVectorValues(field)` → throws `UnsupportedOperationException`
- `getRandomVectorScorer(field, float[] target)` → creates scorer from quantized data
- `getRandomVectorScorer(field, byte[] target)` → throws `UnsupportedOperationException`
- `getQuantizedVectorValues(field)` → returns `OffHeapTurboQuantVectorValues`
- `ramBytesUsed()` → shallow + field map + rotation + raw reader
- `getOffHeapByteSize(fieldInfo)` → merges raw reader map + `Map.of("vetq", dataLength)`
- `checkIntegrity()` → checksums on `.vetq`, `.vemtq` + delegates to raw reader
- `close()` → closes quantized input + raw reader
- **Tests (2.3a):**
  - Write then read: verify `getFloatVectorValues()` returns original vectors
  - `getOffHeapByteSize()` returns non-zero for "vec" and "vetq" keys
  - `checkIntegrity()` passes on valid segment, fails on corrupted file
  - `ramBytesUsed()` > 0

#### 2.4 `OffHeapTurboQuantVectorValues.java`
- Extends `BaseQuantizedByteVectorValues`
- Random access by ordinal into mmap'd `.vetq`
- `vectorValue(int ord)` → reads packed bytes for ordinal
- `size()`, `dimension()`, `iterator()`
- **Tests (2.4a):**
  - Write N vectors, read each by ordinal, verify packed bytes match
  - Iterator visits all docs in order

#### 2.5 `TurboQuantVectorsScorer.java`
- Implements `FlatVectorsScorer`
- `getRandomVectorScorerSupplier(sim, vectorValues)` → returns supplier
- `getRandomVectorScorer(sim, vectorValues, float[] target)`:
  - Rotates query vector once
  - Returns scorer that computes quantized distance per candidate
- `getRandomVectorScorer(sim, vectorValues, byte[] target)` → throws
- Naive (non-SIMD) scoring implementation for correctness — SIMD in Phase 3
- **Tests (2.5a):**
  - Score 100 random query-doc pairs, verify quantized score ≈ exact score within MSE bound
  - All 4 similarity functions produce valid scores (non-NaN, correct sign/range)
  - Scorer supplier creates independent scorers (thread safety)

#### 2.6 `TurboQuantHnswVectorsFormat.java`
- Extends `KnnVectorsFormat`
- Composes `Lucene99HnswVectorsWriter` + `TurboQuantFlatVectorsFormat`
- Constructor parameters: encoding, maxConn, beamWidth, numMergeWorkers, mergeExec, rotationSeed
- Parameter validation (same bounds as Lucene99Hnsw)
- `fieldsWriter(state)` → `new Lucene99HnswVectorsWriter(state, maxConn, beamWidth, turboQuantFlat.fieldsWriter(state), ...)`
- `fieldsReader(state)` → `new Lucene99HnswVectorsReader(state, turboQuantFlat.fieldsReader(state))`
- `getMaxDimensions()` → 16384
- `toString()` with all parameters
- **Tests (2.6a):**
  - `testLimits()` — illegal maxConn, beamWidth, numMergeWorkers throw
  - `testToString()` — output contains encoding and parameters
  - Index 10 vectors, search, verify results returned

#### 2.7 Merge path
- Byte-copy merge when rotation seeds match
- Re-quantization fallback when seeds differ
- `CloseableRandomVectorScorerSupplier` returned correctly
- **Tests (2.7a):**
  - Create 3 segments, force merge to 1, verify all vectors searchable
  - Byte-copy: merged `.vetq` bytes are identical to concatenated source bytes (minus deleted docs)
  - Seed mismatch: create index with explicit seed=1, AddIndexes from index with seed=2, verify merge succeeds via re-quantization
  - Merge with deleted docs: delete 50% of docs, merge, verify only live docs in result

#### Phase 2 Gate

**All of the following must pass before starting Phase 3:**
- [x] `TestTurboQuantFlatVectorsFormat` passes (write/read/score round-trip)
- [x] `TestTurboQuantHnswVectorsFormat extends BaseKnnVectorsFormatTestCase` passes (~50 inherited tests)
  - Override `randomVectorEncoding()` → FLOAT32
  - Override `getQuantizationBits()` → encoding bit-width
  - Override `supportsFloatVectorFallback()` → false
  - Override `assertOffHeapByteSize()` → check "vetq" key
  - Randomize encoding in `@Before`
- [x] All inherited tests pass: `testRandom`, `testRandomBytes`, `testSparseVectors`, `testDeleteAllVectorDocs`, `testSortedIndex`, `testCheckIndexIncludesVectors`, `testRecall`
- [x] `testRandomExceptions()` passes (no resource leaks)
- [x] `testCheckIntegrityReadsAllBytes()` passes
- [x] Merge tests pass (byte-copy, seed mismatch fallback, deleted docs)
- [x] Index + search works at d=4096 and d=768

---

### Phase 3: SIMD Scoring (2–3 weeks)

**Entry criteria:** Phase 2 gate passed. Naive scorer works correctly.

#### 3.1 SIMD dot product for b=4
- LUT-based: 16-entry centroid table fits in one AVX-512 register
- Unpack nibbles, gather centroids via `vpermps`, FMA with query
- Follow `VectorUtil` conventions (static methods, let JVM auto-vectorize)
- **Tests:**
  - SIMD result matches naive result within 1e-6 for 10K random vector pairs at d=4096
  - SIMD result matches naive result at d=768 (block-diagonal rotation)

#### 3.2 SIMD Euclidean distance for b=4
- Same LUT approach: `sum((q_rot[i] - centroids[idx[i]])²)`
- **Tests:**
  - Matches naive within 1e-6 for 10K pairs at d=4096

#### 3.3 SIMD paths for b=2, b=3, b=8
- b=2: 4 centroids, 4 per byte
- b=3: 8 centroids, 3-byte groups
- b=8: 256 centroids, direct byte lookup (no nibble unpacking)
- **Tests:**
  - Each encoding matches naive within 1e-6

#### 3.4 Replace naive scorer with SIMD scorer
- Swap implementation in `TurboQuantVectorsScorer`
- Verify all Phase 2 tests still pass (regression check)

#### 3.5 Performance benchmarks
- Latency per query at d=4096 b=4 vs scalar quant int4
- Latency per query at d=768 b=4 vs scalar quant int4
- QPS on synthetic 100K dataset at d=4096
- Memory bandwidth utilization analysis (2 KB per vector read at d=4096 b=4)

#### Phase 3 Gate

**All of the following must pass before starting Phase 4:**
- [x] All Phase 2 gate tests still pass with SIMD scorer (no regression)
- [x] SIMD vs naive agreement within 1e-6 for all encodings and similarity functions
- [ ] Performance improvement measured: SIMD scorer is ≥ 2x faster than naive at d=4096 *(JMH benchmark created in TurboQuantBenchmark.java — run with `gradlew :lucene:benchmark-jmh:jmh`)*
- [x] No new test failures in `BaseKnnVectorsFormatTestCase`

---

### Phase 4: Comprehensive Testing & Quality Validation (2–3 weeks)

**Entry criteria:** Phase 3 gate passed.

#### 4.1 Recall validation
- Test at d=4096 b=4: recall@10 ≥ 0.9 (efSearch=25, 10K vectors)
- Test at d=768 b=4: recall@10 ≥ 0.9
- Test at b=2: recall@10 ≥ 0.7
- Test at b=8: recall@10 ≥ 0.95
- Randomized dimension: `d = random().nextInt(32, 4097)`, b=4, recall@10 ≥ 0.8
- Compare recall vs scalar quant int4 at d=768 (document result, no hard gate)

#### 4.2 Scoring correctness (extended)
- For each `VectorSimilarityFunction` × each `TurboQuantEncoding`:
  - Quantized score vs exact score error within theoretical MSE bound
  - Score monotonicity: ≥ 95% agreement over 1000 random pairs
- Single vector per segment: score ≈ exact within 0.01

#### 4.3 Edge cases & stress
- Empty segment (zero vectors) — index, merge, search all succeed
- Single vector segment — search returns it
- 10K+ vectors at d=4096 (if CI allows) — index, merge, search
- Mixed fields: one TurboQuant + one scalar quant in same index — both searchable
- Index sorting with vector fields — vectors survive sort
- Concurrent indexing + searching — no crashes or corruption

#### 4.4 Merge stress
- 10 segments → force merge to 1 → all vectors searchable
- Merge with 50% deleted docs → only live docs in result
- AddIndexes from directory with different codec → succeeds
- AddIndexes with mismatched rotation seed → re-quantization fallback works

#### 4.5 CheckIndex
- Checksums valid on `.vetq` and `.vemtq`
- Vector count in metadata matches stored vectors
- Corrupted `.vetq` file detected by `checkIntegrity()`

#### 4.6 Performance benchmarks
- Recall comparison table: TurboQuant b=4 vs scalar quant int4 vs BBQ at d=768, d=4096
- Merge throughput: byte-copy TurboQuant vs re-quantization scalar quant (vectors/sec)
- Memory profiling: heap + off-heap at d=4096, 1M vectors
- JMH benchmark in `lucene/benchmark-jmh/`:
  - `TurboQuantQuantizeBenchmark` — vectors/sec at d=4096
  - `TurboQuantHadamardBenchmark` — rotations/sec at d=4096
  - `TurboQuantScoringBenchmark` — dot products/sec at d=4096 b=4

#### Phase 4 Gate

**All of the following must pass before starting Phase 5:**
- [x] Recall@10 ≥ 0.9 at d=4096 b=4
- [x] Recall@10 ≥ 0.9 at d=768 b=4
- [x] All edge case tests pass
- [x] All merge stress tests pass
- [x] CheckIndex validates TurboQuant segments correctly
- [ ] No test failures in full `ant test` run with randomized codec selection *(run: `gradlew test -Dtests.codec=random`)*
- [ ] Performance benchmarks documented with comparison to scalar quant *(run: `gradlew :lucene:benchmark-jmh:jmh -Pjmh.includes=TurboQuant`)*

---

### Phase 5: Documentation & Contribution (1 week)

**Entry criteria:** Phase 4 gate passed.

#### 5.1 Code documentation
- Javadoc on all public classes and methods
- `package-info.java` with:
  - Format description and algorithm summary
  - File format specification (byte-level layout of `.vetq` and `.vemtq`)
  - When to use TurboQuant vs scalar quant
  - Limitations (d ≥ 32, float32 only)

#### 5.2 Project documentation
- `CHANGES.txt` entry under "New Features"
- Benchmark results summary in commit message

#### 5.3 Contribution process
- JIRA issue with design rationale linking to this plan
- Lucene dev mailing list discussion post
- Patch/PR with all code, tests, and documentation

#### 5.4 Final verification
- [x] `ant precommit` passes (formatting, javadoc, forbidden APIs)
- [x] `ant test -Dtests.codec=TurboQuantHnsw` passes
- [x] No external dependencies (pure Java + precomputed constants)
- [x] All files have ASF license headers

---
