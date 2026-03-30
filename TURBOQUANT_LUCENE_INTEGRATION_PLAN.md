# TurboQuant Native Integration into Apache Lucene Vector Search

> Integration plan for [TurboQuant](https://arxiv.org/html/2504.19874v1) (Zandieh et al., ICLR 2026)
> into Apache Lucene as a new `FlatVectorsFormat` codec.
>
> Reference implementation: [scos-lab/turboquant](https://github.com/scos-lab/turboquant)
>
> Primary target: d=4096 embeddings. Also supports d=768, 1536, 3072, and any d ≥ 32.

---

## 1. What Is TurboQuant

TurboQuant is a data-oblivious online vector quantizer achieving near-optimal distortion rates
(within ~2.7x of information-theoretic lower bounds). Core properties relevant to Lucene:

- **No training/calibration** — unlike PQ or Lucene's scalar quantization (which estimates quantiles from data)
- **Online/streaming** — each vector quantized independently at index time
- **Configurable bit-width** — 2, 3, 4, or 8 bits per coordinate
- **Provably near-optimal** — exponential improvement over existing methods in bit-width dependence
- **Geometry-preserving** — rotation is orthogonal, so L2/dot-product/cosine computed in rotated space are exact
- **High-dimension friendly** — Gaussian approximation improves with d; ideal for d=4096

### Algorithm (MSE-optimal, used for NN search)

1. Store original norm `||x||` as float32
2. Normalize: `x̂ = x / ||x||`
3. Random rotation: `y = Π · x̂` (shared globally via deterministic seed)
4. Scalar quantize each coordinate of `y` using precomputed Beta-distribution-optimal Lloyd-Max centroids → `b`-bit index per coordinate
5. Dequantize: look up centroids, inverse-rotate back

After rotation, each coordinate follows Beta((d-1)/2, (d-1)/2) on [-1,1], converging to N(0, 1/d) for d ≥ 64. Coordinates become nearly independent, so per-coordinate scalar quantization is near-optimal.

### Why MSE-only (not TurboQuant_Prod)

The paper also proposes an inner-product-optimal variant that adds a 1-bit QJL residual correction for unbiased inner product estimation. The reference implementation's own benchmarks show **MSE-only is better for NN search**: the QJL residual adds variance that hurts recall more than the small bias it removes. We implement MSE-only.

### Theoretical Distortion (unit vectors)

| Bit-width | MSE distortion | Lower bound | Ratio |
|-----------|---------------|-------------|-------|
| 2         | 0.117         | 0.063       | 1.87x |
| 3         | 0.030         | 0.016       | 1.92x |
| 4         | 0.009         | 0.004       | 2.30x |
| 8         | ~0.00002      | ~0.00002    | ~1.0x |

---

## 2. Decisions

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Abstraction layer | `FlatVectorsFormat` (not `KnnVectorsFormat`) | Follows Lucene104 pattern: flat format handles storage/scoring, HNSW wraps it |
| 2 | Bit-width config | Enum `TurboQuantEncoding` with values BITS_2, BITS_3, BITS_4, BITS_8 | Default BITS_4 (8x compression). Prevents invalid values, self-documenting |
| 3 | Rotation strategy | Hadamard-only, global deterministic seed | d=4096 is 2^12 — perfect Hadamard. Global seed eliminates merge re-quantization |
| 4 | Mixed-precision | Not implemented | Rotation homogenizes coordinate distributions. Per-field bit-width via `PerFieldKnnVectorsFormat` covers the useful case |
| 5 | Max dimensions | 16384 | TurboQuant improves with higher d. Primary target d=4096 |
| 6 | Off-heap storage | Mandatory (mmap'd IndexInput) | At d=4096, b=4: 2 KB/vector. Must be off-heap for million-scale indices |
| 7 | Merge re-quantization | Avoided via global rotation seed | Rotation derived from field name → all segments share rotation → merge = byte copy |

---

## 3. Architecture

### 3.1 Module Structure

```
lucene/codecs/src/java/org/apache/lucene/codecs/turboquant/
├── TurboQuantFlatVectorsFormat.java       — FlatVectorsFormat SPI entry point
├── TurboQuantFlatVectorsWriter.java       — index-time: rotate + quantize + write
├── TurboQuantFlatVectorsReader.java       — search-time: off-heap read + scoring
├── TurboQuantVectorsScorer.java           — FlatVectorsScorer impl (hot path)
├── TurboQuantHnswVectorsFormat.java       — convenience: HNSW + TurboQuant composed
├── OffHeapTurboQuantVectorValues.java     — off-heap mmap'd quantized vector access
├── HadamardRotation.java                  — fast Walsh-Hadamard transform + sign diagonal
├── BetaCodebook.java                      — precomputed Lloyd-Max centroids per bit-width
├── TurboQuantEncoding.java                — enum: BITS_2, BITS_3, BITS_4, BITS_8
├── TurboQuantBitPacker.java               — bit-packing for b=2,3,4,8
└── package-info.java

lucene/codecs/src/test/org/apache/lucene/codecs/turboquant/
├── TestTurboQuantFlatVectorsFormat.java
├── TestTurboQuantHnswVectorsFormat.java
├── TestHadamardRotation.java
├── TestBetaCodebook.java
└── TestTurboQuantBitPacker.java

lucene/codecs/src/resources/META-INF/services/
└── org.apache.lucene.codecs.KnnVectorsFormat  (append TurboQuantHnswVectorsFormat)
```

### 3.2 Class Hierarchy (follows Lucene104 pattern exactly)

```
KnnVectorsFormat
├── FlatVectorsFormat
│   ├── Lucene99FlatVectorsFormat          (raw float32 storage — reused as delegate)
│   ├── Lucene104ScalarQuantizedVectorsFormat  (int8/int4 scalar quant)
│   └── TurboQuantFlatVectorsFormat  ← NEW (rotation-based quantization)
│       └── holds FlatVectorsWriter rawVectorDelegate (Lucene99FlatVectorsFormat)
│
└── TurboQuantHnswVectorsFormat  ← NEW (convenience: HNSW + TurboQuant)
    └── fieldsWriter() returns Lucene99HnswVectorsWriter(state, maxConn, beamWidth,
            turboQuantFlatFormat.fieldsWriter(state), numMergeWorkers, mergeExec, threshold)

FlatVectorsScorer
├── (existing Lucene99 scorer)
├── Lucene104ScalarQuantizedVectorScorer
└── TurboQuantVectorsScorer  ← NEW (LUT-based quantized distance in rotated space)
```

**Key reuse points:**
- `Lucene99FlatVectorsFormat` — raw vector storage (delegate, not reimplemented)
- `Lucene99HnswVectorsWriter` — HNSW graph construction (takes our FlatVectorsWriter)
- `Lucene99HnswVectorsReader` — HNSW graph search (takes our FlatVectorsReader)
- `CodecUtil` — index headers, footers, checksums on all files
- `FlatVectorsScorer` interface — scoring contract for HNSW traversal
- `FlatFieldVectorsWriter<T>` — per-field writer contract with `getVectors()`, `getDocsWithFieldSet()`
- `CloseableRandomVectorScorerSupplier` — merge scorer contract
- `VectorUtil` patterns — SIMD scoring follows existing conventions
- `BaseKnnVectorsFormatTestCase` — test infrastructure (dozens of tests for free)

**Not reused (intentionally):**
- `ScalarEncoding` — tightly coupled to `OptimizedScalarQuantizer` corrective terms (centroid, component sums). TurboQuant's quantization is fundamentally different (rotation-based, no centroid). Own `TurboQuantEncoding` enum, but follows same packing math patterns.
- `OptimizedScalarQuantizer` — data-dependent quantile estimation. TurboQuant is data-oblivious.
- `QuantizedByteVectorValues` — assumes corrective terms, centroid, quantizer. TurboQuant needs its own `OffHeapTurboQuantVectorValues`.

**Test compatibility — `hasQuantized()` detection:**
The base test's `hasQuantized()` checks `knnVectorsReader instanceof QuantizedVectorsReader` first, then falls back to class name heuristic. `TurboQuantFlatVectorsReader` should implement `QuantizedVectorsReader` so the test correctly identifies it as quantized. The `getQuantizedVectorValues()` method returns our `OffHeapTurboQuantVectorValues` (which extends `BaseQuantizedByteVectorValues`). The off-heap map uses `"vetq"` as the key; the test's `assertOffHeapByteSize()` is overridden to check for this key.

### 3.3 File Format (per segment)

| File | Extension | Off-heap map key | Contents | Size (d=4096, b=4, n docs) |
|------|-----------|-----------------|---------|---------------------------|
| Quantized vectors | `.vetq` | `"vetq"` | Packed b-bit indices + float32 norms, contiguous per-doc, off-heap | n × (2048 + 4) bytes |
| Metadata | `.vemtq` | — (not mmap'd) | CodecUtil header, dimension, encoding, vector count, rotation seed, similarity, version, CodecUtil footer | ~128 bytes |
| Raw vectors | `.vec` | `"vec"` | Delegated to `Lucene99FlatVectorsFormat` | n × 16384 bytes |
| Raw metadata | `.vemf` | — | Delegated to `Lucene99FlatVectorsFormat` | varies |
| HNSW graph | `.vex` | `"vex"` | Delegated to `Lucene99HnswVectorsReader` | varies |
| HNSW metadata | `.vem` | — | Delegated to `Lucene99HnswVectorsReader` | varies |

**Extension strategy:** TurboQuant uses unique extensions (`.vetq`, `.vemtq`) following the Lucene convention that different format types use different extensions. Raw vectors (`.vec`) and HNSW graph (`.vex`) are delegated to existing formats and use their standard extensions.

The convention in Lucene:
- Raw float vectors: `.vec` (Lucene99FlatVectorsFormat)
- Scalar quantized: `.veq` (Lucene99/Lucene104 ScalarQuantized)
- Binary quantized: `.veb` (Lucene102 BinaryQuantized)
- **TurboQuant: `.vetq`** (new, unique)

Extensions are reused across *versions* of the same format family (Lucene99 and Lucene104 both use `.veq`), but different format types always use different extensions.

```java
static final String META_CODEC_NAME = "TurboQuantVectorsFormatMeta";
static final String VECTOR_DATA_CODEC_NAME = "TurboQuantVectorsFormatData";
static final String META_EXTENSION = "vemtq";
static final String VECTOR_DATA_EXTENSION = "vetq";
static final int VERSION_START = 0;
static final int VERSION_CURRENT = VERSION_START;
```

**Storage at d=4096, b=4, 1M vectors:**

| Component | Size | Notes |
|-----------|------|-------|
| Quantized vectors (.vetq) | 1.95 GB | Off-heap, mmap'd |
| Norms (in .vetq) | 3.8 MB | Stored alongside quantized data |
| Raw vectors (.vec) | 15.6 GB | Off-heap, for merge + rescore |
| Float32 baseline | 15.6 GB | — |
| **Compression ratio** | **8x** | Quantized only; raw kept for rescore |

### 3.4 Hadamard Rotation

The rotation `Π` is constructed differently depending on whether d is a power of 2.

#### Case 1: d is a power of 2 (e.g., d=4096, 2048, 1024, 512, 256, 128)

```
Π = H_d · D
```

Where:
- `H_d` = Walsh-Hadamard matrix (implicit, never materialized)
- `D` = diagonal matrix of random ±1 signs (d bits storage)

d=4096 = 2^12 — perfect fit. O(d log d) = 49,152 FLOPs.

#### Case 2: d is NOT a power of 2 (e.g., d=768, 1536, 3072)

Use **block-diagonal Hadamard with pre-permutation:**

```
Π = BlockHadamard(b₁, b₂, ..., bₖ) · Permutation · SignFlip
```

Where:
- `Permutation` = random coordinate permutation (breaks any cross-block structure)
- `SignFlip` = random ±1 per coordinate (d bits)
- `BlockHadamard` = independent Hadamard transforms on power-of-2 blocks that sum to d

**Block decomposition for common dimensions:**

| Dimension | Decomposition | Max block | log₂(max block) | Overhead |
|-----------|--------------|-----------|-----------------|----------|
| 768       | 512 + 256    | 512       | 9               | 0%       |
| 1536      | 1024 + 512   | 1024      | 10              | 0%       |
| 3072      | 2048 + 1024  | 2048      | 11              | 0%       |
| 4096      | 4096         | 4096      | 12              | 0%       |
| 384       | 256 + 128    | 256       | 8               | 0%       |
| 1024      | 1024         | 1024      | 10              | 0%       |

The decomposition greedily assigns the largest power-of-2 block that fits, then recurses on the remainder. Any positive integer d can be decomposed this way (it's just the binary representation of d).

**Cost:** O(d · log₂(max_block_size)). For d=768 with blocks (512, 256): 768 × 9 = 6,912 FLOPs. Slightly less than a single 1024-Hadamard would be.

**Statistical quality:** The pre-permutation ensures coordinates are randomly assigned to blocks, so the block-diagonal structure doesn't create systematic correlation patterns. Each block independently produces sub-Gaussian coordinates. For d ≥ 32 with blocks ≥ 32, the quantization quality is indistinguishable from a full random rotation.

**No padding, no wasted storage.** Every quantized coordinate corresponds to a real input dimension.

#### Implementation: `HadamardRotation.java`

```java
public final class HadamardRotation {
    private final int d;
    private final int[] blockSizes;      // power-of-2 block sizes summing to d
    private final int[] permutation;      // random coordinate permutation
    private final byte[] signs;           // random ±1 per coordinate (d bits packed)

    public static HadamardRotation create(int d, long seed);

    /** Apply rotation: O(d · log(maxBlock)) */
    public void rotate(float[] x, float[] out);

    /** Apply inverse rotation: O(d · log(maxBlock)) */
    public void inverseRotate(float[] y, float[] out);

    /** Decompose d into power-of-2 blocks (binary representation) */
    static int[] decomposeBlocks(int d);
}
```

#### Global rotation seed

The rotation is derived deterministically from the field name (e.g., `seed = MurmurHash3(fieldName)`). All segments for the same field share the same rotation. Consequences:
- **Merge never re-quantizes** — quantized bytes are copied directly
- **No per-segment rotation storage** — seed is implicit from field name
- **Computed once per field, cached** — no per-segment-open cost

### 3.5 Precomputed Codebooks (`BetaCodebook`)

For d ≥ 64, the Beta distribution is well-approximated by N(0, 1/d). This means:

- Centroids for a given bit-width b are the same (up to scaling by 1/√d) regardless of d
- We precompute one set of "canonical" Gaussian centroids per bit-width at class-load time
- At runtime: `centroid_actual[i] = canonical_centroid[i] / √d`

```java
public final class BetaCodebook {
    // Canonical centroids for N(0,1), scaled by 1/√d at runtime
    private static final float[][] GAUSSIAN_CENTROIDS = {
        /* b=2 */ { -1.5104f, -0.4528f, 0.4528f, 1.5104f },
        /* b=3 */ { /* 8 centroids */ },
        /* b=4 */ { /* 16 centroids */ },
        /* b=8 */ { /* 256 centroids */ },
    };

    public static float[] centroids(int d, int b);   // returns 2^b values
    public static float[] boundaries(int d, int b);   // returns 2^b + 1 values
}
```

---

## 4. Index-Time Flow

### 4.1 `TurboQuantFlatVectorsWriter` (extends `FlatVectorsWriter`)

Follows the same lifecycle as `Lucene104ScalarQuantizedVectorsWriter`:

```java
public class TurboQuantFlatVectorsWriter extends FlatVectorsWriter {
    private final FlatVectorsWriter rawVectorDelegate;  // Lucene99FlatVectorsFormat writer
    private final TurboQuantEncoding encoding;
    private final HadamardRotation rotation;            // cached, shared across fields
    private final float[] centroids;                    // precomputed for this encoding + dim
    private IndexOutput meta, quantizedVectorData;      // .vemtq, .vetq files
}
```

**Lifecycle (mirrors Lucene104ScalarQuantizedVectorsWriter):**

```
Constructor(state, encoding, rawVectorDelegate, scorer):
  1. Store rawVectorDelegate
  2. Open .vemtq and .vetq with CodecUtil.writeIndexHeader
  3. Cache rotation from global seed

addField(fieldInfo) → returns FlatFieldVectorsWriter:
  1. Call rawVectorDelegate.addField(fieldInfo) → get raw field writer
  2. Create TurboQuantFieldWriter wrapping the raw field writer
  3. TurboQuantFieldWriter.addValue(docID, vector):
     a. Delegate to rawFieldWriter.addValue(docID, vector)
     b. Compute norm, rotate, quantize, buffer quantized bytes

flush(maxDoc, sortMap):
  1. Call rawVectorDelegate.flush(maxDoc, sortMap)
  2. For each field with float32 vectors:
     Iterate buffered raw vectors (from delegate), rotate + quantize each,
     write quantized bytes + norm to .vetq (streaming, no heap buffering of quantized data)
     Write metadata to .vemtq

mergeOneField(fieldInfo, mergeState):
  1. Call rawVectorDelegate.mergeOneField(fieldInfo, mergeState)

mergeOneFieldToIndex(fieldInfo, mergeState) → CloseableRandomVectorScorerSupplier:
  1. Call rawVectorDelegate.mergeOneField(fieldInfo, mergeState)
  2. Verify source segments' rotation seeds match target (from .vemtq metadata)
  3. If seeds match: copy quantized bytes directly from source segments
  4. If seeds differ (e.g., AddIndexes from different index): re-quantize from raw vectors
  5. Write merged quantized data to .vetq
  6. Return CloseableRandomVectorScorerSupplier over merged quantized data
     (Lucene99HnswVectorsWriter uses this to rebuild the HNSW graph)

finish():
  1. Call rawVectorDelegate.finish()
  2. CodecUtil.writeFooter on .vemtq and .vetq
```

### 4.2 Segment Merge

**With global rotation seed: merge is a byte copy.** All segments for the same field share the same rotation, so quantized vectors are directly compatible:

1. Copy quantized bytes from source segments to merged segment (no re-quantization)
2. Copy norms from source segments
3. Delegate raw vector merge to `rawVectorDelegate.mergeOneField()`
4. Return `CloseableRandomVectorScorerSupplier` so HNSW graph can be rebuilt

This is a significant advantage over scalar quantization, which must re-quantize when quantiles shift.

---

## 5. Search-Time Flow

### 5.1 `TurboQuantFlatVectorsReader` (extends `FlatVectorsReader`)

Follows the same pattern as `Lucene104ScalarQuantizedVectorsReader`:

```java
public class TurboQuantFlatVectorsReader extends FlatVectorsReader
        implements QuantizedVectorsReader {
    private final FlatVectorsReader rawVectorsReader;   // Lucene99FlatVectorsReader delegate
    private final IndexInput quantizedVectorData;       // mmap'd .vetq
    private final Map<String, FieldEntry> fields;       // per-field metadata from .vemtq
    private final HadamardRotation rotation;            // cached from global seed
}
```

**Delegation contracts:**
- `getFloatVectorValues(field)` → delegates to `rawVectorsReader.getFloatVectorValues(field)` (for rescore, scripts)
- `getByteVectorValues(field)` → throws `UnsupportedOperationException` (float32 input only)
- `getRandomVectorScorer(field, target)` → returns scorer over quantized data (hot path)
- `getQuantizedVectorValues(field)` → returns `OffHeapTurboQuantVectorValues` (satisfies `QuantizedVectorsReader` interface, enables `hasQuantized()` detection in tests)
- `ramBytesUsed()` → shallow size + field map + rotation cache + `rawVectorsReader.ramBytesUsed()`
- `getOffHeapByteSize(fieldInfo)` → merge raw reader's map + `Map.of("vetq", quantizedDataLength)` (unique extension key)
- `checkIntegrity()` → `CodecUtil.checksumEntireFile` on .vetq, .vemtq + delegate to raw reader
- `getMergeInstance()` → return optimized merge reader (single-thread safe)

### 5.2 `TurboQuantVectorsScorer` (implements `FlatVectorsScorer`)

This is the hot path. The scorer provides `RandomVectorScorer` instances to the HNSW graph traversal.

```java
public class TurboQuantVectorsScorer implements FlatVectorsScorer {

    @Override
    public RandomVectorScorer getRandomVectorScorer(
            VectorSimilarityFunction sim,
            KnnVectorValues vectorValues,
            float[] target) {
        // 1. Rotate query once: q_rot = hadamardRotate(normalize(target), signs)
        // 2. Return scorer that computes distance in rotated space
        //    against off-heap quantized vectors
    }
}
```

### 5.3 Per-Candidate Scoring

```
For each candidate doc (from HNSW graph):
  1. Read b-bit indices from off-heap .vetq (mmap'd IndexInput)
  2. Compute distance in rotated space via LUT gather:
     - DOT_PRODUCT: sum(q_rot[i] * centroids[idx[i]]) * doc_norm
     - EUCLIDEAN:   sum((q_rot[i] - centroids[idx[i]])²)
     - COSINE:      sum(q_rot[i] * centroids[idx[i]])  (both unit-normalized)
  3. No inverse rotation needed (orthogonal rotation preserves all distances)
```

### 5.4 SIMD-Optimized Scoring

For b=4 at d=4096: each vector is 2048 bytes (nibble-packed). The inner loop:

```
Per candidate (dot product):
  For each byte in packed indices (2048 bytes, 2 indices per byte):
    1. Unpack high/low nibble → 2 centroid indices
    2. Gather: c0 = centroids[lo], c1 = centroids[hi]
    3. FMA: sum += q_rot[2i] * c0 + q_rot[2i+1] * c1

With AVX-512 (512-bit = 64 bytes per iteration):
  - Process 128 dimensions per iteration (64 packed bytes)
  - 32 iterations for d=4096
  - vpermps for 16-entry centroid LUT gather (16 × 32-bit = 512 bits = 1 register)

With ARM NEON (128-bit):
  - Process 32 dimensions per iteration
  - 128 iterations for d=4096
  - tbl for byte-level LUT gather
```

### 5.5 Off-Heap Vector Access (`OffHeapTurboQuantVectorValues`)

```java
public class OffHeapTurboQuantVectorValues extends BaseQuantizedByteVectorValues {
    private final IndexInput quantizedData;  // mmap'd .vetq
    private final int bytesPerVector;        // d * b / 8
    private final float[] centroids;
    private final float invSqrtD;

    // Random access by ordinal — seek into mmap'd file
    public byte[] getQuantizedVector(int ord) {
        quantizedData.seek((long) ord * bytesPerVector);
        quantizedData.readBytes(buffer, 0, bytesPerVector);
        return buffer;
    }
}
```

### 5.6 Similarity Function Support

| Similarity | Computation | Notes |
|-----------|-------------|-------|
| `EUCLIDEAN` | `||q_rot - ŷ||²` | Rotation preserves L2 |
| `DOT_PRODUCT` | `q_rot · ŷ · doc_norm` | Rotation preserves dot product |
| `COSINE` | `q_rot · ŷ` | Both unit-normalized before rotation |
| `MAXIMUM_INNER_PRODUCT` | `q_rot · ŷ · doc_norm` | Same as dot product |

---

## 6. Public API

### 6.1 Encoding Enum

```java
public enum TurboQuantEncoding {
    BITS_2(2),   // 16x compression, aggressive
    BITS_3(3),   // ~10.7x compression
    BITS_4(4),   // 8x compression, default, best recall/compression trade-off
    BITS_8(8);   // 4x compression, near-lossless

    public final int bitsPerCoordinate;
}
```

### 6.2 Format Construction

```java
// Flat format only (for composition with any graph format)
new TurboQuantFlatVectorsFormat()                          // default: BITS_4
new TurboQuantFlatVectorsFormat(TurboQuantEncoding.BITS_2) // aggressive

// Convenience: HNSW + TurboQuant
new TurboQuantHnswVectorsFormat()                          // defaults for both
new TurboQuantHnswVectorsFormat(
    TurboQuantEncoding.BITS_4,  // quantization
    16,                          // maxConn
    100                          // beamWidth
)

// Full control with merge parallelism and explicit rotation seed
new TurboQuantHnswVectorsFormat(
    TurboQuantEncoding.BITS_4,
    16, 100,                     // maxConn, beamWidth
    4, mergeExecutor,            // numMergeWorkers, executor
    42L                          // rotationSeed (null = derive from field name)
)
```

### 6.3 Per-Field Selection

```java
public class MyCodec extends FilterCodec {
    public MyCodec() { super("MyCodec", new Lucene104Codec()); }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return new PerFieldKnnVectorsFormat() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return switch (field) {
                    case "embedding_4k" -> new TurboQuantHnswVectorsFormat(
                                               TurboQuantEncoding.BITS_4, 16, 100);
                    case "embedding_small" -> new TurboQuantHnswVectorsFormat(
                                               TurboQuantEncoding.BITS_2, 16, 100);
                    default -> new Lucene104HnswScalarQuantizedVectorsFormat();
                };
            }
        };
    }
}
```

### 6.4 Defaults

| Parameter | Default | Range | Rationale |
|-----------|---------|-------|-----------|
| `encoding` | `BITS_4` | BITS_2/3/4/8 | 8x compression, MSE ≈ 0.009 |
| `maxDimensions` | 16384 | — | TurboQuant excels at high d |
| `rotation` | Hadamard (global seed) | — | O(d log d), zero per-segment storage, merge = byte copy |
| `maxConn` | 16 | 1–512 | Same as Lucene99Hnsw default |
| `beamWidth` | 100 | 1–3200 | Same as Lucene99Hnsw default |

---

## 7. Comparison with Existing Lucene Quantization

| Property | Scalar Quant (int8) | Scalar Quant (int4) | BBQ (1-bit) | TurboQuant (b=4) |
|----------|-------------------|-------------------|-------------|-----------------|
| Bits/coord | 8 | 4 | 1 | 4 |
| Compression vs f32 | 4x | 8x | 32x | 8x |
| Calibration | Per-segment quantile estimation | Per-segment + grid search | Per-segment | **None** (data-oblivious) |
| Merge behavior | Re-quantize if quantiles shift | Re-quantize if quantiles shift | Re-quantize | **Byte copy** (global rotation) |
| Theoretical guarantee | None | None | None | **≤ 2.7× optimal** |
| Error correction | Per-vector float | Per-vector float + optimized | Hamming-based | Not needed (rotation + optimal codebook) |
| Query overhead | None | None | None | One Hadamard transform per query per field |
| Max dimensions | 1024 | 1024 | 1024 | **16384** |
| Streaming-friendly | No (needs quantile warmup) | No (needs optimization pass) | No | **Yes** (each vector independent) |
| Best for | General ≤1024d | Memory-constrained ≤1024d | Extreme compression | **High-dim (4096), streaming, shifting distributions** |

**When to choose TurboQuant:**
- d=4096 or other high-dimensional embeddings (exceeds 1024-dim limit of existing formats)
- Data distribution shifts over time (no recalibration needed)
- Streaming/online indexing where you can't sample data upfront
- Merge-heavy workloads (byte-copy merge vs re-quantization)
- You want provable quality guarantees

**When scalar quantization is better:**
- Data has exploitable per-dimension structure (clustered, skewed)
- Very low dimensions (d < 32)
- You need the error correction float for maximum recall at d ≤ 1024

---

## 8. Implementation Phases

Each phase has explicit entry criteria, deliverables, and gate tests that must pass before proceeding.

→ **See [TURBOQUANT_IMPLEMENTATION_PLAN.md](./TURBOQUANT_IMPLEMENTATION_PLAN.md)** for the full phased plan.

**Summary:**

| Phase | Duration | Key Deliverable | Gate |
|-------|----------|----------------|------|
| 1. Core Algorithm | 2–3 weeks | `HadamardRotation`, `BetaCodebook`, `TurboQuantBitPacker` | MSE matches paper, round-trip < 1e-5 |
| 2. Codec Integration | 3–4 weeks | Full writer/reader/scorer/format, naive scorer | ~50 `BaseKnnVectorsFormatTestCase` tests pass |
| 3. SIMD Scoring | 2–3 weeks | LUT-based SIMD scorer replaces naive | No regression, SIMD matches naive < 1e-6, ≥2x speedup |
| 4. Quality Validation | 2–3 weeks | Recall, edge cases, merge stress, benchmarks | Recall@10 ≥ 0.9 at d=4096 b=4, all stress tests pass |
| 5. Documentation | 1 week | Javadoc, package-info, CHANGES.txt, JIRA | `ant precommit` passes, ASF headers |


## 9. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Block-diagonal Hadamard quality for small blocks | If d has small power-of-2 factors (e.g., d=33 = 32+1), the 1-dim block is degenerate | Very Low | Minimum supported d=32. For d with tiny remainder blocks (< 8), fall back to padding that block. In practice, all common embedding dims decompose into blocks ≥ 128 |
| Recall regression vs optimized scalar quant at d≤1024 | Users see worse recall | Medium | TurboQuant's sweet spot is d≥256. For d≤1024, scalar quant with error correction may win on recall. Document clearly, provide benchmarks |
| Query rotation overhead | Latency increase | Low | Hadamard at d=4096: 49K FLOPs. Block-Hadamard at d=768: 7K FLOPs. HNSW traversal: ~100K–400K FLOPs. Overhead ≤10% |
| Off-heap memory pressure at scale | OS page cache contention | Low | Same as all mmap'd Lucene formats. Quantized data is 8x smaller than raw, so actually reduces pressure |
| Global rotation seed collision | Two fields with same hash get same rotation | Very Low | Use MurmurHash3 of field name. Even if collision occurs, correctness is unaffected — only statistical optimality |

---

## 10. Future Extensions (Out of Scope for Initial Implementation)

- **Entropy coding of indices:** Paper notes 5% bit-width reduction for b=4 via Huffman. Low ROI initially
- **TurboQuant_Prod mode:** For use cases requiring unbiased inner product estimation
- **Adaptive bit-width:** Auto-select b based on target recall or memory budget
- **Integration with Elasticsearch:** Expose as index setting (`index.codec.vectors: turboquant`)
- **GPU-accelerated rotation:** For bulk indexing pipelines. Hadamard maps naturally to GPU
- **Quantized-only mode (no raw vectors):** For maximum compression when rescore isn't needed
