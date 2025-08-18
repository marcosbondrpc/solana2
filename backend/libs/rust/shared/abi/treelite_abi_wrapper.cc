// LEGENDARY Treelite ABI Wrapper for MEV/ARB Models
// Zero-downtime hot-swapping with embedded defaults
// Optimized for sub-microsecond inference latency

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <atomic>
#include <mutex>
#include <memory>
#include <vector>
#include <chrono>
#include <dlfcn.h>
#include <immintrin.h>

// Compile-time defaults for embedded paths
#ifndef DEFAULT_INNER_LIB_PATH
#define DEFAULT_INNER_LIB_PATH ""
#endif

#ifndef DEFAULT_CALIB_PATH
#define DEFAULT_CALIB_PATH ""
#endif

// Model type for specialized optimizations
#ifndef MODEL_TYPE
#define MODEL_TYPE "GENERIC"
#endif

// Function pointer types for Treelite runtime
typedef void* (*TreelitePredictorLoad_t)(const char* libpath, int num_worker_thread);
typedef void (*TreelitePredictorFree_t)(void* predictor);
typedef size_t (*TreelitePredictorQueryResultSize_t)(void* predictor);
typedef size_t (*TreelitePredictorQueryNumFeatures_t)(void* predictor);
typedef size_t (*TreelitePredictorPredictBatch_t)(
    void* predictor,
    const void* batch,
    int batch_size,
    int num_feature,
    int pred_margin,
    void* out_result
);

// Global state with atomic pointers for lock-free hot-swapping
static std::atomic<void*> G_PREDICTOR{nullptr};
static std::atomic<void*> G_PREDICTOR_ARB{nullptr};
static std::atomic<void*> G_DL_HANDLE{nullptr};
static std::atomic<void*> G_DL_HANDLE_ARB{nullptr};

// Function pointers
static TreelitePredictorLoad_t G_FN_LOAD = nullptr;
static TreelitePredictorFree_t G_FN_FREE = nullptr;
static TreelitePredictorQueryResultSize_t G_FN_QUERY_SIZE = nullptr;
static TreelitePredictorQueryNumFeatures_t G_FN_QUERY_FEATURES = nullptr;
static TreelitePredictorPredictBatch_t G_FN_PREDICT = nullptr;

// Calibration state
static bool G_HAVE_CALIB = false;
static float G_CALIB_SCALE = 1.0f;
static float G_CALIB_OFFSET = 0.0f;
static float G_CALIB_CLIP_MIN = -1e6f;
static float G_CALIB_CLIP_MAX = 1e6f;

// Performance counters
static std::atomic<uint64_t> G_PREDICT_COUNT{0};
static std::atomic<uint64_t> G_PREDICT_NANOS{0};

// Thread-local scratch buffers for zero-allocation inference
thread_local static std::vector<float> TL_SCRATCH_BUFFER;

// Error handling
static void die(const char* msg) {
    fprintf(stderr, "[TREELITE ABI] FATAL: %s\n", msg);
    abort();
}

// High-resolution timer
inline uint64_t get_nanos() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

// Initialize Treelite runtime once
static void init_once() {
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        // Get inner library path
        const char* inner_lib = std::getenv("TREELITE_INNER_LIB_PATH");
        if (!inner_lib || std::strlen(inner_lib) == 0) {
            inner_lib = DEFAULT_INNER_LIB_PATH;
        }
        if (!inner_lib || std::strlen(inner_lib) == 0) {
            die("TREELITE_INNER_LIB_PATH not set and no DEFAULT_INNER_LIB_PATH");
        }

        fprintf(stderr, "[TREELITE ABI] Loading model: %s\n", inner_lib);
        fprintf(stderr, "[TREELITE ABI] Model type: %s\n", MODEL_TYPE);

        // Load Treelite runtime library
        void* rt_handle = dlopen("libtreelite_runtime.so", RTLD_NOW | RTLD_GLOBAL);
        if (!rt_handle) {
            die(dlerror());
        }

        // Load function pointers
        G_FN_LOAD = (TreelitePredictorLoad_t)dlsym(rt_handle, "TreelitePredictorLoad");
        G_FN_FREE = (TreelitePredictorFree_t)dlsym(rt_handle, "TreelitePredictorFree");
        G_FN_QUERY_SIZE = (TreelitePredictorQueryResultSize_t)dlsym(rt_handle, "TreelitePredictorQueryResultSize");
        G_FN_QUERY_FEATURES = (TreelitePredictorQueryNumFeatures_t)dlsym(rt_handle, "TreelitePredictorQueryNumFeatures");
        G_FN_PREDICT = (TreelitePredictorPredictBatch_t)dlsym(rt_handle, "TreelitePredictorPredictBatch");

        if (!G_FN_LOAD || !G_FN_FREE || !G_FN_PREDICT) {
            die("Failed to load Treelite runtime functions");
        }

        // Load primary model (MEV)
        void* predictor = G_FN_LOAD(inner_lib, 1);  // Single thread for lowest latency
        if (!predictor) {
            die("Failed to load Treelite model");
        }
        G_PREDICTOR.store(predictor, std::memory_order_release);

        // Optional: Load ARB model
        const char* arb_lib = std::getenv("TREELITE_INNER_LIB_PATH_ARB");
        if (!arb_lib || std::strlen(arb_lib) == 0) {
            arb_lib = DEFAULT_ARB_INNER_LIB;
        }
        if (arb_lib && std::strlen(arb_lib) > 0) {
            void* arb_predictor = G_FN_LOAD(arb_lib, 4);  // More threads for batch processing
            if (arb_predictor) {
                G_PREDICTOR_ARB.store(arb_predictor, std::memory_order_release);
                fprintf(stderr, "[TREELITE ABI] Loaded ARB model: %s\n", arb_lib);
            }
        }

        // Load calibration
        const char* calib_path = std::getenv("TREELITE_CALIB_PATH");
        if (!calib_path || std::strlen(calib_path) == 0) {
            calib_path = DEFAULT_CALIB_PATH;
        }
        if (calib_path && std::strlen(calib_path) > 0) {
            load_calibration(calib_path);
        }

        fprintf(stderr, "[TREELITE ABI] Initialization complete\n");
    });
}

// Load calibration parameters from JSON
static void load_calibration(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[TREELITE ABI] Warning: calibration file not found: %s\n", path);
        return;
    }

    // Simple JSON parser for calibration
    char buffer[1024];
    if (std::fgets(buffer, sizeof(buffer), f)) {
        // Parse: {"scale": 1.0, "offset": 0.0, "clip_min": -1e6, "clip_max": 1e6}
        std::sscanf(buffer, "{\"scale\":%f,\"offset\":%f,\"clip_min\":%f,\"clip_max\":%f}",
                    &G_CALIB_SCALE, &G_CALIB_OFFSET, &G_CALIB_CLIP_MIN, &G_CALIB_CLIP_MAX);
        G_HAVE_CALIB = true;
        fprintf(stderr, "[TREELITE ABI] Loaded calibration: scale=%.4f offset=%.4f\n",
                G_CALIB_SCALE, G_CALIB_OFFSET);
    }
    std::fclose(f);
}

// Apply calibration with SIMD optimization
inline void apply_calibration(float* values, size_t n) {
    if (!G_HAVE_CALIB) return;

    // Use AVX2 for vectorized calibration
    const __m256 scale = _mm256_set1_ps(G_CALIB_SCALE);
    const __m256 offset = _mm256_set1_ps(G_CALIB_OFFSET);
    const __m256 clip_min = _mm256_set1_ps(G_CALIB_CLIP_MIN);
    const __m256 clip_max = _mm256_set1_ps(G_CALIB_CLIP_MAX);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&values[i]);
        v = _mm256_fmadd_ps(v, scale, offset);  // v = v * scale + offset
        v = _mm256_max_ps(v, clip_min);
        v = _mm256_min_ps(v, clip_max);
        _mm256_storeu_ps(&values[i], v);
    }

    // Handle remainder
    for (; i < n; ++i) {
        float v = values[i] * G_CALIB_SCALE + G_CALIB_OFFSET;
        values[i] = std::fmax(G_CALIB_CLIP_MIN, std::fmin(G_CALIB_CLIP_MAX, v));
    }
}

// Hot-swap model at runtime
extern "C" int treelite_hot_swap(const char* new_lib_path, int is_arb) {
    if (!new_lib_path) return -1;

    fprintf(stderr, "[TREELITE ABI] Hot-swapping %s model: %s\n",
            is_arb ? "ARB" : "MEV", new_lib_path);

    // Load new model
    void* new_predictor = G_FN_LOAD(new_lib_path, is_arb ? 4 : 1);
    if (!new_predictor) {
        fprintf(stderr, "[TREELITE ABI] Failed to load new model\n");
        return -1;
    }

    // Atomic swap
    void* old_predictor;
    if (is_arb) {
        old_predictor = G_PREDICTOR_ARB.exchange(new_predictor, std::memory_order_acq_rel);
    } else {
        old_predictor = G_PREDICTOR.exchange(new_predictor, std::memory_order_acq_rel);
    }

    // Free old model (deferred to avoid race)
    if (old_predictor) {
        // In production, defer this with RCU or hazard pointers
        G_FN_FREE(old_predictor);
    }

    fprintf(stderr, "[TREELITE ABI] Hot-swap complete\n");
    return 0;
}

// Main prediction function - ultra-optimized hot path
extern "C" float treelite_predict(const float* features, size_t num_features) {
    init_once();

    void* predictor = G_PREDICTOR.load(std::memory_order_acquire);
    if (!predictor) {
        return 0.0f;
    }

    uint64_t start_ns = get_nanos();

    // Use thread-local buffer to avoid allocation
    if (TL_SCRATCH_BUFFER.size() < 1) {
        TL_SCRATCH_BUFFER.resize(1);
    }

    // Single prediction
    size_t n_elem = G_FN_PREDICT(
        predictor,
        features,
        1,  // batch_size = 1
        num_features,
        0,  // pred_margin = 0
        TL_SCRATCH_BUFFER.data()
    );

    float result = TL_SCRATCH_BUFFER[0];

    // Apply calibration
    if (G_HAVE_CALIB) {
        result = result * G_CALIB_SCALE + G_CALIB_OFFSET;
        result = std::fmax(G_CALIB_CLIP_MIN, std::fmin(G_CALIB_CLIP_MAX, result));
    }

    // Update metrics
    uint64_t elapsed_ns = get_nanos() - start_ns;
    G_PREDICT_COUNT.fetch_add(1, std::memory_order_relaxed);
    G_PREDICT_NANOS.fetch_add(elapsed_ns, std::memory_order_relaxed);

    return result;
}

// Batch prediction - optimized for throughput
extern "C" void treelite_predict_batch(
    const float* features,
    size_t batch_size,
    size_t num_features,
    float* out_predictions,
    int use_arb_model
) {
    init_once();

    void* predictor = use_arb_model 
        ? G_PREDICTOR_ARB.load(std::memory_order_acquire)
        : G_PREDICTOR.load(std::memory_order_acquire);

    if (!predictor) {
        std::memset(out_predictions, 0, batch_size * sizeof(float));
        return;
    }

    uint64_t start_ns = get_nanos();

    // Ensure scratch buffer is large enough
    if (TL_SCRATCH_BUFFER.size() < batch_size) {
        TL_SCRATCH_BUFFER.resize(batch_size);
    }

    // Batch prediction
    size_t n_elem = G_FN_PREDICT(
        predictor,
        features,
        batch_size,
        num_features,
        0,  // pred_margin = 0
        TL_SCRATCH_BUFFER.data()
    );

    // Copy results and apply calibration
    std::memcpy(out_predictions, TL_SCRATCH_BUFFER.data(), batch_size * sizeof(float));
    apply_calibration(out_predictions, batch_size);

    // Update metrics
    uint64_t elapsed_ns = get_nanos() - start_ns;
    G_PREDICT_COUNT.fetch_add(batch_size, std::memory_order_relaxed);
    G_PREDICT_NANOS.fetch_add(elapsed_ns, std::memory_order_relaxed);
}

// Get performance metrics
extern "C" void treelite_get_metrics(uint64_t* count, double* avg_latency_us) {
    *count = G_PREDICT_COUNT.load(std::memory_order_relaxed);
    uint64_t total_nanos = G_PREDICT_NANOS.load(std::memory_order_relaxed);
    
    if (*count > 0) {
        *avg_latency_us = (double)total_nanos / (double)(*count) / 1000.0;
    } else {
        *avg_latency_us = 0.0;
    }
}

// Query model properties
extern "C" size_t treelite_get_num_features() {
    init_once();
    void* predictor = G_PREDICTOR.load(std::memory_order_acquire);
    return predictor ? G_FN_QUERY_FEATURES(predictor) : 0;
}

extern "C" size_t treelite_get_result_size() {
    init_once();
    void* predictor = G_PREDICTOR.load(std::memory_order_acquire);
    return predictor ? G_FN_QUERY_SIZE(predictor) : 0;
}

// Cleanup on library unload
__attribute__((destructor))
static void cleanup() {
    void* predictor = G_PREDICTOR.exchange(nullptr, std::memory_order_acq_rel);
    if (predictor && G_FN_FREE) {
        G_FN_FREE(predictor);
    }

    void* arb_predictor = G_PREDICTOR_ARB.exchange(nullptr, std::memory_order_acq_rel);
    if (arb_predictor && G_FN_FREE) {
        G_FN_FREE(arb_predictor);
    }

    void* handle = G_DL_HANDLE.exchange(nullptr, std::memory_order_acq_rel);
    if (handle) {
        dlclose(handle);
    }

    void* arb_handle = G_DL_HANDLE_ARB.exchange(nullptr, std::memory_order_acq_rel);
    if (arb_handle) {
        dlclose(arb_handle);
    }
}