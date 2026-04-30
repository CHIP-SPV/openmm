/**
 * This file contains the device function for using cross-lane operations (ballot and shuffle)
 */

#if defined(TILE_SIZE)
#if !defined(AMD_RDNA)
// Two subwarps per warp
#define SHFL(var, srcLane) __shfl(var, (srcLane) & (TILE_SIZE - 1), TILE_SIZE)
#define BALLOT(var) (unsigned int)(__ballot(var) >> (threadIdx.x & ((64 - 1) ^ (TILE_SIZE - 1))))
#else
#define SHFL(var, srcLane) __shfl(var, srcLane)
#define BALLOT(var) __ballot(var)
#endif
#endif

#ifdef USE_SLM_SHFL
// File-scope SLM scratch shared by all warpShuffle<T> instantiations.
// Without this (declaring __shared__ inside the template), each template
// instantiation gets its own array, multiplying SLM usage by the number of
// distinct T types passed. On Arc that drives the workgroup over the
// per-Xe-core SLM budget and tanks occupancy; observable as slm_size jumping
// from ~32 B to ~4 KB in the kernel zeinfo.
static __device__ __shared__ int slm_shfl_scratch[256];
#endif

template<class T>
static __inline__ __device__
T warpShuffle(const T& input, const int src_lane) {
    static_assert(sizeof(T) % sizeof(int) == 0, "incorrect type size");
    constexpr int words_no = sizeof(T) / sizeof(int);

    T output;
    #pragma unroll
    for(int i = 0; i < words_no; i++) {
        int word;
        __builtin_memcpy(&word, reinterpret_cast<const char*>(&input) + i * sizeof(int), sizeof(int));
#ifdef USE_SLM_SHFL
        // SLM-backed shuffle for chipStar on Intel: subgroup shuffle
        // (OpSubgroupShuffleINTEL) compiles to slower GenISA on Xe-HPC than
        // SLM reads via block-load patterns. Single file-scope scratch is
        // shared across all template instantiations to keep occupancy.
        slm_shfl_scratch[threadIdx.x] = word;
        __syncwarp();
        const unsigned int warp_base = threadIdx.x & ~(warpSize - 1);
        word = slm_shfl_scratch[warp_base + (src_lane & (warpSize - 1))];
#else
        word = __shfl(word, src_lane & (warpSize - 1));
#endif
        __builtin_memcpy(reinterpret_cast<char*>(&output) + i * sizeof(int), &word, sizeof(int));
    }

    return output;
}

template<int Subwarp, class T>
static __inline__ __device__
typename std::enable_if<(Subwarp == warpSize), T>::type
warpRotateLeft(const T& input) {
    return warpShuffle(input, threadIdx.x + 1);
}

template<int Subwarp, class T>
static __inline__ __device__
typename std::enable_if<!(Subwarp == warpSize), T>::type
warpRotateLeft(const T& input) {
    return warpShuffle(input, ((threadIdx.x + 1) & (Subwarp - 1)) | (threadIdx.x & ~(Subwarp - 1)));
}
