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
        // SLM-backed shuffle for chipStar on Intel: subgroup shuffle (OpSubgroupShuffleINTEL)
        // compiles to slower GenISA on Xe-HPC than SLM reads via block-load patterns. Each
        // warp gets its own warpSize-entry slot in shared scratch. Sized for the largest
        // workgroup any HIP kernel uses (256), to keep this header self-contained.
        __shared__ int slm_shfl[256];
        slm_shfl[threadIdx.x] = word;
        __syncwarp();
        const unsigned int warp_base = threadIdx.x & ~(warpSize - 1);
        word = slm_shfl[warp_base + (src_lane & (warpSize - 1))];
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
