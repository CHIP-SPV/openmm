/**
 * This file contains HIP definitions for the macros and functions needed for the
 * common compute framework.
 */

#define KERNEL extern "C" __global__
#define DEVICE __device__
#define LOCAL __shared__
#define LOCAL_ARG
#define GLOBAL
#define RESTRICT __restrict__
#define LOCAL_ID threadIdx.x
#define LOCAL_SIZE blockDim.x
#define GLOBAL_ID (blockIdx.x*blockDim.x+threadIdx.x)
#define GLOBAL_SIZE (blockDim.x*gridDim.x)
#define GROUP_ID blockIdx.x
#define NUM_GROUPS gridDim.x
#define SYNC_THREADS __syncthreads();
#define SYNC_WARPS __syncthreads(); 
//#define SYNC_WARPS {__builtin_amdgcn_wave_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");}
#define MEM_FENCE __threadfence_block();
#define ATOMIC_ADD(dest, value) atomicAdd(dest, value)

#ifdef USE_INT64_ATOMIC_ADD_WORKAROUND
// Intel Arc / IGC bug: OpAtomicIAdd (int32 and int64) fires 4x per call due to
// a SubgroupSize-32 lowering bug. Use CAS retry loops instead, which map to
// OpAtomicCompareExchange — a different SPIR-V instruction that is not affected.
__device__ inline void atomicAddI32(int* addr, int val) {
    int assumed, old = *addr;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, assumed + val);
    } while (assumed != old);
}
__device__ inline void atomicAddI64(unsigned long long* addr, unsigned long long val) {
    unsigned long long assumed, old = *addr;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, assumed + val);
    } while (assumed != old);
}
#undef ATOMIC_ADD
#define ATOMIC_ADD(dest, value) atomicAddI64((unsigned long long*)(dest), (unsigned long long)(value))
#endif

typedef long long mm_long;
typedef unsigned long long mm_ulong;

#define SUPPORTS_DOUBLE_PRECISION 0

#ifdef USE_DOUBLE_PRECISION

__device__ inline long long realToFixedPoint(double x) {
    // Guard against NaN using integer bit comparison (not affected by -ffast-math).
    // NaN has all-ones exponent and non-zero mantissa; Inf has all-ones exponent and
    // zero mantissa. Inf is handled correctly by the saturating integer cast (same as
    // CUDA), so we only zero out NaN.
    // Use __double_as_longlong (HIP intrinsic) for a reliable bitcast in SPIR-V.
    unsigned long long xbits = static_cast<unsigned long long>(__double_as_longlong(x));
    if ((xbits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL &&
        (xbits & 0x000FFFFFFFFFFFFFULL) != 0ULL) return 0LL;
    return static_cast<long long>(x * 0x100000000);
}

#else

__device__ inline long long realToFixedPoint(float x) {
    // Guard against NaN using integer bit comparison (not affected by -ffast-math).
    // On Intel GPU (SPIR-V), static_cast<long long>(NaN * scale) returns INT64_MAX instead
    // of 0 as CUDA does, causing bogus large forces in the fixed-point buffer.
    // Use __float_as_uint (HIP intrinsic) for a reliable bitcast in SPIR-V.
    // NaN: all-ones exponent + non-zero mantissa.  Inf: all-ones exponent + zero mantissa.
    // Inf is handled correctly by the saturating cast on Intel GPU (same as CUDA).
    unsigned int xbits = __float_as_uint(x);
    if ((xbits & 0x7F800000u) == 0x7F800000u && (xbits & 0x007FFFFFu) != 0u) return 0LL;
    return static_cast<long long>(x * (long long)0x100000000LL);
}

#endif
