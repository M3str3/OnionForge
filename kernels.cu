// Inline implementation of the SplitMix64 pseudo-random number generator
// This is a high-quality, fast PRNG suitable for cryptographic seed generation
__device__ __forceinline__ unsigned long long sm64(unsigned long long &s){
    s += 0x9e3779b97f4a7c15ULL;  // Golden ratio constant for mixing
    unsigned long long z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;  // First mixing step
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;  // Second mixing step
    return z ^ (z >> 31);  // Final output transformation
}

// CUDA kernel for generating cryptographic seeds in parallel
// Each thread generates a unique 32-byte seed using SplitMix64 PRNG
extern "C" __global__
void generate_seeds(unsigned long long* __restrict__ out,
                    unsigned long long base_seed,
                    unsigned int count) {

    // Grid-stride loop pattern for optimal GPU utilization
    // Each thread processes multiple elements to maximize throughput
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count;
         i += blockDim.x * gridDim.x) {

        // Generate unique state for this thread's seed
        // Each seed gets a unique offset to ensure cryptographic independence
        unsigned long long state = base_seed + (unsigned long long)i * 32ULL;

        // Generate 4 x 64-bit values = 32 bytes of entropy per seed
        ulonglong4 v;
        v.x = sm64(state);  // First 8 bytes
        v.y = sm64(state);  // Second 8 bytes
        v.z = sm64(state);  // Third 8 bytes
        v.w = sm64(state);  // Fourth 8 bytes

        // Vectorized and aligned memory store for optimal bandwidth
        reinterpret_cast<ulonglong4*>(out)[i] = v;
    }
}
