/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2015 Stanford University and the Authors.      *
 * Portions copyright (c) 2021 Advanced Micro Devices, Inc.                   *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "HipFFT3D.h"
#include "HipContext.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <complex>
#include <cmath>
#include <vector>

using namespace OpenMM;
using namespace std;

// ============================================================
// CPU FFT fallback for Intel Arc via chipStar/CHIP-SPV.
// VkFFT generates GPU kernels compiled by IGC, whose JIT
// optimizations reorder floating-point operations between runs,
// producing non-deterministic results. The CPU implementation
// below uses a general mixed-radix Cooley-Tukey algorithm and
// is exact (deterministic) for any grid size with factors ≤ 7.
// ============================================================

using cfloat = complex<float>;

// In-place 1D FFT: general mixed-radix, forward (inverse=false) or inverse (inverse=true).
// Unnormalized: the inverse does not divide by N.
static void fft1d_cpu(cfloat* data, int n, bool inverse) {
    if (n <= 1) return;
    const float sign = inverse ? 1.0f : -1.0f;
    // Find smallest prime factor of n.
    int p = 2;
    while (n % p != 0) p++;
    const int m = n / p; // n = p * m

    // Collect p sub-arrays of size m: G[s][j] = data[j*p + s].
    vector<cfloat> G(n);
    for (int s = 0; s < p; s++) {
        for (int j = 0; j < m; j++)
            G[s*m + j] = data[j*p + s];
        fft1d_cpu(G.data() + s*m, m, inverse);
    }
    // Combine: X[k] = sum_{s=0}^{p-1} W_n^{sk} * G_s[k mod m]
    // where W_n = exp(sign * 2*pi*i / n).
    const float c = sign * 2.0f * static_cast<float>(M_PI) / n;
    for (int k = 0; k < n; k++) {
        cfloat val(0.0f, 0.0f);
        const int q = k % m;
        for (int s = 0; s < p; s++) {
            const float angle = c * static_cast<float>(s * k);
            val += cfloat(cosf(angle), sinf(angle)) * G[s*m + q];
        }
        data[k] = val;
    }
}

// 3D R2C forward FFT: real[nx][ny][nz] → complex[nx][ny][nz/2+1].
// Both arrays are laid out with z as the fastest-varying index.
static void fft3d_r2c_cpu(const float* in_r, cfloat* out_c, int nx, int ny, int nz) {
    const int nzout = nz / 2 + 1;
    vector<cfloat> work(nx * ny * nz);
    for (int i = 0; i < nx*ny*nz; i++)
        work[i] = cfloat(in_r[i], 0.0f);

    // FFT along z for each (x, y).
    for (int xy = 0; xy < nx*ny; xy++)
        fft1d_cpu(work.data() + xy*nz, nz, false);

    // FFT along y for each (x, z).
    vector<cfloat> row(ny);
    for (int x = 0; x < nx; x++) {
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) row[y] = work[(x*ny+y)*nz+z];
            fft1d_cpu(row.data(), ny, false);
            for (int y = 0; y < ny; y++) work[(x*ny+y)*nz+z] = row[y];
        }
    }

    // FFT along x for each (y, z).
    vector<cfloat> col(nx);
    for (int y = 0; y < ny; y++) {
        for (int z = 0; z < nz; z++) {
            for (int x = 0; x < nx; x++) col[x] = work[(x*ny+y)*nz+z];
            fft1d_cpu(col.data(), nx, false);
            for (int x = 0; x < nx; x++) work[(x*ny+y)*nz+z] = col[x];
        }
    }

    // Copy the non-redundant z slice to output.
    for (int xy = 0; xy < nx*ny; xy++)
        for (int z = 0; z < nzout; z++)
            out_c[xy*nzout+z] = work[xy*nz+z];
}

// 3D C2R inverse FFT: complex[nx][ny][nz/2+1] → real[nx][ny][nz].
// Inverse transforms along x and y first; C2R along z last (preserves
// the Hermitian symmetry that makes the z result real).
static void fft3d_c2r_cpu(const cfloat* in_c, float* out_r, int nx, int ny, int nz) {
    const int nzin = nz / 2 + 1;
    vector<cfloat> work(static_cast<size_t>(nx) * ny * nzin);
    for (int i = 0; i < nx*ny*nzin; i++) work[i] = in_c[i];

    // Inverse FFT along x for each (y, kz).
    vector<cfloat> col(nx);
    for (int y = 0; y < ny; y++) {
        for (int z = 0; z < nzin; z++) {
            for (int x = 0; x < nx; x++) col[x] = work[(x*ny+y)*nzin+z];
            fft1d_cpu(col.data(), nx, true);
            for (int x = 0; x < nx; x++) work[(x*ny+y)*nzin+z] = col[x];
        }
    }

    // Inverse FFT along y for each (x, kz).
    vector<cfloat> row(ny);
    for (int x = 0; x < nx; x++) {
        for (int z = 0; z < nzin; z++) {
            for (int y = 0; y < ny; y++) row[y] = work[(x*ny+y)*nzin+z];
            fft1d_cpu(row.data(), ny, true);
            for (int y = 0; y < ny; y++) work[(x*ny+y)*nzin+z] = row[y];
        }
    }

    // C2R along z for each (x, y): reconstruct Hermitian-symmetric full spectrum,
    // inverse FFT, take real part.
    vector<cfloat> ztmp(nz);
    for (int xy = 0; xy < nx*ny; xy++) {
        const cfloat* wz = work.data() + xy*nzin;
        for (int k = 0; k < nzin; k++) ztmp[k] = wz[k];
        // Hermitian fill: ztmp[nz-k] = conj(ztmp[k]) for k = 1..(nz/2 - 1).
        for (int k = 1; k*2 < nz; k++) ztmp[nz-k] = conj(ztmp[k]);
        fft1d_cpu(ztmp.data(), nz, true);
        for (int z = 0; z < nz; z++)
            out_r[xy*nz+z] = ztmp[z].real();
    }
}

// 3D C2C FFT (in-place): complex[nx][ny][nz].
static void fft3d_c2c_cpu(cfloat* data, int nx, int ny, int nz, bool inverse) {
    for (int xy = 0; xy < nx*ny; xy++)
        fft1d_cpu(data + xy*nz, nz, inverse);

    vector<cfloat> row(ny);
    for (int x = 0; x < nx; x++) {
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) row[y] = data[(x*ny+y)*nz+z];
            fft1d_cpu(row.data(), ny, inverse);
            for (int y = 0; y < ny; y++) data[(x*ny+y)*nz+z] = row[y];
        }
    }

    vector<cfloat> col(nx);
    for (int y = 0; y < ny; y++) {
        for (int z = 0; z < nz; z++) {
            for (int x = 0; x < nx; x++) col[x] = data[(x*ny+y)*nz+z];
            fft1d_cpu(col.data(), nx, inverse);
            for (int x = 0; x < nx; x++) data[(x*ny+y)*nz+z] = col[x];
        }
    }
}
// ============================================================

HipFFT3D::HipFFT3D(HipContext& context, int xsize, int ysize, int zsize, bool realToComplex, hipStream_t stream, HipArray& in, HipArray& out) :
        context(context), stream(stream), cpuFallback(false), xsize(xsize), ysize(ysize), zsize(zsize), isRealToComplex(realToComplex) {

    deviceIndex = context.getDeviceIndex();
    inputBuffer = in.getDevicePointer();
    outputBuffer = out.getDevicePointer();
    size_t valueSize = context.getUseDoublePrecision() ? sizeof(double) : sizeof(float);
    inputBufferSize = zsize * ysize * xsize * valueSize;
    if (realToComplex) {
        outputBufferSize = (zsize/2 + 1) * ysize * xsize * valueSize * 2;
    }
    else {
        outputBufferSize = zsize * ysize * xsize * valueSize;
    }

    // Use a CPU FFT fallback on Intel Arc via chipStar/CHIP-SPV. VkFFT generates HIP kernels
    // compiled by IGC, whose JIT optimizations reorder floating-point operations between runs,
    // producing non-deterministic results that prevent DIIS convergence. The CPU implementation
    // is deterministic. Only enabled for single precision (double is broken on Intel Arc anyway).
    bool intelGPU = !context.getSupportsHardwareFloatGlobalAtomicAdd();
    if (intelGPU && !context.getUseDoublePrecision()) {
        cpuFallback = true;
        app = nullptr;
        // Allocate CPU buffers with correct sizes (pmeGrid uses 2*valueSize per element for complex).
        size_t inBytes  = static_cast<size_t>(xsize) * ysize * zsize * sizeof(float);         // real input
        size_t outBytes = static_cast<size_t>(xsize) * ysize * (zsize/2+1) * 2 * sizeof(float); // complex output
        size_t c2cBytes = static_cast<size_t>(xsize) * ysize * zsize * 2 * sizeof(float);     // C2C
        cpuInBuf.resize(realToComplex ? inBytes  : c2cBytes);
        cpuOutBuf.resize(realToComplex ? outBytes : c2cBytes);
        return;
    }

    VkFFTConfiguration configuration = {};
    configuration.performR2C = realToComplex;
    configuration.device = &deviceIndex;
    configuration.num_streams = 1;
    configuration.stream = &this->stream;
    configuration.doublePrecision = context.getUseDoublePrecision();

    configuration.FFTdim = 3;
    configuration.size[0] = zsize;
    configuration.size[1] = ysize;
    configuration.size[2] = xsize;

    configuration.inverseReturnToInputBuffer = true;
    configuration.isInputFormatted = true;
    configuration.inputBufferSize = &inputBufferSize;
    configuration.inputBuffer = &inputBuffer;
    configuration.inputBufferStride[0] = zsize;
    configuration.inputBufferStride[1] = configuration.inputBufferStride[0] * ysize;
    configuration.inputBufferStride[2] = configuration.inputBufferStride[1] * xsize;

    configuration.bufferSize = &outputBufferSize;
    configuration.buffer = &outputBuffer;
    configuration.bufferStride[0] = realToComplex ? (zsize/2 + 1) : zsize;
    configuration.bufferStride[1] = configuration.bufferStride[0] * ysize;
    configuration.bufferStride[2] = configuration.bufferStride[1] * xsize;

    // Limit VkFFT to ≤512 threads per block. Intel Arc via CHIP-SPV has a hard
    // OpenCL WGS limit of 512; CHIP-SPV does not fill hipDeviceProp_t reliably
    // so we cannot auto-detect. 512 is also a safe bound on AMD/CUDA.
    configuration.maxThreadsNum = 512;

    // On Intel Arc via chipStar, force LUT twiddle factors. The CPU fallback handles the
    // non-determinism fix; this is just a belt-and-suspenders measure for double-precision mode.
    if (intelGPU)
        configuration.useLUT = 1;

    // Combine all parameters into a unique key
    stringstream info;
    int runtimeVersion;
    (void)hipRuntimeGetVersion(&runtimeVersion);
    info << runtimeVersion;
    info << " " << VkFFTGetVersion();
    info << " " << xsize << " " << ysize << " " << zsize;
    info << " " << realToComplex << " " << context.getUseDoublePrecision()
         << " " << (int)configuration.useLUT;

    string cacheFile = context.getCacheFileName(info.str());

    bool hasCache = false;
    vector<char> cacheContent;

    ifstream cache(cacheFile.c_str(), ios::in | ios::binary);
    if (cache.is_open()) {
        cacheContent.insert(cacheContent.begin(), istreambuf_iterator<char>(cache), istreambuf_iterator<char>());
        cache.close();
        hasCache = true;
        // There is an existing cache, load VkFFT kernels from it
        configuration.loadApplicationFromString = 1;
        configuration.loadApplicationString = cacheContent.data();
    }
    else {
        // There is no existing cache, request saving
        configuration.saveApplicationToString = 1;
    }

    app = new VkFFTApplication();
    VkFFTResult fftResult = initializeVkFFT(app, configuration);
    if (fftResult != VKFFT_SUCCESS) {
        throw OpenMMException("Error executing initializeVkFFT: "+context.intToString(fftResult));
    }

    if (!hasCache) {
        // There is no existing cache, create it
        string outputFile = context.getTempFileName() + ".vkfftcache";
        try {
            ofstream out(outputFile.c_str(), ios::out | ios::binary);
            out.write(reinterpret_cast<char*>(app->saveApplicationString), size_t(app->applicationStringSize));
            out.close();
            if (!out.fail()) {
                if (rename(outputFile.c_str(), cacheFile.c_str()) != 0)
                    remove(outputFile.c_str());
            }
        }
        catch (...) {
            // An error occurred.  Possibly we don't have permission to write to the temp directory.
        }
    }
}

HipFFT3D::~HipFFT3D() {
    if (app != nullptr) {
        deleteVkFFT(app);
        delete app;
    }
}

void HipFFT3D::execFFT(bool forward) {
    if (cpuFallback) {
        // Synchronize the stream so GPU writes to the source buffer are complete before we read it.
        hipStreamSynchronize(stream);
        if (isRealToComplex) {
            if (forward) {
                // R2C: inputBuffer (GPU float[nx][ny][nz]) → outputBuffer (GPU cfloat[nx][ny][nzout])
                hipMemcpy(cpuInBuf.data(), inputBuffer, cpuInBuf.size(), hipMemcpyDeviceToHost);
                fft3d_r2c_cpu(reinterpret_cast<const float*>(cpuInBuf.data()),
                               reinterpret_cast<cfloat*>(cpuOutBuf.data()),
                               xsize, ysize, zsize);
                hipMemcpy(outputBuffer, cpuOutBuf.data(), cpuOutBuf.size(), hipMemcpyHostToDevice);
            } else {
                // C2R: outputBuffer (GPU cfloat[nx][ny][nzout]) → inputBuffer (GPU float[nx][ny][nz])
                hipMemcpy(cpuOutBuf.data(), outputBuffer, cpuOutBuf.size(), hipMemcpyDeviceToHost);
                fft3d_c2r_cpu(reinterpret_cast<const cfloat*>(cpuOutBuf.data()),
                               reinterpret_cast<float*>(cpuInBuf.data()),
                               xsize, ysize, zsize);
                hipMemcpy(inputBuffer, cpuInBuf.data(), cpuInBuf.size(), hipMemcpyHostToDevice);
            }
        } else {
            // C2C: inputBuffer and outputBuffer are both complex[nx][ny][nz] (2 floats per element).
            if (forward) {
                hipMemcpy(cpuInBuf.data(), inputBuffer, cpuInBuf.size(), hipMemcpyDeviceToHost);
                fft3d_c2c_cpu(reinterpret_cast<cfloat*>(cpuInBuf.data()), xsize, ysize, zsize, false);
                hipMemcpy(outputBuffer, cpuInBuf.data(), cpuInBuf.size(), hipMemcpyHostToDevice);
            } else {
                hipMemcpy(cpuInBuf.data(), outputBuffer, cpuInBuf.size(), hipMemcpyDeviceToHost);
                fft3d_c2c_cpu(reinterpret_cast<cfloat*>(cpuInBuf.data()), xsize, ysize, zsize, true);
                hipMemcpy(inputBuffer, cpuInBuf.data(), cpuInBuf.size(), hipMemcpyHostToDevice);
            }
        }
        return;
    }
    VkFFTResult fftResult = VkFFTAppend(app, forward ? -1 : 1, NULL);
    if (fftResult != VKFFT_SUCCESS) {
        throw OpenMMException("Error executing VkFFTAppend: "+context.intToString(fftResult));
    }
}

int HipFFT3D::findLegalDimension(int minimum) {
    if (minimum < 1)
        return 1;
    while (true) {
        // Attempt to factor the current value.

        int unfactored = minimum;
        bool valid = true;
        // VkFFT supports prime factors up to 13, but IGC on Intel Arc
        // miscompiles radix-11 and radix-13 kernels (wrong FFT output).
        // Restrict to prime factors ≤7 to avoid those code paths.
        for (int factor = 2; factor <= 7 && unfactored > 1; factor++) {
            int count = 0;
            while (unfactored % factor == 0) {
                unfactored /= factor;
                count++;
            }
            // For odd prime factors, avoid ≥5 occurrences (e.g. 3^5=243, 5^5=3125).
            // Such dimensions cause non-deterministic VkFFT kernel failures on Intel Arc
            // due to race conditions in the generated multi-stage butterfly kernels.
            // Powers of 2 (factor==2) are exempt: VkFFT handles those correctly.
            if (factor > 2 && count >= 5)
                valid = false;
        }
        if (unfactored == 1 && valid)
            return minimum;
        minimum++;
    }
}
