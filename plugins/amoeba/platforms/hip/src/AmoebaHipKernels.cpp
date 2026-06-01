/* -------------------------------------------------------------------------- *
 *                               OpenMMAmoeba                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Portions copyright (c) 2021 Advanced Micro Devices, Inc.                   *
 * Authors: Peter Eastman, Mark Friedrichs                                    *
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

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "AmoebaHipKernels.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/AmoebaGeneralizedKirkwoodForceImpl.h"
#include "openmm/internal/AmoebaMultipoleForceImpl.h"
#include "openmm/internal/AmoebaWcaDispersionForceImpl.h"
#include "openmm/internal/AmoebaTorsionTorsionForceImpl.h"
#include "openmm/internal/AmoebaVdwForceImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "HipBondedUtilities.h"
#include "HipFFT3D.h"
#include "HipForceInfo.h"
#include "HipKernelSources.h"
#include "SimTKOpenMMRealType.h"
#include "jama_lu.h"

#include <algorithm>
#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

/* -------------------------------------------------------------------------- *
 *                             AmoebaMultipole                                *
 * -------------------------------------------------------------------------- */

HipCalcAmoebaMultipoleForceKernel::~HipCalcAmoebaMultipoleForceKernel() {
    ContextSelector selector(cc);
    if (fft != NULL)
        delete fft;
#ifdef OPENMM_HIP_WITH_HIPFFT
    if (useHipFFT)
        hipfftDestroy(hipFft);
#endif
}

void HipCalcAmoebaMultipoleForceKernel::initialize(const System& system, const AmoebaMultipoleForce& force) {
    CommonCalcAmoebaMultipoleForceKernel::initialize(system, force);
    if (usePME) {
        ContextSelector selector(cc);
#ifdef OPENMM_HIP_WITH_HIPFFT
        int cufftVersion;
        hipfftGetVersion(&cufftVersion);
        useHipFFT = (cufftVersion >= 7050);
        if (useHipFFT) {
            hipfftResult result = hipfftPlan3d(&hipFft, gridSizeX, gridSizeY, gridSizeZ, cc.getUseDoublePrecision() ? HIPFFT_Z2Z : HIPFFT_C2C);
            if (result != HIPFFT_SUCCESS)
                throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
        }
        else
#endif
            fft = new HipFFT3D(cu, gridSizeX, gridSizeY, gridSizeZ, false);
    }
}

void HipCalcAmoebaMultipoleForceKernel::computeFFT(bool forward) {
    HipArray& grid1 = cu.unwrap(pmeGrid1);
    HipArray& grid2 = cu.unwrap(pmeGrid2);
#ifdef OPENMM_HIP_WITH_HIPFFT
    if (useHipFFT) {
        if (forward) {
            if (cc.getUseDoublePrecision())
                hipfftExecZ2Z(hipFft, (double2*) grid1.getDevicePointer(), (double2*) grid2.getDevicePointer(), HIPFFT_FORWARD);
            else
                hipfftExecC2C(hipFft, (float2*) grid1.getDevicePointer(), (float2*) grid2.getDevicePointer(), HIPFFT_FORWARD);
        }
        else {
            if (cc.getUseDoublePrecision())
                hipfftExecZ2Z(hipFft, (double2*) grid2.getDevicePointer(), (double2*) grid1.getDevicePointer(), HIPFFT_BACKWARD);
            else
                hipfftExecC2C(hipFft, (float2*) grid2.getDevicePointer(), (float2*) grid1.getDevicePointer(), HIPFFT_BACKWARD);
        }
        return;
    }
#endif
    if (forward)
        fft->execFFT(grid1, grid2, true);
    else
        fft->execFFT(grid2, grid1, false);
}

/* -------------------------------------------------------------------------- *
 *                           HippoNonbondedForce                              *
 * -------------------------------------------------------------------------- */

HipCalcHippoNonbondedForceKernel::~HipCalcHippoNonbondedForceKernel() {
    ContextSelector selector(cc);
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
    if (dfft != NULL)
        delete dfft;
#ifdef OPENMM_HIP_WITH_HIPFFT
    if (useHipFFT) {
        hipfftDestroy(fftForward);
        hipfftDestroy(fftBackward);
        hipfftDestroy(dfftForward);
        hipfftDestroy(dfftBackward);
    }
#endif
}

void HipCalcHippoNonbondedForceKernel::initialize(const System& system, const HippoNonbondedForce& force) {
    CommonCalcHippoNonbondedForceKernel::initialize(system, force);
    if (usePME) {
        ContextSelector selector(cc);
        sort = new HipSort(cu, new SortTrait(), cc.getNumAtoms());
#ifdef OPENMM_HIP_WITH_HIPFFT
        int cufftVersion;
        hipfftGetVersion(&cufftVersion);
        useHipFFT = (cufftVersion >= 7050);
        if (useHipFFT) {
            hipfftResult result;
            result = hipfftPlan3d(&fftForward, gridSizeX, gridSizeY, gridSizeZ, cc.getUseDoublePrecision() ? HIPFFT_D2Z : HIPFFT_R2C);
            if (result != HIPFFT_SUCCESS) throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
            result = hipfftPlan3d(&fftBackward, gridSizeX, gridSizeY, gridSizeZ, cc.getUseDoublePrecision() ? HIPFFT_Z2D : HIPFFT_C2R);
            if (result != HIPFFT_SUCCESS) throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
            result = hipfftPlan3d(&dfftForward, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, cc.getUseDoublePrecision() ? HIPFFT_D2Z : HIPFFT_R2C);
            if (result != HIPFFT_SUCCESS) throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
            result = hipfftPlan3d(&dfftBackward, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, cc.getUseDoublePrecision() ? HIPFFT_Z2D : HIPFFT_C2R);
            if (result != HIPFFT_SUCCESS) throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
        }
        else
#endif
        {
            fft = new HipFFT3D(cu, gridSizeX, gridSizeY, gridSizeZ, true);
            dfft = new HipFFT3D(cu, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, true);
        }
    }
}

void HipCalcHippoNonbondedForceKernel::computeFFT(bool forward, bool dispersion) {
    HipArray& grid1 = cu.unwrap(pmeGrid1);
    HipArray& grid2 = cu.unwrap(pmeGrid2);
#ifdef OPENMM_HIP_WITH_HIPFFT
    if (useHipFFT) {
        if (forward) {
            hipfftHandle plan = dispersion ? dfftForward : fftForward;
            if (cc.getUseDoublePrecision())
                hipfftExecD2Z(plan, (double*) grid1.getDevicePointer(), (double2*) grid2.getDevicePointer());
            else
                hipfftExecR2C(plan, (float*) grid1.getDevicePointer(), (float2*) grid2.getDevicePointer());
        }
        else {
            hipfftHandle plan = dispersion ? dfftBackward : fftBackward;
            if (cc.getUseDoublePrecision())
                hipfftExecZ2D(plan, (double2*) grid2.getDevicePointer(), (double*) grid1.getDevicePointer());
            else
                hipfftExecC2R(plan, (float2*) grid2.getDevicePointer(), (float*) grid1.getDevicePointer());
        }
        return;
    }
#endif
    HipFFT3D* f = dispersion ? dfft : fft;
    if (forward)
        f->execFFT(grid1, grid2, true);
    else
        f->execFFT(grid2, grid1, false);
}

void HipCalcHippoNonbondedForceKernel::sortGridIndex() {
    sort->sort(dynamic_cast<HipContext&>(cc).unwrap(pmeAtomGridIndex));
}
