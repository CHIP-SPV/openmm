/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2015 Stanford University and the Authors.           *
 * Portions copyright (c) 2020 Advanced Micro Devices, Inc.                   *
 * Authors: Peter Eastman, Nicholas Curtis                                    *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#define CUSTOM_MANY_PARTICLE_MAIN_DEFINED
#include "HipTests.h"
#include "TestCustomManyParticleForce.h"
#include <string>

void runPlatformTests() {
}

// argv[2] selects a single test by name; if omitted, runs all tests.
int main(int argc, char* argv[]) {
    try {
        if (argc > 1)
            platform.setPropertyDefaultValue("Precision", std::string(argv[1]));
        std::string testFilter = (argc > 2) ? std::string(argv[2]) : "";

#define RUN(name) if (testFilter.empty() || testFilter == #name) { name(); if (!testFilter.empty()) { std::cout << "Done" << std::endl; return 0; } }
#define RUN2(name, arg) if (testFilter.empty() || testFilter == #name) { name(arg); if (!testFilter.empty()) { std::cout << "Done" << std::endl; return 0; } }

        if (testFilter.empty() || testFilter == "testNoCutoff") { testNoCutoff(true); testNoCutoff(false); if (!testFilter.empty()) { std::cout << "Done" << std::endl; return 0; } }
        if (testFilter.empty() || testFilter == "testCutoff") { testCutoff(true); testCutoff(false); if (!testFilter.empty()) { std::cout << "Done" << std::endl; return 0; } }
        if (testFilter.empty() || testFilter == "testPeriodic") { testPeriodic(true); testPeriodic(false); if (!testFilter.empty()) { std::cout << "Done" << std::endl; return 0; } }
        if (testFilter.empty() || testFilter == "testTriclinic") { testTriclinic(true); testTriclinic(false); if (!testFilter.empty()) { std::cout << "Done" << std::endl; return 0; } }
        RUN(testExclusions)
        RUN(testAllTerms)
        RUN(testParameters)
        RUN(testTabulatedFunctions)
        RUN(testTypeFilters)
        RUN(testLargeSystem)
        RUN(testCentralParticleModeNoCutoff)
        RUN(testCentralParticleModeCutoff)
        RUN(testCentralParticleModeLargeSystem)
        RUN(testIllegalVariable)
        if (testFilter.empty()) runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
