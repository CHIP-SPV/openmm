/**
 * This file defines vector operations to simplify code elsewhere.
 */

// Versions of make_x() that take a single value and set all components to that.

inline __device__ int2 make_int2(int a) {
    return make_int2(a, a);
}

inline __device__ int3 make_int3(int a) {
    return make_int3(a, a, a);
}

inline __device__ int4 make_int4(int a) {
    return make_int4(a, a, a, a);
}

inline __device__ float2 make_float2(float a) {
    return make_float2(a, a);
}

inline __device__ float3 make_float3(float a) {
    return make_float3(a, a, a);
}

inline __device__ float4 make_float4(float a) {
    return make_float4(a, a, a, a);
}

inline __device__ double2 make_double2(double a) {
    return make_double2(a, a);
}

inline __device__ double3 make_double3(double a) {
    return make_double3(a, a, a);
}

inline __device__ double4 make_double4(double a) {
    return make_double4(a, a, a, a);
}

// Multiply a vector by a constant.

inline __device__ int2 operator*(int2 a, int b) {
    return make_int2(a.x*b, a.y*b);
}

inline __device__ int3 operator*(int3 a, int b) {
    return make_int3(a.x*b, a.y*b, a.z*b);
}

inline __device__ int4 operator*(int4 a, int b) {
    return make_int4(a.x*b, a.y*b, a.z*b, a.w*b);
}

inline __device__ int2 operator*(int a, int2 b) {
    return make_int2(a*b.x, a*b.y);
}

inline __device__ int3 operator*(int a, int3 b) {
    return make_int3(a*b.x, a*b.y, a*b.z);
}

inline __device__ int4 operator*(int a, int4 b) {
    return make_int4(a*b.x, a*b.y, a*b.z, a*b.w);
}

inline __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x*b, a.y*b);
}

inline __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x*b, a.y*b, a.z*b, a.w*b);
}

inline __device__ float2 operator*(float a, float2 b) {
    return make_float2(a*b.x, a*b.y);
}

inline __device__ float3 operator*(float a, float3 b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

inline __device__ float4 operator*(float a, float4 b) {
    return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);
}

inline __device__ double2 operator*(double2 a, double b) {
    return make_double2(a.x*b, a.y*b);
}

inline __device__ double3 operator*(double3 a, double b) {
    return make_double3(a.x*b, a.y*b, a.z*b);
}

inline __device__ double4 operator*(double4 a, double b) {
    return make_double4(a.x*b, a.y*b, a.z*b, a.w*b);
}

inline __device__ double2 operator*(double a, double2 b) {
    return make_double2(a*b.x, a*b.y);
}

inline __device__ double3 operator*(double a, double3 b) {
    return make_double3(a*b.x, a*b.y, a*b.z);
}

inline __device__ double4 operator*(double a, double4 b) {
    return make_double4(a*b.x, a*b.y, a*b.z, a*b.w);
}

// Divide a vector by a constant.

inline __device__ int2 operator/(int2 a, int b) {
    return make_int2(a.x/b, a.y/b);
}

inline __device__ int3 operator/(int3 a, int b) {
    return make_int3(a.x/b, a.y/b, a.z/b);
}

inline __device__ int4 operator/(int4 a, int b) {
    return make_int4(a.x/b, a.y/b, a.z/b, a.w/b);
}

inline __device__ float2 operator/(float2 a, float b) {
    float scale = 1.0f/b;
    return a*scale;
}

inline __device__ float3 operator/(float3 a, float b) {
    float scale = 1.0f/b;
    return a*scale;
}

inline __device__ float4 operator/(float4 a, float b) {
    float scale = 1.0f/b;
    return a*scale;
}

inline __device__ double2 operator/(double2 a, double b) {
    double scale = 1.0/b;
    return a*scale;
}

inline __device__ double3 operator/(double3 a, double b) {
    double scale = 1.0/b;
    return a*scale;
}

inline __device__ double4 operator/(double4 a, double b) {
    double scale = 1.0/b;
    return a*scale;
}

// *= operator (multiply vector by constant)

inline __device__ void operator*=(int2& a, int b) {
    a.x *= b; a.y *= b;
}

inline __device__ void operator*=(int3& a, int b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline __device__ void operator*=(int4& a, int b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

inline __device__ void operator*=(float2& a, float b) {
    a.x *= b; a.y *= b;
}

inline __device__ void operator*=(float3& a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline __device__ void operator*=(float4& a, float b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

inline __device__ void operator*=(double2& a, double b) {
    a.x *= b; a.y *= b;
}

inline __device__ void operator*=(double3& a, double b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline __device__ void operator*=(double4& a, double b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

// Dot product

inline __device__ float dot(float3 a, float3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

inline __device__ double dot(double3 a, double3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

// Cross product

inline __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

inline __device__ float4 cross(float4 a, float4 b) {
    return make_float4(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x, 0.0f);
}

inline __device__ double3 cross(double3 a, double3 b) {
    return make_double3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

inline __device__ double4 cross(double4 a, double4 b) {
    return make_double4(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x, 0.0);
}

// Normalize a vector

inline __device__ float2 normalize(float2 a) {
    return a*rsqrtf(a.x*a.x+a.y*a.y);
}

inline __device__ float3 normalize(float3 a) {
    return a*rsqrtf(a.x*a.x+a.y*a.y+a.z*a.z);
}

inline __device__ float4 normalize(float4 a) {
    return a*rsqrtf(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);
}

inline __device__ double2 normalize(double2 a) {
    return a*rsqrt(a.x*a.x+a.y*a.y);
}

inline __device__ double3 normalize(double3 a) {
    return a*rsqrt(a.x*a.x+a.y*a.y+a.z*a.z);
}

inline __device__ double4 normalize(double4 a) {
    return a*rsqrt(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);
}

// Strip off the fourth component of a vector.

inline __device__ short3 trimTo3(short4 v) {
    return make_short3(v.x, v.y, v.z);
}

inline __device__ int3 trimTo3(int4 v) {
    return make_int3(v.x, v.y, v.z);
}

inline __device__ float3 trimTo3(float4 v) {
    return make_float3(v.x, v.y, v.z);
}

inline __device__ double3 trimTo3(double4 v) {
    return make_double3(v.x, v.y, v.z);
}

// 2D bicubic and 3D tricubic interpolation helpers.
// Defined as static (internal linkage) with __attribute__((noinline)) so that LLVM
// treats noinline as a hard constraint rather than a hint.  This keeps the body as a
// separate SPIR-V function instead of inlining it 5x into the nonbonded kernel inner
// function, which would make it too large for IGC on Intel Arc GPU (triggering IGC
// RetryManager and silent kernel drops with SubgroupSize 32).
// Float and double overloads are provided so the right one is selected by the
// generated code regardless of the 'real' typedef.
//
// interp3D_tricubic_eval: 3D tricubic interpolation.
// table      - coefficient table; 16 consecutive float4 entries per grid cell
// coeffIndex - first entry index for this cell (= 16*(s + nx*(t + ny*u)))
// da,db,dc   - fractional offsets after coordinate normalization
// out_val    - output: f(x,y,z)
// out_dx     - df/dx in normalized coords (caller scales by x grid spacing)
// out_dy     - df/dy in normalized coords
// out_dz     - df/dz in normalized coords
static __device__ __attribute__((noinline)) void interp3D_tricubic_eval(
    const float4* __restrict__ table,
    int coeffIndex, float da, float db, float dc,
    float& out_val, float& out_dx, float& out_dy, float& out_dz)
{
    float4 c0  = table[coeffIndex+0];
    float4 c1  = table[coeffIndex+1];
    float4 c2  = table[coeffIndex+2];
    float4 c3  = table[coeffIndex+3];
    float4 c4  = table[coeffIndex+4];
    float4 c5  = table[coeffIndex+5];
    float4 c6  = table[coeffIndex+6];
    float4 c7  = table[coeffIndex+7];
    float4 c8  = table[coeffIndex+8];
    float4 c9  = table[coeffIndex+9];
    float4 c10 = table[coeffIndex+10];
    float4 c11 = table[coeffIndex+11];
    float4 c12 = table[coeffIndex+12];
    float4 c13 = table[coeffIndex+13];
    float4 c14 = table[coeffIndex+14];
    float4 c15 = table[coeffIndex+15];
    // value[m] = bicubic in (da,db) for the m-th z-slice; accumulated via Horner in db
    float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    v0 = db*v0 + ((c3.w*da+c3.z)*da+c3.y)*da+c3.x;
    v1 = db*v1 + ((c7.w*da+c7.z)*da+c7.y)*da+c7.x;
    v2 = db*v2 + ((c11.w*da+c11.z)*da+c11.y)*da+c11.x;
    v3 = db*v3 + ((c15.w*da+c15.z)*da+c15.y)*da+c15.x;
    v0 = db*v0 + ((c2.w*da+c2.z)*da+c2.y)*da+c2.x;
    v1 = db*v1 + ((c6.w*da+c6.z)*da+c6.y)*da+c6.x;
    v2 = db*v2 + ((c10.w*da+c10.z)*da+c10.y)*da+c10.x;
    v3 = db*v3 + ((c14.w*da+c14.z)*da+c14.y)*da+c14.x;
    v0 = db*v0 + ((c1.w*da+c1.z)*da+c1.y)*da+c1.x;
    v1 = db*v1 + ((c5.w*da+c5.z)*da+c5.y)*da+c5.x;
    v2 = db*v2 + ((c9.w*da+c9.z)*da+c9.y)*da+c9.x;
    v3 = db*v3 + ((c13.w*da+c13.z)*da+c13.y)*da+c13.x;
    v0 = db*v0 + ((c0.w*da+c0.z)*da+c0.y)*da+c0.x;
    v1 = db*v1 + ((c4.w*da+c4.z)*da+c4.y)*da+c4.x;
    v2 = db*v2 + ((c8.w*da+c8.z)*da+c8.y)*da+c8.x;
    v3 = db*v3 + ((c12.w*da+c12.z)*da+c12.y)*da+c12.x;
    out_val = v0 + dc*(v1 + dc*(v2 + dc*v3));
    out_dz  = v1 + dc*(2.0f*v2 + dc*3.0f*v3);
    // derivx[m] = d(value[m])/da; accumulated via Horner' in da, Horner in db
    float dx0 = 0, dx1 = 0, dx2 = 0, dx3 = 0;
    dx0 = db*dx0 + (3.0f*c3.w*da+2.0f*c3.z)*da+c3.y;
    dx1 = db*dx1 + (3.0f*c7.w*da+2.0f*c7.z)*da+c7.y;
    dx2 = db*dx2 + (3.0f*c11.w*da+2.0f*c11.z)*da+c11.y;
    dx3 = db*dx3 + (3.0f*c15.w*da+2.0f*c15.z)*da+c15.y;
    dx0 = db*dx0 + (3.0f*c2.w*da+2.0f*c2.z)*da+c2.y;
    dx1 = db*dx1 + (3.0f*c6.w*da+2.0f*c6.z)*da+c6.y;
    dx2 = db*dx2 + (3.0f*c10.w*da+2.0f*c10.z)*da+c10.y;
    dx3 = db*dx3 + (3.0f*c14.w*da+2.0f*c14.z)*da+c14.y;
    dx0 = db*dx0 + (3.0f*c1.w*da+2.0f*c1.z)*da+c1.y;
    dx1 = db*dx1 + (3.0f*c5.w*da+2.0f*c5.z)*da+c5.y;
    dx2 = db*dx2 + (3.0f*c9.w*da+2.0f*c9.z)*da+c9.y;
    dx3 = db*dx3 + (3.0f*c13.w*da+2.0f*c13.z)*da+c13.y;
    dx0 = db*dx0 + (3.0f*c0.w*da+2.0f*c0.z)*da+c0.y;
    dx1 = db*dx1 + (3.0f*c4.w*da+2.0f*c4.z)*da+c4.y;
    dx2 = db*dx2 + (3.0f*c8.w*da+2.0f*c8.z)*da+c8.y;
    dx3 = db*dx3 + (3.0f*c12.w*da+2.0f*c12.z)*da+c12.y;
    out_dx = dx0 + dc*(dx1 + dc*(dx2 + dc*dx3));
    // derivy[m] = d(value[m])/db; Horner in da outer, Horner' in db inner
    float dy0 = 0, dy1 = 0, dy2 = 0, dy3 = 0;
    dy0 = da*dy0 + (3.0f*c3.w*db+2.0f*c2.w)*db+c1.w;
    dy1 = da*dy1 + (3.0f*c7.w*db+2.0f*c6.w)*db+c5.w;
    dy2 = da*dy2 + (3.0f*c11.w*db+2.0f*c10.w)*db+c9.w;
    dy3 = da*dy3 + (3.0f*c15.w*db+2.0f*c14.w)*db+c13.w;
    dy0 = da*dy0 + (3.0f*c3.z*db+2.0f*c2.z)*db+c1.z;
    dy1 = da*dy1 + (3.0f*c7.z*db+2.0f*c6.z)*db+c5.z;
    dy2 = da*dy2 + (3.0f*c11.z*db+2.0f*c10.z)*db+c9.z;
    dy3 = da*dy3 + (3.0f*c15.z*db+2.0f*c14.z)*db+c13.z;
    dy0 = da*dy0 + (3.0f*c3.y*db+2.0f*c2.y)*db+c1.y;
    dy1 = da*dy1 + (3.0f*c7.y*db+2.0f*c6.y)*db+c5.y;
    dy2 = da*dy2 + (3.0f*c11.y*db+2.0f*c10.y)*db+c9.y;
    dy3 = da*dy3 + (3.0f*c15.y*db+2.0f*c14.y)*db+c13.y;
    dy0 = da*dy0 + (3.0f*c3.x*db+2.0f*c2.x)*db+c1.x;
    dy1 = da*dy1 + (3.0f*c7.x*db+2.0f*c6.x)*db+c5.x;
    dy2 = da*dy2 + (3.0f*c11.x*db+2.0f*c10.x)*db+c9.x;
    dy3 = da*dy3 + (3.0f*c15.x*db+2.0f*c14.x)*db+c13.x;
    out_dy = dy0 + dc*(dy1 + dc*(dy2 + dc*dy3));
}
// double overload of interp3D_tricubic_eval
static __device__ __attribute__((noinline)) void interp3D_tricubic_eval(
    const float4* __restrict__ table,
    int coeffIndex, double da, double db, double dc,
    double& out_val, double& out_dx, double& out_dy, double& out_dz)
{
    float4 c0  = table[coeffIndex+0];
    float4 c1  = table[coeffIndex+1];
    float4 c2  = table[coeffIndex+2];
    float4 c3  = table[coeffIndex+3];
    float4 c4  = table[coeffIndex+4];
    float4 c5  = table[coeffIndex+5];
    float4 c6  = table[coeffIndex+6];
    float4 c7  = table[coeffIndex+7];
    float4 c8  = table[coeffIndex+8];
    float4 c9  = table[coeffIndex+9];
    float4 c10 = table[coeffIndex+10];
    float4 c11 = table[coeffIndex+11];
    float4 c12 = table[coeffIndex+12];
    float4 c13 = table[coeffIndex+13];
    float4 c14 = table[coeffIndex+14];
    float4 c15 = table[coeffIndex+15];
    double v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    v0 = db*v0 + ((c3.w*da+c3.z)*da+c3.y)*da+c3.x;
    v1 = db*v1 + ((c7.w*da+c7.z)*da+c7.y)*da+c7.x;
    v2 = db*v2 + ((c11.w*da+c11.z)*da+c11.y)*da+c11.x;
    v3 = db*v3 + ((c15.w*da+c15.z)*da+c15.y)*da+c15.x;
    v0 = db*v0 + ((c2.w*da+c2.z)*da+c2.y)*da+c2.x;
    v1 = db*v1 + ((c6.w*da+c6.z)*da+c6.y)*da+c6.x;
    v2 = db*v2 + ((c10.w*da+c10.z)*da+c10.y)*da+c10.x;
    v3 = db*v3 + ((c14.w*da+c14.z)*da+c14.y)*da+c14.x;
    v0 = db*v0 + ((c1.w*da+c1.z)*da+c1.y)*da+c1.x;
    v1 = db*v1 + ((c5.w*da+c5.z)*da+c5.y)*da+c5.x;
    v2 = db*v2 + ((c9.w*da+c9.z)*da+c9.y)*da+c9.x;
    v3 = db*v3 + ((c13.w*da+c13.z)*da+c13.y)*da+c13.x;
    v0 = db*v0 + ((c0.w*da+c0.z)*da+c0.y)*da+c0.x;
    v1 = db*v1 + ((c4.w*da+c4.z)*da+c4.y)*da+c4.x;
    v2 = db*v2 + ((c8.w*da+c8.z)*da+c8.y)*da+c8.x;
    v3 = db*v3 + ((c12.w*da+c12.z)*da+c12.y)*da+c12.x;
    out_val = v0 + dc*(v1 + dc*(v2 + dc*v3));
    out_dz  = v1 + dc*(2.0*v2 + dc*3.0*v3);
    double dx0 = 0, dx1 = 0, dx2 = 0, dx3 = 0;
    dx0 = db*dx0 + (3.0*c3.w*da+2.0*c3.z)*da+c3.y;
    dx1 = db*dx1 + (3.0*c7.w*da+2.0*c7.z)*da+c7.y;
    dx2 = db*dx2 + (3.0*c11.w*da+2.0*c11.z)*da+c11.y;
    dx3 = db*dx3 + (3.0*c15.w*da+2.0*c15.z)*da+c15.y;
    dx0 = db*dx0 + (3.0*c2.w*da+2.0*c2.z)*da+c2.y;
    dx1 = db*dx1 + (3.0*c6.w*da+2.0*c6.z)*da+c6.y;
    dx2 = db*dx2 + (3.0*c10.w*da+2.0*c10.z)*da+c10.y;
    dx3 = db*dx3 + (3.0*c14.w*da+2.0*c14.z)*da+c14.y;
    dx0 = db*dx0 + (3.0*c1.w*da+2.0*c1.z)*da+c1.y;
    dx1 = db*dx1 + (3.0*c5.w*da+2.0*c5.z)*da+c5.y;
    dx2 = db*dx2 + (3.0*c9.w*da+2.0*c9.z)*da+c9.y;
    dx3 = db*dx3 + (3.0*c13.w*da+2.0*c13.z)*da+c13.y;
    dx0 = db*dx0 + (3.0*c0.w*da+2.0*c0.z)*da+c0.y;
    dx1 = db*dx1 + (3.0*c4.w*da+2.0*c4.z)*da+c4.y;
    dx2 = db*dx2 + (3.0*c8.w*da+2.0*c8.z)*da+c8.y;
    dx3 = db*dx3 + (3.0*c12.w*da+2.0*c12.z)*da+c12.y;
    out_dx = dx0 + dc*(dx1 + dc*(dx2 + dc*dx3));
    double dy0 = 0, dy1 = 0, dy2 = 0, dy3 = 0;
    dy0 = da*dy0 + (3.0*c3.w*db+2.0*c2.w)*db+c1.w;
    dy1 = da*dy1 + (3.0*c7.w*db+2.0*c6.w)*db+c5.w;
    dy2 = da*dy2 + (3.0*c11.w*db+2.0*c10.w)*db+c9.w;
    dy3 = da*dy3 + (3.0*c15.w*db+2.0*c14.w)*db+c13.w;
    dy0 = da*dy0 + (3.0*c3.z*db+2.0*c2.z)*db+c1.z;
    dy1 = da*dy1 + (3.0*c7.z*db+2.0*c6.z)*db+c5.z;
    dy2 = da*dy2 + (3.0*c11.z*db+2.0*c10.z)*db+c9.z;
    dy3 = da*dy3 + (3.0*c15.z*db+2.0*c14.z)*db+c13.z;
    dy0 = da*dy0 + (3.0*c3.y*db+2.0*c2.y)*db+c1.y;
    dy1 = da*dy1 + (3.0*c7.y*db+2.0*c6.y)*db+c5.y;
    dy2 = da*dy2 + (3.0*c11.y*db+2.0*c10.y)*db+c9.y;
    dy3 = da*dy3 + (3.0*c15.y*db+2.0*c14.y)*db+c13.y;
    dy0 = da*dy0 + (3.0*c3.x*db+2.0*c2.x)*db+c1.x;
    dy1 = da*dy1 + (3.0*c7.x*db+2.0*c6.x)*db+c5.x;
    dy2 = da*dy2 + (3.0*c11.x*db+2.0*c10.x)*db+c9.x;
    dy3 = da*dy3 + (3.0*c15.x*db+2.0*c14.x)*db+c13.x;
    out_dy = dy0 + dc*(dy1 + dc*(dy2 + dc*dy3));
}

// interp2D_bicubic_eval: 2D bicubic interpolation.
// table      - coefficient table; 4 consecutive float4 entries per grid cell
// coeffIndex - first entry index for this cell (= 4*(s + nx*t))
// da,db      - fractional offsets after coordinate normalization
// out_val    - output: f(x,y)
// out_dx     - df/dx in normalized coordinates (caller scales by grid spacing)
// out_dy     - df/dy in normalized coordinates
static __device__ __attribute__((noinline)) void interp2D_bicubic_eval(
    const float4* __restrict__ table,
    int coeffIndex, float da, float db,
    float& out_val, float& out_dx, float& out_dy) {
    float4 c0 = table[coeffIndex];
    float4 c1 = table[coeffIndex+1];
    float4 c2 = table[coeffIndex+2];
    float4 c3 = table[coeffIndex+3];
    float v = 0;
    v = da*v + ((c3.w*db + c3.z)*db + c3.y)*db + c3.x;
    v = da*v + ((c2.w*db + c2.z)*db + c2.y)*db + c2.x;
    v = da*v + ((c1.w*db + c1.z)*db + c1.y)*db + c1.x;
    v = da*v + ((c0.w*db + c0.z)*db + c0.y)*db + c0.x;
    out_val = v;
    float dvx = 0;
    dvx = db*dvx + (3.0f*c3.w*da + 2.0f*c2.w)*da + c1.w;
    dvx = db*dvx + (3.0f*c3.z*da + 2.0f*c2.z)*da + c1.z;
    dvx = db*dvx + (3.0f*c3.y*da + 2.0f*c2.y)*da + c1.y;
    dvx = db*dvx + (3.0f*c3.x*da + 2.0f*c2.x)*da + c1.x;
    out_dx = dvx;
    float dvy = 0;
    dvy = da*dvy + (3.0f*c3.w*db + 2.0f*c3.z)*db + c3.y;
    dvy = da*dvy + (3.0f*c2.w*db + 2.0f*c2.z)*db + c2.y;
    dvy = da*dvy + (3.0f*c1.w*db + 2.0f*c1.z)*db + c1.y;
    dvy = da*dvy + (3.0f*c0.w*db + 2.0f*c0.z)*db + c0.y;
    out_dy = dvy;
}
// double overload of interp2D_bicubic_eval
static __device__ __attribute__((noinline)) void interp2D_bicubic_eval(
    const float4* __restrict__ table,
    int coeffIndex, double da, double db,
    double& out_val, double& out_dx, double& out_dy) {
    float4 c0 = table[coeffIndex];
    float4 c1 = table[coeffIndex+1];
    float4 c2 = table[coeffIndex+2];
    float4 c3 = table[coeffIndex+3];
    double v = 0;
    v = da*v + ((c3.w*db + c3.z)*db + c3.y)*db + c3.x;
    v = da*v + ((c2.w*db + c2.z)*db + c2.y)*db + c2.x;
    v = da*v + ((c1.w*db + c1.z)*db + c1.y)*db + c1.x;
    v = da*v + ((c0.w*db + c0.z)*db + c0.y)*db + c0.x;
    out_val = v;
    double dvx = 0;
    dvx = db*dvx + (3.0*c3.w*da + 2.0*c2.w)*da + c1.w;
    dvx = db*dvx + (3.0*c3.z*da + 2.0*c2.z)*da + c1.z;
    dvx = db*dvx + (3.0*c3.y*da + 2.0*c2.y)*da + c1.y;
    dvx = db*dvx + (3.0*c3.x*da + 2.0*c2.x)*da + c1.x;
    out_dx = dvx;
    double dvy = 0;
    dvy = da*dvy + (3.0*c3.w*db + 2.0*c3.z)*db + c3.y;
    dvy = da*dvy + (3.0*c2.w*db + 2.0*c2.z)*db + c2.y;
    dvy = da*dvy + (3.0*c1.w*db + 2.0*c1.z)*db + c1.y;
    dvy = da*dvy + (3.0*c0.w*db + 2.0*c0.z)*db + c0.y;
    out_dy = dvy;
}
