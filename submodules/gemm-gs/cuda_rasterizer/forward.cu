/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_bf16.h>

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>


namespace cg = cooperative_groups;
#define SIZE 512
#define BATCH_SIZE 16
#define VEC_LEN 8
#define WARP_LEN 8
#define log2e 1.442695f
using namespace nvcuda;

__forceinline__ __device__ float fast_exp2_approx(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__forceinline__ __device__ void load_matrix_x4(
	uint &reg0, uint &reg1, uint &reg2, uint &reg3,
	size_t smem_addr
)
{
	asm volatile(
		"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
		: "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3) 
		: "l"(smem_addr)
	);
}

__forceinline__ __device__ void load_matrix_x2(
	uint &reg0, uint &reg1,
	size_t smem_addr //Shared memory address
)
{
	asm volatile(
		"ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" 
		: "=r"(reg0), "=r"(reg1) 
		: "l"(smem_addr)
	);
}
__forceinline__ __device__ void load_matrix_x1(
	uint &reg,
	size_t smem_addr//Shared memory address
)
{
	asm volatile(
		"ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" 
		: "=r"(reg) 
		: "l"(smem_addr)
	);
}


__forceinline__ __device__ void mma_16x8x8_fp16(
	uint &regD0, uint &regD1,
	uint regA0, uint regA1,
	uint regB,
	uint regC0, uint regC1
)
{
	asm volatile( \
		"mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16" 
		"{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n" 
		:"=r"(regD0), "=r"(regD1) : 
		"r"(regA0), "r"(regA1), "r"(regB), "r"(regC0), "r"(regC1)
	);
} 

__forceinline__ __device__ void mma16x16_fp16_output(                       // fp 16 output
        const __half (*A)[8],   // vg_shared  : [16][8]   row-major
        const __half (*B)[8],   // vp_shared  : [256][8]  col-major
        __half       (*C)[264],   // power_sh   : [16][256] row-major
		int lane,
		int gid,
		int tid4,
		int VG_OFFSET,
        int vp_col_base)        // 0,32,64,96 ...
{

    const __half* A_tile_ptr = &A[lane + VG_OFFSET][0];
    const __half* B_tile_ptr = &B[lane + vp_col_base][0];


    uint32_t RA0, RA1;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(RA0), "=r"(RA1)
        : "l"(__cvta_generic_to_shared(A_tile_ptr)));


    uint32_t RB0;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
        : "=r"(RB0)
        : "l"(__cvta_generic_to_shared(B_tile_ptr)));

    //---------------- Tensor Core ----------------
    uint32_t RC0 = 0, RC1 = 0;


    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3}, "
        "{%4}, "
        "{%5, %6};\n"
        : "=r"(RC0), "=r"(RC1)
        : "r"(RA0), "r"(RA1),
          "r"(RB0), 
          "r"(RC0), "r"(RC1)
    );

    int col0 = tid4 * 2;
    int col1 = col0 + 1;
    int row0 = gid;
    int row1 = gid + 8;

    *(uint32_t*)&C[row0][vp_col_base + col0] = RC0;
    *(uint32_t*)&C[row1][vp_col_base + col0] = RC1;

}
// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* rgb_opacity,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
		rgb_opacity[idx] = { result.x, result.y, result.z, opacities[idx] };
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ rgb_opacity,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}


// Main rasterization method with precomputed power matrix using Tensor Core.
// Each block processes one tile, each thread handles one pixel.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_gemm(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ rgb_opacity,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and coordinate system
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
    int pixel_id = block.thread_rank();

	// Check if the pixel is within image bounds
	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
    bool warp_done = (__ballot_sync(~0, done) == (~0));

	// Query Gaussian range for this specific tile
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);      // size
    int toDo = range.y - range.x;

    float2 tile_center;    // Tile center coordinates in pixel space

	// Shared memory buffers for double buffering and Tensor Core operands
	__shared__ int collected_id[2][BLOCK_SIZE];
	__shared__ float4 collected_rgb_opacity[2][BLOCK_SIZE];

	// collected_vg stores Gaussian/Pixel parameters for MMA [Gaussian/Pixel, VectorComponents]
	__shared__ __half collected_vg[2][BLOCK_SIZE][WARP_LEN];   
	// power_matrix stores intermediate dot products computed by Tensor Cores
	__shared__ __half power_matrix[BATCH_SIZE][BLOCK_SIZE + 8]; 

	float T = 1.0f;
	float C[CHANNELS] = { 0 };
	uint32_t stage = 0;

    int warp_id = pixel_id / 32;
    int lane_id = pixel_id & 31;

    // MMA matrix layout mapping
    const int gid    = lane_id >> 2;   // Group ID (0-7) within warp
    const int tid4   = lane_id &  3;   // Thread index (0-3) within group

	const int col0 = tid4 * 2;
	const int row0 = gid;
	const int row1 = gid + 8;

	// Initial prefetch of Gaussian IDs for the first batch
	if (range.x + pixel_id < range.y)
	{
		__pipeline_memcpy_async(&collected_id[1][pixel_id], 
			reinterpret_cast<const int*>(&point_list[range.x + pixel_id]), sizeof(int));
		__pipeline_commit();
	}

	tile_center = make_float2(block.group_index().x * BLOCK_X + (BLOCK_X - 1) * 0.5f, block.group_index().y * BLOCK_Y + (BLOCK_Y - 1) * 0.5f);

    int px = pixel_id % BLOCK_X;
    int py = pixel_id / BLOCK_Y;

    float dx = (BLOCK_X - 1) * 0.5f - px;
    float dy = (BLOCK_Y - 1) * 0.5f - py;

	// Prepare pixel-specific vectors for Tensor Core dot product calculation
	// These values represent terms in the Gaussian power expression relative to tile center
    collected_vg[0][pixel_id][0] = __float2half(-0.5f);
    collected_vg[0][pixel_id][1] = __float2half(-0.5f * dx * dx);
    collected_vg[0][pixel_id][2] = __float2half(dx);
    collected_vg[0][pixel_id][3] = __float2half(-0.5f * dy * dy);
    collected_vg[0][pixel_id][4] = __float2half(dy);
    collected_vg[0][pixel_id][5] = __float2half(dx * dy);
    collected_vg[0][pixel_id][6] = __float2half(0);
    collected_vg[0][pixel_id][7] = __float2half(0);

    uint32_t vp_reg[4];
    const __half* B_tile_ptr = &collected_vg[0][lane_id + warp_id * 32][0];

	__syncwarp();
	// Load pixel vectors into registers using ldmatrix for MMA
    load_matrix_x4(
        vp_reg[0], vp_reg[1], vp_reg[2], vp_reg[3],
        __cvta_generic_to_shared(B_tile_ptr)
    );
	if (range.x + pixel_id < range.y)
	{
		__pipeline_wait_prior(0);  // Ensure ID prefetch is done

		int id = collected_id[1][block.thread_rank()];

		// Cascade async copies for the next batch (double buffering)
		if (range.x + BLOCK_SIZE + pixel_id < range.y)
		{
			__pipeline_memcpy_async(&collected_id[0][block.thread_rank()], reinterpret_cast<const int*>(&point_list[range.x + BLOCK_SIZE + block.thread_rank()]), sizeof(int));
			__pipeline_commit();
		}
		__pipeline_memcpy_async(&collected_rgb_opacity[0][pixel_id], &rgb_opacity[id], sizeof(float4));
		__pipeline_commit();

		// Compute Gaussian-specific transformation terms for MMA
		float2 center = points_xy_image[id];       
		float4 con_o = conic_opacity[id];

		float d0x = center.x - tile_center.x;
		float d0y = center.y - tile_center.y;

		collected_vg[1][pixel_id][0] = __float2half(con_o.x * d0x * d0x + con_o.z * d0y * d0y + 2.f * con_o.y * d0x * d0y);
		collected_vg[1][pixel_id][1] = __float2half(con_o.x);
		collected_vg[1][pixel_id][2] = __float2half(-(con_o.x*d0x + con_o.y*d0y));
		collected_vg[1][pixel_id][3] = __float2half(con_o.z);
		collected_vg[1][pixel_id][4] = __float2half(-(con_o.z*d0y + con_o.y*d0x));
		collected_vg[1][pixel_id][5] = __float2half(-con_o.y);
		collected_vg[1][pixel_id][6] = __float2half(0.f);
		collected_vg[1][pixel_id][7] = __float2half(0.f);
	}

    // Main loop iterating over batches of Gaussians
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

		// Asynchronous pipeline: Load next Gaussian data while processing current batch
        int progress = (i+1) * BLOCK_SIZE + pixel_id;
        if (range.x + progress < range.y) {
			__pipeline_wait_prior(0); 

			if (range.x + BLOCK_SIZE + progress < range.y) {
				__pipeline_memcpy_async(&collected_id[stage^1][pixel_id], reinterpret_cast<const int*>(&point_list[range.x + BLOCK_SIZE + progress]), sizeof(int));
				__pipeline_commit();
			}

			int id = collected_id[stage][pixel_id];

			__pipeline_memcpy_async(&collected_rgb_opacity[stage^1][pixel_id], &rgb_opacity[id], sizeof(float4));
			__pipeline_commit();
			
			float2 center = points_xy_image[id];       
			float4 con_o = conic_opacity[id];

			float d0x = center.x - tile_center.x;
			float d0y = center.y - tile_center.y;

			__half* dst = collected_vg[stage][pixel_id];

			dst[0] = __float2half(con_o.x * d0x * d0x + con_o.z * d0y * d0y + 2.f * con_o.y * d0x * d0y);
			dst[1] = __float2half(con_o.x);
			dst[2] = __float2half(-(con_o.x*d0x + con_o.y*d0y));
			dst[3] = __float2half(con_o.z);
			dst[4] = __float2half(-(con_o.z*d0y + con_o.y*d0x));
			dst[5] = __float2half(-con_o.y);
        }
		
		// Inner loop: Process Gaussians in chunks of BATCH_SIZE (16) using Tensor Cores
        for (int m = 0; !warp_done && m < min(BLOCK_SIZE, toDo); m += BATCH_SIZE)
        {
			const __half* A_tile_ptr = &collected_vg[stage^1][lane_id + m][0];

			// Load Gaussian coefficient matrix A into registers
			uint32_t vg_reg[2];
			load_matrix_x2(vg_reg[0], vg_reg[1],__cvta_generic_to_shared(A_tile_ptr));

			// Execute MMA operations to compute Gaussian power terms (dot products between Gaussians and pixels)
			uint32_t RC[2] = {0, 0};
            mma_16x8x8_fp16(RC[0], RC[1], vg_reg[0], vg_reg[1], vp_reg[0], RC[0], RC[1]);
			*(uint32_t*)&power_matrix[row0][warp_id * 32 + col0] = RC[0];
			*(uint32_t*)&power_matrix[row1][warp_id * 32 + col0] = RC[1];

			RC[0] = RC[1] = 0;
            mma_16x8x8_fp16(RC[0], RC[1], vg_reg[0], vg_reg[1], vp_reg[1], RC[0], RC[1]);
			*(uint32_t*)&power_matrix[row0][warp_id * 32 + 8 + col0] = RC[0];
			*(uint32_t*)&power_matrix[row1][warp_id * 32 + 8 + col0] = RC[1];

			RC[0] = RC[1] = 0;
            mma_16x8x8_fp16(RC[0], RC[1], vg_reg[0], vg_reg[1], vp_reg[2], RC[0], RC[1]);
			*(uint32_t*)&power_matrix[row0][warp_id * 32 + 16 + col0] = RC[0];
			*(uint32_t*)&power_matrix[row1][warp_id * 32 + 16 + col0] = RC[1];

			RC[0] = RC[1] = 0;
            mma_16x8x8_fp16(RC[0], RC[1], vg_reg[0], vg_reg[1], vp_reg[3], RC[0], RC[1]);
			*(uint32_t*)&power_matrix[row0][warp_id * 32 + 24 + col0] = RC[0];
			*(uint32_t*)&power_matrix[row1][warp_id * 32 + 24 + col0] = RC[1];

            // Perform alpha blending based on the computed Gaussian powers
            #pragma unroll
            for (int j = 0; j < BATCH_SIZE; j++)
            {
				__half power_h = power_matrix[j][pixel_id];
				// Skip if the pixel is outside the Gaussian influence
				if (__hgt(power_h, __float2half(0.0f))) continue; 
				float power = __half2float(power_h);
				float exp_term = fast_exp2_approx(power * log2e);
				float4 rgb_opacity_array = collected_rgb_opacity[stage][j + m];

                // Integration and accumulation (Eq. 2 & 3 from 3DGS paper)
				float alpha = fminf(0.99f, rgb_opacity_array.w * exp_term);

                if (alpha < 1.0f / 255.0f)
                    continue;
                float test_T = T * (1 - alpha);
                if (test_T < 0.0001f) // Early exit if opacity saturates
                {
                    done = true;
                    continue;
                }

                C[0] += rgb_opacity_array.x * alpha * T;
                C[1] += rgb_opacity_array.y * alpha * T;
                C[2] += rgb_opacity_array.z * alpha * T;

                T = test_T;
            }
			// Check if all threads in the warp are done
			if(__ballot_sync(~0, done) == (~0))
				warp_done = true;
        }
		stage ^= 1; // Flip double buffer index
    }

	// Write final pixel colors to output buffer
    if (inside)
    {
        #pragma unroll
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
    }
}



void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* rgb_opacity,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	bool use_gemm = true;

	if (use_gemm) {
		renderCUDA_gemm<NUM_CHANNELS> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means2D,
			colors,
			rgb_opacity,
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			out_color,
			depths, 
			depth);
	} else {
		renderCUDA<NUM_CHANNELS> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means2D,
			colors,
			rgb_opacity,
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			out_color,
			depths, 
			depth);
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* rgb_opacity,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		rgb_opacity,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
