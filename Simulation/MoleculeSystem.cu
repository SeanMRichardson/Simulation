#pragma once

#include "Molecule.h"
#include <glm/glm.hpp>
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_gl.h>
#include <helper_math.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "MoleculeSystem.cuh"

__constant__ SimulationParameters params;


extern "C"
{
	void setParameters(SimulationParameters *hostParams)
	{
		// copy parameters to constant memory
		checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimulationParameters)));
	}

	//Round a / b to nearest higher integer value
	uint iDivUp(uint a, uint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	// compute grid and thread block size for a given number of elements
	void computeGridSize(GLuint n, GLuint blockSize, GLuint &numBlocks, GLuint &numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}

}

__device__ float BaryCentric(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec2 pos) {
	float det = (p2.z - p3.z) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.z - p3.z);
	float l1 = ((p2.z - p3.z) * (pos.x - p3.x) + (p3.x - p2.x) * (pos.y - p3.z)) / det;
	float l2 = ((p3.z - p1.z) * (pos.x - p3.x) + (p1.x - p3.x) * (pos.y - p3.z)) / det;
	float l3 = 1.0f - l1 - l2;
	return l1 * p1.y + l2 * p2.y + l3 * p3.y;
}

__device__ float GetVertexHeight(glm::vec3* vertices, float x, float z)
{
	float gridSquareSize = MESH_WIDTH / (MESH_WIDTH - 1);
	int gridX = floor(x / gridSquareSize);
	int gridZ = floor(z / gridSquareSize);

	if (gridX >= MESH_WIDTH - 1 || gridZ >= MESH_WIDTH - 1 || gridX < 0 || gridZ < 0)
		return 0;

	float xCoord = fmod(x, gridSquareSize) / gridSquareSize;
	float zCoord = fmod(z, gridSquareSize) / gridSquareSize;

	float height;

	if (xCoord <= (1 - zCoord))
	{
		int offset1 = (gridX*MESH_WIDTH) + gridZ;
		int offset2 = ((gridX + 1)*MESH_WIDTH) + gridZ;
		int offset3 = (gridX*MESH_WIDTH) + gridZ + 1;
		height = BaryCentric(glm::vec3(0, vertices[offset1].y, 0), glm::vec3(1, vertices[offset2].y, 0), glm::vec3(0, vertices[offset3].y, 1), glm::vec2(xCoord, zCoord));
	}
	else
	{
		int offset1 = ((gridX + 1)*MESH_WIDTH) + gridZ;
		int offset2 = ((gridX + 1)*MESH_WIDTH) + gridZ + 1;
		int offset3 = (gridX*MESH_WIDTH) + gridZ + 1;
		height = BaryCentric(glm::vec3(1, vertices[offset1].y, 0), glm::vec3(1, vertices[offset2].y, 1), glm::vec3(0, vertices[offset3].y, 1), glm::vec2(xCoord, zCoord));
	}

	return height;
}

__global__
void integrate_velocity_kernel(glm::vec3* pos, glm::vec3*vel, GLuint numberOfParticles, float deltaTime, glm::vec3* vertices)
{
	GLuint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= numberOfParticles)
	{
		return;
	}

	// get original position and velocity
	glm::vec3 pPos = pos[index];
	glm::vec3 pVel = vel[index];

	pVel.x += params.gravity.x * deltaTime;
	pVel.y += params.gravity.y * deltaTime;
	pVel.z += params.gravity.z * deltaTime;
	pVel *= params.globalDamping;

	pPos += pVel * deltaTime;

	if (pPos.x > GRID_SIZE - params.particleRadius)
	{
		pPos.x = GRID_SIZE - params.particleRadius;
		pVel.x *= params.boundaryDamping;
	}

	if (pPos.x < 0 + params.particleRadius)
	{
		pPos.x = 0 + params.particleRadius;
		pVel.x *= params.boundaryDamping;
	}

	if (pPos.y > GRID_SIZE - params.particleRadius)
	{
		pPos.y = GRID_SIZE - params.particleRadius;
		pVel.y *= params.boundaryDamping;
	}

	if (pPos.z > GRID_SIZE - params.particleRadius)
	{
		pPos.z = GRID_SIZE - params.particleRadius;
		pVel.z *= params.boundaryDamping;
	}

	if (pPos.z < 0 + params.particleRadius)
	{
		pPos.z = 0 + params.particleRadius;
		pVel.z *= params.boundaryDamping;
	}

	float height = GetVertexHeight(vertices, pPos.x, pPos.z);

	if (pPos.y < height + params.particleRadius)
	{
		pPos.y = height + params.particleRadius;
		pVel.y *= params.boundaryDamping;
	}

	pos[index] = pPos;
	vel[index] = pVel;
}

// calculate position in uniform grid
__device__ glm::ivec3 calcGridPos(glm::vec3 p)
{
	glm::ivec3 gridPos;
	gridPos.x = floor((p.x) / params.cellSize.x);
	gridPos.y = floor((p.y) / params.cellSize.y);
	gridPos.z = floor((p.z) / params.cellSize.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ GLuint calcGridHash(glm::ivec3 gridPos)
{
	gridPos.x = gridPos.x & ((int)params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & ((int)params.gridSize.y - 1);
	gridPos.z = gridPos.z & ((int)params.gridSize.z - 1);
	return ((gridPos.z* params.gridSize.y)* params.gridSize.x) + (gridPos.y* params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calculate_hash_kernel(GLuint *gridParticleHash, GLuint *gridParticleIndex, glm::vec3 *pos, int numParticles)
{
	GLuint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	glm::vec3 p = pos[index];

	// get address in grid
	glm::ivec3 gridPos = calcGridPos(p);

	GLuint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorder_data_and_find_cell_start_kernel(uint   *cellStart,        // output: cell start index
	GLuint   *cellEnd,          // output: cell end index
	glm::vec3 *sortedPos,        // output: sorted positions
	glm::vec3 *sortedVel,        // output: sorted velocities
	GLuint   *gridParticleHash, // input: sorted grid hashes
	GLuint   *gridParticleIndex,// input: sorted particle indices
	glm::vec3 *oldPos,           // input: sorted position array
	glm::vec3 *oldVel,           // input: sorted velocity array
	GLuint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = (blockIdx.x* blockDim.x) + threadIdx.x;

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	__syncthreads();

	if (index < numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		GLuint sortedIndex = gridParticleIndex[index];
		glm::vec3 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
		glm::vec3 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh

		sortedPos[index] = pos;
		sortedVel[index] = vel;
	}
}

// collide two spheres using DEM method
__device__
glm::vec3 collideSpheres(glm::vec3 posA, glm::vec3 posB,
	glm::vec3 velA, glm::vec3 velB,
	float radiusA, float radiusB,
	float attraction)
{
	// calculate relative position
	glm::vec3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	glm::vec3 force = glm::vec3(0.0f);

	if (dist < collideDist)
	{
		glm::vec3 norm = relPos / dist;

		// relative velocity
		glm::vec3 relVel = velB - velA;

		// relative tangential velocity
		glm::vec3 tanVel = relVel - (dot(relVel, norm) * norm);

		// spring force
		force = -params.spring*(collideDist - dist) * norm;
		// dashpot (damping) force
		force += params.damping*relVel;
		// tangential shear force
		force += params.shear*tanVel;
		// attraction
		force += attraction*relPos;
	}

	return force;
}

// collide a particle against all other particles in a given cell
__device__
glm::vec3 collideCell(glm::ivec3    gridPos,
	GLuint    index,
	glm::vec3  pos,
	glm::vec3  vel,
	glm::vec3 *oldPos,
	glm::vec3 *oldVel,
	GLuint   *cellStart,
	GLuint   *cellEnd)
{
	GLuint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	GLuint startIndex = FETCH(cellStart, gridHash);

	glm::vec3 force = glm::vec3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		GLuint endIndex = FETCH(cellEnd, gridHash);

		for (GLuint j = startIndex; j<endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				glm::vec3 pos2 = glm::vec3(FETCH(oldPos, j));
				glm::vec3 vel2 = glm::vec3(FETCH(oldVel, j));

				// collide two spheres
				force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
			}
		}
	}

	return force;
}

__global__
void collision_kernel(glm::vec3 *newVel,               // output: new velocity
	glm::vec3 *oldPos,               // input: sorted positions
	glm::vec3 *oldVel,               // input: sorted velocities
	GLuint   *gridParticleIndex,    // input: sorted particle indices
	GLuint   *cellStart,
	GLuint   *cellEnd,
	GLuint    numParticles)
{
	GLuint index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	glm::vec3 pos = glm::vec3(FETCH(oldPos, index));
	glm::vec3 vel = glm::vec3(FETCH(oldVel, index));

	// get address in grid
	glm::ivec3 gridPos = calcGridPos(pos);

	// examine neighbouring cells
	glm::vec3 force = glm::vec3(0.0f);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				glm::ivec3 neighbourPos = gridPos + glm::ivec3(x, y, z);
				force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
			}
		}
	}

	// collide with cursor sphere
	//force += collideSpheres(pos, params.colliderPos, vel, glm::vec3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

	// write new velocity back to original unsorted location
	GLuint originalIndex = gridParticleIndex[index];
	newVel[originalIndex] = glm::vec3(vel + force);
}

extern "C" void calculateHash(GLuint *gridParticleHash, GLuint *gridParticleIndex, glm::vec3 *pos, int numParticles)
{
	GLuint numThreads, numBlocks;
	computeGridSize(numParticles, 512, numBlocks, numThreads);

	// execute the kernel
	calculate_hash_kernel << < numBlocks, numThreads >> > (gridParticleHash, gridParticleIndex, (glm::vec3 *)pos, numParticles);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");
}

extern "C" void integrateVelocity(glm::vec3* pos, glm::vec3*vel, GLuint numberOfParticles, float deltaTime, glm::vec3* vertices)
{
	if (numberOfParticles == 0)
	{
		return;
	}

	GLuint threads;
	GLuint blocks;
	computeGridSize(numberOfParticles, 512, blocks, threads);

	integrate_velocity_kernel << < blocks, threads >> >(pos, vel, numberOfParticles, deltaTime, vertices);
	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");
}

extern "C" void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
		thrust::device_ptr<uint>(dGridParticleHash + numParticles),
		thrust::device_ptr<uint>(dGridParticleIndex));
}

extern "C" void reorderDataAndFindCellStart(GLuint *cellStart, GLuint *cellEnd, glm::vec3 *sortedPos, glm::vec3 *sortedVel, GLuint *gridParticleHash, GLuint *gridParticleIndex, glm::vec3 *oldPos, glm::vec3 *oldVel, GLuint numParticles, GLuint numCells)
{
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 512, numBlocks, numThreads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

	uint smemSize = sizeof(uint)*(numThreads + 1);
	reorder_data_and_find_cell_start_kernel << < numBlocks, numThreads, smemSize >> > (cellStart, cellEnd, (glm::vec3 *)sortedPos, (glm::vec3 *)sortedVel, gridParticleHash, gridParticleIndex, (glm::vec3 *)oldPos, (glm::vec3 *)oldVel, numParticles);

	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

}

extern "C" void collide(glm::vec3 *newVel, glm::vec3 *sortedPos, glm::vec3 *sortedVel,	GLuint  *gridParticleIndex, GLuint *cellStart, GLuint  *cellEnd, GLuint numParticles, GLuint numCells)
{
	// thread per particle
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 512, numBlocks, numThreads);

	// execute the kernel
	collision_kernel << < numBlocks, numThreads >> >((glm::vec3 *)newVel, (glm::vec3 *)sortedPos, (glm::vec3 *)sortedVel, gridParticleIndex, cellStart, cellEnd, numParticles);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");
}