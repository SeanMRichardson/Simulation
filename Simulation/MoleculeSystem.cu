#pragma once

#include "Molecule.h"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_gl.h>
#include <helper_math.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "MoleculeSystem.cuh"

// the parameters used to control the simulation
__constant__ SimulationParameters params;


extern "C"
{
	// copy parameters to constant memory
	void setParameters(SimulationParameters *hostParams)
	{	
		checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimulationParameters)));
	}

	// Round a/b to nearest higher integer value
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

// calculate the barycentric value for a triangle at a particular position in the triangle
__device__ float BaryCentric(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec2 pos) {
	float det = (p2.z - p3.z) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.z - p3.z);
	float l1 = ((p2.z - p3.z) * (pos.x - p3.x) + (p3.x - p2.x) * (pos.y - p3.z)) / det;
	float l2 = ((p3.z - p1.z) * (pos.x - p3.x) + (p1.x - p3.x) * (pos.y - p3.z)) / det;
	float l3 = 1.0f - l1 - l2;
	return l1 * p1.y + l2 * p2.y + l3 * p3.y;
}

// Get the height of the vertex at the specified coordinates
// because the particle may not be over a specific height map vertex we use
// a bary centric average to estimate the height 
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

// kernel function to update the velocity for a particle in a single time step
__global__
void integrate_velocity_kernel(glm::vec3* pos, glm::vec3*vel, glm::vec3* accel, GLuint numberOfParticles, float deltaTime, glm::vec3* vertices)
{
	// get particle index
	GLuint index = blockIdx.x*blockDim.x + threadIdx.x;

	// make sure particle valid
	if (index >= numberOfParticles)
	{
		return;
	}

	// get original position and velocity
	glm::vec3 pPos = pos[index];
	glm::vec3 pVel = vel[index];
	glm::vec3 pAcc = accel[index];

	// update velocity based on acceletation
	pVel.x += pAcc.x * deltaTime;
	pVel.y += pAcc.y * deltaTime;
	pVel.z += pAcc.z * deltaTime;
	pVel *= params.globalDamping;

	// update position based on new velocity
	pPos += pVel * deltaTime;

	// check for collision with "walls"
	if (pPos.x > (MESH_WIDTH - 1) - params.particleRadius)
	{
		pPos.x = (MESH_WIDTH - 1) - params.particleRadius ;
		pVel.x *= params.boundaryDamping;
	}

	if (pPos.x <= 0.0f + params.particleRadius)
	{
		pPos.x = 0.0f + params.particleRadius;
		pVel.x *= params.boundaryDamping;
	}

	if (pPos.z > (MESH_WIDTH - 1) - params.particleRadius)
	{
		pPos.z = (MESH_WIDTH - 1) - params.particleRadius;
		pVel.z *= params.boundaryDamping;
	}

	if (pPos.z <= 0.0f + params.particleRadius)
	{
		pPos.z = 0.0f + params.particleRadius;
		pVel.z *= params.boundaryDamping;
	}

	// make sure particles don't bounce out of top
	if (pPos.y > (MESH_WIDTH - 1) - params.particleRadius)
	{
		pPos.y = (MESH_WIDTH - 1) - params.particleRadius;
		pVel.y *= params.boundaryDamping;
	}

	// get the height at the particle position
	float height = GetVertexHeight(vertices, pPos.z, pPos.x);

	// check for collision with height map
	if (pPos.y < height + params.particleRadius)
	{
		pPos.y = height + params.particleRadius;
		pVel.y *= params.boundaryDamping;
	}

	pos[index] = pPos;
	vel[index] = pVel;
}

//  get the grid position
__device__ glm::ivec3 getGridPosition(glm::vec3 p)
{
	glm::ivec3 gridPos;
	gridPos.x = floor((p.x) / params.cellSize.x);
	gridPos.y = floor((p.y) / params.cellSize.y);
	gridPos.z = floor((p.z) / params.cellSize.z);
	return gridPos;
}

// calculate a hash value from the position in the grid
__device__ GLuint calculateGridHash(glm::ivec3 gridPos)
{
	gridPos.x = gridPos.x & ((int)params.gridSize.x - 1);
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
	glm::ivec3 gridPos = getGridPosition(p);

	GLuint hash = calculateGridHash(gridPos);

	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorder_data_and_find_cell_start_kernel(uint   *cellStart,	GLuint   *cellEnd, glm::vec3 *sortedPos, glm::vec3 *sortedVel, GLuint *gridParticleHash, GLuint *gridParticleIndex, glm::vec3 *oldPos,	glm::vec3 *oldVel, GLuint numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	
	// get the particle index
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

		// use the sorted index to reorder the pos and vel data
		GLuint sortedIndex = gridParticleIndex[index];
		glm::vec3 pos = FETCH(oldPos, sortedIndex);
		glm::vec3 vel = FETCH(oldVel, sortedIndex);

		sortedPos[index] = pos;
		sortedVel[index] = vel;
	}
}

// resolve collision between two spheres
__device__
glm::vec3 collideSpheres(glm::vec3 posA, glm::vec3 posB, glm::vec3 velA, glm::vec3 velB, float radiusA, float radiusB, float attraction)
{
	// calculate relative position
	glm::vec3 relPos = posB - posA;

	// calcuate distance between spheres
	float d = length(relPos);
	float collisionDistance = radiusA + radiusB;

	// initialise force
	glm::vec3 force = glm::vec3(0.0f);

	// if colliding
	if (d < collisionDistance)
	{
		// get the normal
		glm::vec3 norm = relPos / d;

		// relative velocity
		glm::vec3 relVel = velB - velA;

		// relative tangential velocity
		glm::vec3 tanVel = relVel - (dot(relVel, norm) * norm);

		// apply spring force
		force = -params.spring*(collisionDistance - d) * norm;
		
		// apply damping force
		force += params.damping*relVel;
		
		// apply shear force
		force += params.shear*tanVel;
		
		// apply attraction force
		force += attraction*relPos;
	}

	return force;
}

// collide a particle against all other particles in a given cell
__device__
glm::vec3 collideCell(glm::ivec3 gridPos,GLuint index, glm::vec3 pos, glm::vec3 vel, glm::vec3 *oldPos,	glm::vec3 *oldVel, GLuint *cellStart,GLuint *cellEnd)
{
	// ge tthe hash value
	GLuint gridHash = calculateGridHash(gridPos);

	// get start index for particles in this grid cell
	GLuint startIndex = FETCH(cellStart, gridHash);

	// initialise force
	glm::vec3 force = glm::vec3(0.0f);

	// if cell is valid
	if (startIndex != 0xffffffff)
	{
		// ge tthe end index
		GLuint endIndex = FETCH(cellEnd, gridHash);

		// loop through each particle in the grid cell
		for (GLuint j = startIndex; j < endIndex; j++)
		{
			if (j != index)  // don't check for collision with self
			{
				// get the position and velocity data for particle under test
				glm::vec3 pos2 = glm::vec3(FETCH(oldPos, j));
				glm::vec3 vel2 = glm::vec3(FETCH(oldVel, j));

				// add any forces based on collision between particles
				force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
			}
		}
	}
	return force;
}

// handle collision for each particle
__global__
void collision_kernel(glm::vec3 *newVel, glm::vec3 *oldPos,	glm::vec3 *oldVel, GLuint   *gridParticleIndex,	GLuint   *cellStart, GLuint   *cellEnd,	GLuint    numParticles)
{
	// ge tthe particle index
	GLuint index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// make sure were dealing with a valid particle
	if (index >= numParticles)
	{
		return;
	}

	// read particle data from sorted arrays
	glm::vec3 pos = glm::vec3(FETCH(oldPos, index));
	glm::vec3 vel = glm::vec3(FETCH(oldVel, index));

	// get the grid position
	glm::ivec3 gridPos = getGridPosition(pos);

	// initialise force
	glm::vec3 force = glm::vec3(0.0f);

	// loop through all neighbouring cells
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				// add force based on collision with particlar neighbour
				glm::ivec3 neighbourPos = gridPos + glm::ivec3(x, y, z);
				force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
			}
		}
	}

	// write new velocity back to original unsorted location
	GLuint originalIndex = gridParticleIndex[index];
	newVel[originalIndex] = glm::vec3(vel + force);
}

// calculate a hash value for each particle
extern "C" void calculateHash(GLuint *gridParticleHash, GLuint *gridParticleIndex, glm::vec3 *pos, int numParticles)
{
	// calculate cuda config
	GLuint numThreads, numBlocks;
	computeGridSize(numParticles, 512, numBlocks, numThreads);

	// execute the kernel
	calculate_hash_kernel << < numBlocks, numThreads >> > (gridParticleHash, gridParticleIndex, (glm::vec3 *)pos, numParticles);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");
}

// integrate the velocity for each particle
extern "C" void integrateVelocity(glm::vec3* pos, glm::vec3*vel, glm::vec3* accel, GLuint numberOfParticles, float deltaTime, glm::vec3* vertices)
{
	// check we have some particles
	if (numberOfParticles == 0)
	{
		return;
	}

	// calcualte cuda config
	GLuint threads;
	GLuint blocks;
	computeGridSize(numberOfParticles, 512, blocks, threads);

	// execute kernel
	integrate_velocity_kernel << < blocks, threads >> >(pos, vel, accel, numberOfParticles, deltaTime, vertices);
	
	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");
}

// sort the array of particle data using fast sort based on hash key
extern "C" void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
		thrust::device_ptr<uint>(dGridParticleHash + numParticles),
		thrust::device_ptr<uint>(dGridParticleIndex));
}

extern "C" void reorderDataAndFindCellStart(GLuint *cellStart, GLuint *cellEnd, glm::vec3 *sortedPos, glm::vec3 *sortedVel, GLuint *gridParticleHash, GLuint *gridParticleIndex, glm::vec3 *oldPos, glm::vec3 *oldVel, GLuint numParticles, GLuint numCells)
{
	// calculate the cuda config
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 512, numBlocks, numThreads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

	// execute the kernel
	uint smemSize = sizeof(uint)*(numThreads + 1);
	reorder_data_and_find_cell_start_kernel << < numBlocks, numThreads, smemSize >> > (cellStart, cellEnd, (glm::vec3 *)sortedPos, (glm::vec3 *)sortedVel, gridParticleHash, gridParticleIndex, (glm::vec3 *)oldPos, (glm::vec3 *)oldVel, numParticles);

	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

}

// handle particle collision
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

// calcuate the density component for a particular neighbour
__device__
float computeCellDensity(GLuint index, glm::vec3 neighbor, glm::vec3 *dPos, GLuint *dHash, GLuint *dIndex, GLuint *dStart, GLuint *dEnd, GLuint numParticles, GLuint numberOfCells)
{
	// initialise the total density to zero
	float totalCellDensity = 0.0f;

	// calculate the grid hash value for the neighbouring particle
	GLuint gridHash = calculateGridHash(glm::vec3(neighbor.x, neighbor.y, neighbor.z));

	// if neighbour not in grid
	if (gridHash == 0xffffffff)
	{
		return totalCellDensity;
	}

	// set up poly6 equation used in calulation of density (based on Mueller paper)
	float mass = params.mass;
	float l = params.smoothingRadius;
	float l2 = l*l;
	float l4 = l2*l2;
	float l8 = l4*l4;
	float poly6 = (4.0f*mass) / glm::pi<float>()*l8;

	glm::vec3 neighbouringParticle; // the neighbouring particle
	glm::vec3 r; // the relative position between particle and its neighbour
	float r2; // r squared

	GLuint neighbor_index; // the index position of the neighbour particle in the data array
	GLuint start_index = dStart[gridHash]; // the index position of the first particle in the cell

	// if we have a valid start index
	if (start_index != 0xffffffff)
	{
		// get the end index
		GLuint end_index = dEnd[gridHash];

		// lop through all the particles in the grid cell
		for (GLuint count_index = start_index; count_index<end_index; count_index++)
		{
			// get the neighbour we are interested in
			neighbor_index = dIndex[count_index];
			neighbouringParticle = dPos[neighbor_index];

			// calculate the relative position between particle and neighbour
			r = neighbouringParticle - dPos[index];
			r2 = r.x*r.x + r.y*r.y + r.z*r.z;

			// disregard if particle too close to neighbour
			if (r2 < SMALL || r2 >= l2)
			{
				continue;
			}

			// add the density to running total
			totalCellDensity += poly6 * pow(l2 - r2, 3);
		}
	}
	return totalCellDensity;
}

// calcuate the "smoothed" density for a particle and its neighbours
__global__
void compute_density_kernel(glm::vec3 *dPos, float *dDensity, float *dPressure, GLuint *dHash, GLuint *dIndex, GLuint *dStart, GLuint *dEnd, GLuint numParticles, GLuint numberOfCells)
{
	// get the particle index
	GLuint index = blockIdx.x*blockDim.x + threadIdx.x;

	// make sure index is valid
	if (index >= numParticles)
	{
		return;
	}

	// initialise the total calculated density
	float totalDensity = 0;

	// get the grid position that the particle resides in
	glm::vec3 cellPos = getGridPosition(dPos[index]);

	// calculate the "smoothed" density based on the neighbouring particles
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				// add the density attributed to the specific neighbour
				glm::vec3 neighbor_pos = cellPos + glm::vec3(x, y, z);
				totalDensity += computeCellDensity(index, neighbor_pos, dPos, dHash, dIndex, dStart, dEnd, numParticles, numberOfCells);
			}
		}
	}

	// set up the poly6 equation used in calculating the density
	float mass = params.mass;
	float l = params.smoothingRadius;
	float l2 = l*l;
	float l4 = l2*l2;
	float l8 = l4*l4;
	float poly6 = (4.0f*mass) / glm::pi<float>()*l8;

	// calculate the density for the target particle and apply it the total from the neighbours
	totalDensity += poly6 * pow(l2,3);
	dDensity[index] = totalDensity;

	// calculate the particle pressure based on the density
	float k = params.gasConstant;
	dPressure[index] = (dDensity[index] - params.restDensity) * k;
}

// calculate the forces acting on a particle with respect to a neighbouring particle
__device__
glm::vec3 computeCellForce(GLuint index, glm::ivec3 neighbor, glm::vec3* dPos, glm::vec3* dVelocity, float* dDensity, float* dPressure, glm::vec3* dAcceleration, GLuint *dHash, GLuint *dIndex, GLuint *dStart, GLuint *dEnd, GLuint numParticles, GLuint numberOfCells)
{
	// initialise total force to zero
	glm::vec3 totalCellForce = glm::vec3(0.0f);

	// calculate the grid hash value for the neighbouring particle
	GLuint gridHash = calculateGridHash(neighbor);

	// if neighbour not in grid
	if (gridHash == 0xffffffff)
	{
		return totalCellForce;
	}
	
	// set up spiky and viscosity equations used in calulation of forces (base on Mueller paper)
	float l = params.smoothingRadius;
	float l2 = l*l;
	float l5 = l2*l2*l;
	float spiky = 10 / (glm::pi<float>()*l5);
	float viscosity = 10 / (9 * glm::pi<float>()*l5);
	float viscosityCoefficient;
	
	glm::vec3 neighbouringParticle; // the negihbouring particle being examined
	glm::vec3 r; // the relative position between particle and its neighbour

	GLuint neighbourIndex; // the index position of the neighbour particle in the data array
	GLuint startIndex = dStart[gridHash]; // the index position of the first particle in the cell
	
	// if we have a valid start index
	if (startIndex != 0xffffffff)
	{
		// get the end index
		GLuint endIndex = dEnd[gridHash];

		// loop through all the particles in the grid cell
		for (GLuint countIndex = startIndex; countIndex < endIndex; countIndex++)
		{
			// get the neighbour we are interested in
			neighbourIndex = dIndex[countIndex];
			neighbouringParticle = dPos[neighbourIndex];

			// calculate the relative position between particle and neighbour
			r = dPos[index] - neighbouringParticle; 

			// distance between the particle and neighbour
			double dist = glm::length(r);			

			// if they are separated
			if (dist > 0.0)
			{
				// normalise r
				r /= dist;  

				// calculate acceleration due to pressure
				float diff = (l - dist);
				float pterm = (dPressure[index] + dPressure[neighbourIndex]) / (2.0f * dDensity[index] * dDensity[neighbourIndex]);

				// add to total force
				totalCellForce -= (float)(pterm*spiky*diff*diff)*r;

				// calculate acceleration due to viscosity
				float e = viscosity * diff;
				glm::vec3 velDifference = dVelocity[neighbourIndex] - dVelocity[index];
				
				// add to total force
				totalCellForce += (float)(viscosityCoefficient*(1 / dDensity[neighbourIndex]) * e)*velDifference;				
			}
		}
	}	
	return totalCellForce;
}

// calculate the forces acting on a particle in the uniform grid
// For each particle position use grid to iterate over neighboring particles and compute force
__global__
void compute_force_kernel(glm::vec3* dPos, glm::vec3* dVelocity, float* dDensity, float* dPressure, glm::vec3* dAcceleration, GLuint *dHash, GLuint *dIndex, GLuint *dStart, GLuint *dEnd, GLuint numParticles, GLuint numberOfCells)
{
	// get the particle index
	GLuint index = blockIdx.x*blockDim.x + threadIdx.x;

	// make sure we have a valid particle to work with
	if (index >= numParticles)
	{
		return;
	}

	// initialise force to zero
	glm::vec3 totalForce = glm::vec3(0.0f);

	// find out which grid position the particle resides in
	glm::ivec3 cellPosition = getGridPosition(dPos[index]);

	// look at all cells surrounding the specific particle
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				// add component of force attributes to neighbour
				glm::ivec3 neighbor_pos = cellPosition + glm::ivec3(x, y, z);
				totalForce += computeCellForce(index, neighbor_pos, dPos, dVelocity, dDensity, dPressure, dAcceleration, dHash, dIndex, dStart, dEnd, numParticles, numberOfCells);
			}
		}
	}

	// add acceleration due to gravity
	totalForce.x += params.gravity.x;
	totalForce.y += params.gravity.y;
	totalForce.z += params.gravity.z;

	// set acceleration data
	dAcceleration[index] = totalForce;
}

// execute the SPH algorithm for the set of particles passed in
extern "C" void computeSPH(glm::vec3 *dPos, glm::vec3* dVelocity, float *dDensity, float *dPressure, glm::vec3* dAcceleration, GLuint *dHash, GLuint *dIndex, GLuint *dStart, GLuint *dEnd, GLuint numParticles, GLuint numberOfCells)
{
	// make sure there is some particle data to work with
	if (numParticles == 0)
	{
		return;
	}

	// calculate the cuda configuration required
	GLuint numThreads;
	GLuint numBlocks;
	computeGridSize(numParticles, 512, numBlocks, numThreads);
	
	// call the kernels
	compute_density_kernel <<< numBlocks, numThreads >> >(dPos, dDensity, dPressure, dHash, dIndex, dStart, dEnd, numParticles, numberOfCells);
	compute_force_kernel <<< numBlocks, numThreads >> >(dPos, dVelocity, dDensity, dPressure, dAcceleration, dHash, dIndex, dStart, dEnd, numParticles, numberOfCells);
}