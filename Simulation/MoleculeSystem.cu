#pragma once

#include "Molecule.h"
#include <glm/glm.hpp>

#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_gl.h>
#include <helper_math.h>

__global__ void /// needs to be renamed as molecules do not need to be generated on GPU
generate_molecule_box_kernel(Molecule* molecules, long maxParticles, glm::vec3 origin)
{
	const unsigned long long int  blockId = blockIdx.x // 1D
		+ blockIdx.y * gridDim.x // 2D
		+ gridDim.x * gridDim.y * blockIdx.z; //3D

}