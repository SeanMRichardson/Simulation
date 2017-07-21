#ifndef _SIMPLETEXTURE3D_KERNEL_CU_
#define _SIMPLETEXTURE3D_KERNEL_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_gl.h>
#include <helper_math.h>

//Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b)
{
	return (a + (b - 1)) / b;
}

__global__ void
calculate_vertices_kernel(float3* vertices, GLbyte* data, int width, int height)
{
	GLuint x = (blockIdx.x * blockDim.x) + threadIdx.x;
	GLuint z = (blockIdx.y * blockDim.y) + threadIdx.y;

	int offset = (x * width) + z;

	if (x < width && z < height)
	{
		vertices[offset].x = x;
		vertices[offset].y = abs(data[offset]);
		vertices[offset].z = z;
	}
}

__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}

extern "C" void Add(int* a, int* b, int* c)
{
	add << <1, 1 >> > (a, b, c);
}

extern "C" void CalculateVertices(float3* vertices, GLbyte* data, int width, int height)
{
	dim3 block(4, 4);
	dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y));
	calculate_vertices_kernel << <grid, block >> > (vertices, data, width, height);
}

#endif