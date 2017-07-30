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
calculate_vertices_kernel(glm::vec3* vertices, GLbyte* data, int width, int height)
{
	//GLuint x = (blockIdx.x * blockDim.x) + threadIdx.x;
	//GLuint z = (blockIdx.y * blockDim.y) + threadIdx.y;

	GLuint x = blockIdx.x;
	GLuint z = blockIdx.y;

	int offset = (z * height) + x;

	if (x < width && z < height)
	{
		vertices[offset].x = x;
		vertices[offset].y = abs(data[offset]*0.4)-3.6f; //reading in the heights from the data, and applying offsets to make it more readable
		vertices[offset].z = z;
	}
}

__global__ void
calculate_indices_kernel(GLuint* indices, GLint numIndices, int width, int height)
{
	GLuint x = blockIdx.x;
	GLuint y = blockIdx.y;

	int offset = (y * height * 6) + (x * 6);

	if (x < width && y < height)
	{
		long a = (x * (width)) + y;
		long b = ((x + 1) * (width)) + y;
		long c = ((x + 1) * (width)) + (y + 1);
		long d = (x * (width)) + (y + 1);

		indices[offset] = c;
		indices[offset + 1] = b;
		indices[offset + 2] = a;

		indices[offset + 3] = a;
		indices[offset + 4] = d;
		indices[offset + 5] = c;
	}
}

__global__ void
calculate_normals_kernel(glm::vec3* normals, GLuint* indices, glm::vec3* vertices, int width, int height)
{
	GLuint x = blockIdx.x;
	GLuint y = blockIdx.y;

	int offset = (y * height * 3) + (x* 3);

	unsigned int a = indices[offset];
	unsigned int b = indices[offset + 1];
	unsigned int c = indices[offset + 2];

	glm::vec3 normal = cross((vertices[b] - vertices[c]), (vertices[a] - vertices[c]));

	normals[a] += normal;
	normals[b] += normal;
	normals[c] += normal;
}

/*
Process each vertex from the raw file in its own block
Each block needs just one thread
*/
extern "C" void CalculateVertices(glm::vec3* vertices, GLbyte* data, int width, int height)
{
	dim3 grid(width, height);
	dim3 block(1, 1, 1);
	calculate_vertices_kernel << <grid, block >> > (vertices, data, width, height);
}

/*
Each block in the grid represents the space between 4 vertices
Each space is split into 2 triangles
*/
extern "C" void CalculateIndices(GLuint* indices, GLint numIndices, int width, int height)
{

	dim3 block(1, 1, 1);
	dim3 grid(width-1, width-1);
	calculate_indices_kernel << <grid, block >> > (indices, numIndices, width, height);
}

/*
The surface normal of each triangle is processed in its own block/thread
There are 2 triangles per set of 4 vertices so we create a grid with twice the dimensions
*/
extern "C" void CalculateNormals(glm::vec3* normals, GLuint* indices, glm::vec3* vertices, int width, int height)
{
	dim3 block(1, 1, 1);
	dim3 grid(width*2, height*2);
	calculate_normals_kernel << <grid, block >> > (normals, indices, vertices, width, height);
}
#endif