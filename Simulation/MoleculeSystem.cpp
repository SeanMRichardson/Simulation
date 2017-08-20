#include "stdafx.h"
#include "MoleculeSystem.h"

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_gl_interop.h>

// cuda funnctions
extern "C" void computeSPH(glm::vec3 *dPos, glm::vec3* dVelocity, float *dDensity, float *dPressure, glm::vec3* dAcceleration, GLuint *dHash, GLuint *dIndex, GLuint *dStart, GLuint *dEnd, GLuint numParticles, GLuint numberOfCells);
extern "C" void integrateVelocity(glm::vec3* pos, glm::vec3*vel, glm::vec3* accel,GLuint numberOfParticles, float deltaTime, glm::vec3* vertices);
extern "C" void calculateHash(GLuint *gridParticleHash, GLuint *gridParticleIndex, glm::vec3 *pos, int numParticles);
extern "C" void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);
extern "C" void reorderDataAndFindCellStart(GLuint *cellStart, GLuint *cellEnd, glm::vec3 *sortedPos, glm::vec3 *sortedVel, GLuint *gridParticleHash, GLuint *gridParticleIndex, glm::vec3 *oldPos, glm::vec3 *oldVel, GLuint numParticles, GLuint numCells);
extern "C" void collide(glm::vec3 *newVel, glm::vec3 *sortedPos, glm::vec3 *sortedVel, GLuint  *gridParticleIndex, GLuint *cellStart, GLuint  *cellEnd, GLuint numParticles, GLuint numCells);

MoleculeSystem::MoleculeSystem(int meshWidth, glm::vec3* vertices, glm::vec3 origin) : m_vertices(vertices)
{
	// set the number of particles in the simulation
	m_parameters.numberOfParticles = 16384;

	// grid configuration
	m_numberOfGridCells = GRID_SIZE * GRID_SIZE * GRID_SIZE;
	m_parameters.particleRadius = 2.0f / GRID_SIZE;
	m_parameters.gridSize = make_float3((float)GRID_SIZE, (float)GRID_SIZE, (float)GRID_SIZE);

	// each grid set to the size of a particle
	float cellSize = m_parameters.particleRadius * 2.0f;
	m_parameters.cellSize = make_float3(cellSize, cellSize, cellSize);

	// particle simulation
	m_parameters.spring = 10.0f;
	m_parameters.damping = 0.05f;
	m_parameters.shear = 0.05f;
	m_parameters.attraction = 0.02f;
	m_parameters.boundaryDamping = -0.02f;
	m_parameters.gravity = make_float3(0.0f, -1.0f, 0.0f);
	m_parameters.globalDamping = 1.0f;

	// sph simulation parameters
	m_parameters.mass = 0.5f;
	m_parameters.restDensity = 1000.0f;
	m_parameters.gasConstant = 1.0f;
	m_parameters.viscosityCoefficient = 0.018f;
	m_parameters.smoothingRadius = 1.3f;

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");

	// copy parameters to constant memory
	setParameters(&m_parameters);

	// create opengl shaders
	m_shader = new Shader("Shaders/particleVert.glsl", "Shaders/particleFrag.glsl");
	if (!m_shader->LinkProgram())
	{
		std::cout << "ERROR: linking shader program" << std::endl;
	}

	// generate our initial set of particles
	GenerateMolecules(m_parameters.numberOfParticles, origin);
}

// clean up
MoleculeSystem::~MoleculeSystem()
{
	cudaFree(m_dCellEnd);
	cudaFree(m_dVelocity);
	cudaFree(m_dDensity);
	cudaFree(m_dPressure);
	cudaFree(m_dCellStart);
	cudaFree(m_dGridParticleHash);
	cudaFree(m_dGridParticleIndex);
	cudaFree(m_dSortedPosition);
	cudaFree(m_dSortedVelocity);
	cudaFree(m_dAcceleration);

	delete[] m_hPosition;
	delete[] m_hVelocity;
	delete[] m_hDensity;
	delete[] m_hPressure;
	delete[] m_hCellStart;
	delete[] m_hCellEnd;
	delete[] m_hAcceleration;
}

// Generate a set of particles at the specified origin point
void MoleculeSystem::GenerateMolecules(GLuint numberOfParticles, glm::vec3 origin)
{
	// store the number of particles
	m_numParticles = numberOfParticles;

	// allocate host storage
	m_hPosition = new glm::vec3[m_numParticles];
	m_hVelocity = new glm::vec3[m_numParticles];
	m_hDensity = new float[m_numParticles];
	m_hPressure = new float[m_numParticles];
	m_hAcceleration = new glm::vec3[m_numParticles];

	// initialise the storage
	memset(m_hPosition, 0, m_numParticles *  sizeof(glm::vec3));
	memset(m_hVelocity, 0, m_numParticles *  sizeof(glm::vec3));
	memset(m_hDensity, 0, m_numParticles * sizeof(float));
	memset(m_hPressure, 0, m_numParticles * sizeof(float));
	memset(m_hAcceleration, 0, m_numParticles * sizeof(glm::vec3));

	// initialise cell start and end point storage used in SPH
	m_hCellStart = new GLuint[m_numberOfGridCells];
	memset(m_hCellStart, 0, m_numberOfGridCells * sizeof(GLuint));
	m_hCellEnd = new GLuint[m_numberOfGridCells];
	memset(m_hCellEnd, 0, m_numberOfGridCells * sizeof(GLuint));

	// allocate device data
	unsigned int memSize = sizeof(glm::vec3) * m_numParticles;

	checkCudaErrors(cudaMalloc((void **)&m_dVelocity, memSize));
	checkCudaErrors(cudaMalloc((void **)&m_dDensity, m_numParticles * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&m_dPressure, m_numParticles * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&m_dAcceleration, memSize));
	checkCudaErrors(cudaMalloc((void **)&m_dSortedPosition, memSize));
	checkCudaErrors(cudaMalloc((void **)&m_dSortedVelocity, memSize));
	checkCudaErrors(cudaMalloc((void **)&m_dGridParticleHash, m_numParticles * sizeof(GLuint)));
	checkCudaErrors(cudaMalloc((void **)&m_dGridParticleIndex, m_numParticles * sizeof(GLuint)));
	checkCudaErrors(cudaMalloc((void **)&m_dCellStart, m_numberOfGridCells * sizeof(GLuint)));
	checkCudaErrors(cudaMalloc((void **)&m_dCellEnd, m_numberOfGridCells * sizeof(GLuint)));

	// set up data for opengl
	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_vbo);
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, &m_hPosition[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// initialise the grid with particles
	float jitter = m_parameters.particleRadius*0.01f;
	GLuint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
	GLuint gridSize[3];
	gridSize[0] = gridSize[1] = gridSize[2] = s;
	initGrid(gridSize, m_parameters.particleRadius*2.0f, jitter, m_numParticles, origin);

	// set up the data buffer and regsiter with cuda
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_numParticles * sizeof(glm::vec3), m_hPosition);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_posvbo_resource, m_vbo, cudaGraphicsMapFlagsNone));

	// copy buffers to cuda
	checkCudaErrors(cudaMemcpy((char *)m_dVelocity, m_hVelocity, m_numParticles * sizeof(glm::vec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy((char *)m_dAcceleration, m_hAcceleration, m_numParticles * sizeof(glm::vec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy((char *)m_dDensity, m_hDensity, m_numParticles * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy((char *)m_dPressure, m_hPressure, m_numParticles * sizeof(float), cudaMemcpyHostToDevice));
}

// update the particle system for this time step
void MoleculeSystem::Update(float deltaTime, glm::vec3* vertices, glm::vec3* normals)
{
	// map the data to cuda
	glm::vec3* dPos;
	void *ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_posvbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, m_cuda_posvbo_resource));
	dPos = (glm::vec3*) ptr;
	
	// update constants
	setParameters(&m_parameters);	

	// calculate a hash value for each partile
	calculateHash(m_dGridParticleHash, m_dGridParticleIndex, dPos, m_numParticles);

	// sort particles based on the hash value
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	// reorder particle arrays into sorted order and find start and end of each cell
	reorderDataAndFindCellStart(m_dCellStart, m_dCellEnd, m_dSortedPosition, m_dSortedVelocity, m_dGridParticleHash, m_dGridParticleIndex, dPos, m_dVelocity, m_numParticles, m_numberOfGridCells);
	
	// process collisions
	collide(m_dVelocity, m_dSortedPosition,	m_dSortedVelocity, m_dGridParticleIndex, m_dCellStart, m_dCellEnd, m_numParticles, m_numberOfGridCells);

	// execute the sph algorithm
	computeSPH(dPos, m_dVelocity, m_dDensity, m_dPressure, m_dAcceleration, m_dGridParticleHash, m_dGridParticleIndex, m_dCellStart, m_dCellEnd, m_numParticles, m_numberOfGridCells);
	
	// update positions, velocities etc. for this time step
	integrateVelocity(dPos, m_dVelocity, m_dAcceleration, m_numParticles, deltaTime, vertices);
	
	// release the cuda resourcses
	checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_posvbo_resource, 0));
}

// render the current state
void MoleculeSystem::Render(glm::mat4 proj, glm::mat4 view, int height, float fov)
{
	// bind the data
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

	// set the shader
	glUseProgram(m_shader->GetProgram());
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	// set shader variables
	glUniformMatrix4fv(glGetUniformLocation(m_shader->GetProgram(), "matProjection"), 1, GL_FALSE, glm::value_ptr(proj));
	glUniformMatrix4fv(glGetUniformLocation(m_shader->GetProgram(), "matModelview"), 1, GL_FALSE, glm::value_ptr(view));
	glUniform1f(glGetUniformLocation(m_shader->GetProgram(), "pointScale"), height / tanf(fov*0.5f*glm::pi<float>() / 180.0f));
	glUniform1f(glGetUniformLocation(m_shader->GetProgram(), "pointRadius"), m_parameters.particleRadius);

	// draw each particle as a point
	glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);

	// draw the particles
	glDrawArrays(GL_POINTS, 0, m_numParticles);

	// remove the binding to the data
	glBindVertexArray(0);

	// disable the shader
	glDisable(m_shader->GetProgram());
}


//initialise a grid of particles in the form of a cube
void MoleculeSystem::initGrid(GLuint *size, float spacing, float jitter, GLuint numParticles, glm::vec3 origin)
{
	for (GLuint z = 0; z<size[2]; z++)
	{
		for (GLuint y = 0; y<size[1]; y++)
		{
			for (GLuint x = 0; x<size[0]; x++)
			{
				GLuint i = (z*size[1] * size[0]) + (y*size[0]) + x;

				if (i < numParticles)
				{
					// uniform grid position calculated
					m_hPosition[i] = glm::vec3(origin.x + m_parameters.particleRadius + x * spacing, origin.y + m_parameters.particleRadius + y * spacing, origin.z + m_parameters.particleRadius + z * spacing);
					
					// make sure the particle is stationary with no forces actin on it
					m_hVelocity[i] = glm::vec3(0.0f);
					m_hAcceleration[i] = glm::vec3(0.0f);
					m_hDensity[i] = 0.0f;
					m_hPressure[i] = 0.0f;
				}
			}
		}
	}
}
