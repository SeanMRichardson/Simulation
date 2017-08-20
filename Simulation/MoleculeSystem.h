#pragma once

#include <random>
#include <vector>
#include <glm/gtc/type_ptr.hpp>

#include "Molecule.h"
#include "Shader.h"
#include "Octree.h"
#include "MoleculeSystem.cuh"

class MoleculeSystem
{
public:
	
	MoleculeSystem(int meshWidth, glm::vec3* vertices, glm::vec3 origin = glm::vec3(0));
	~MoleculeSystem();

	void GenerateMolecules(GLuint numberOfParticles, glm::vec3 origin); // generate a "cube" of molecules at the specific origin point

	void Update(float deltaTime, glm::vec3* vertices, glm::vec3* normals); // update the system
	void Render(glm::mat4 proj, glm::mat4 view, int height, float fov); // render the system

	void initGrid(GLuint *size, float spacing, float jitter, GLuint numParticles, glm::vec3 origin); // initialise the particles in a grid formation

private:

	glm::vec3* m_vertices; // the number of vertices in the heightmap


	Shader* m_shader; // the opengl shader program
	GLuint m_vao, m_vbo; // opengl buffers

	GLuint m_numberOfGridCells; // the number of cells in the uniform grid used to hold the particles

	long m_numParticles; // the number of particles in the simulation

	// host data
	glm::vec3 *m_hPosition;  // particle positions
	glm::vec3 *m_hVelocity;  // particle velocities
	glm::vec3 *m_hAcceleration; // particle accelerations
	float * m_hDensity; // particle densities - used in SPH
	float * m_hPressure; // particle pressures - used in SPH

	// device representation of particle data
	glm::vec3 *m_dPosition;
	glm::vec3 *m_dVelocity;
	glm::vec3 *m_dSortedPosition;
	glm::vec3 *m_dSortedVelocity;
	glm::vec3 *m_dAcceleration;
	float * m_dDensity;
	float * m_dPressure;

	GLuint  *m_hParticleHash; // location of the hash table
	GLuint  *m_hCellStart; // start of particles for a particular cell
	GLuint  *m_hCellEnd; // end of particles for a particluar cell

	// grid data for sorting method
	GLuint  *m_dGridParticleHash; // grid hash value for each particle
	GLuint  *m_dGridParticleIndex;// particle index for each particle
	GLuint  *m_dCellStart;        // index of start of each cell in sorted list
	GLuint  *m_dCellEnd;          // index of end of cell

	SimulationParameters m_parameters; // the set of parameters used to control the simulation

	struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange

};

