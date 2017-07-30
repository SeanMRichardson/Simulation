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

	void GenerateMolecules(GLuint numberOfParticles, glm::vec3 origin);

	void Update(float deltaTime, glm::vec3* vertices, glm::vec3* normals);
	void Render(glm::mat4 proj, glm::mat4 view, int height, float fov);

	void initGrid(GLuint *size, float spacing, float jitter, GLuint numParticles, glm::vec3 origin);

private:
	int m_meshWidth;
	float m_dampingFactor = 0.9f;

	long m_numParticles;
	long m_maxParticles;
	std::vector<Molecule> m_molecules;
	std::vector<CollisionPair> m_BroadphaseCollisionPairs;
	glm::vec3* m_vertices;


	Shader* m_shader;

	GLuint m_vao, m_vbo;

	GLuint m_numberOfGridCells; // the number of cells in the uniform grid used to hold the particles


	// host data
	glm::vec3 *m_hPosition;  // particle positions
	glm::vec3 *m_hVelocity;  // particle velocities
	glm::vec3 *m_hAcceleration;
	float * m_hDensity;
	float * m_hPressure;

	// device data
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

	SimulationParameters m_parameters;

	struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange

};

