#pragma once

#include <random>
#include <vector>
#include <glm/gtc/type_ptr.hpp>

#include "Molecule.h"
#include "Shader.h"
#include "Octree.h"
#include "MoleculeSystem.cuh"

enum ParticleSystemType { BOX };
enum Wall { WALL_LEFT, WALL_RIGHT, WALL_FAR, WALL_NEAR };



class MoleculeSystem
{
public:
	MoleculeSystem(ParticleSystemType pType, int maxParticles, int meshWidth, glm::vec3* vertices, glm::vec3 origin = glm::vec3(0));
	~MoleculeSystem();

	void GenerateMolecules(GLuint numberOfParticles, glm::vec3 origin);
	void AddMolecule(Molecule m);
	void KillMolecule();

	void Update(float deltaTime, glm::vec3* vertices, glm::vec3* normals);
	void Render(glm::mat4 proj, glm::mat4 view);

	float GetVertexHeight(glm::vec3* vertices, float x, float z);
	glm::vec3 GetVertexNormal(glm::vec3* normals, float x, float z);

	void BroadphaseCollisionDetection();
	void NarrowphaseCollisionDetection(glm::vec3* vertices, glm::vec3* normals, float deltaTime);

	bool CheckMoleculeCollisionWithTerrain(Molecule m, glm::vec3 normal, float height = 0.0f);
	bool CheckMoleculeCollisionWithWall(Molecule m, Wall w);
	bool CheckMoleculeMoleculeCollision(Molecule* m1, Molecule* m2);

	void HandleTerrainCollision(std::vector<Molecule>& molecules, glm::vec3 normal, float height, int index);
	void HandleWallCollision(std::vector<Molecule>& molecules, int index);
	void HandleMoleculeCollision(CollisionPair* cp);

	float BarryCentric(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec2 pos);
	void Reset();

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

	glm::vec3 GetWallDirection(Wall w);
	glm::vec3 GetPointOnWall(Wall w);

	GLuint m_numberOfGridCells; // the number of cells in the uniform grid used to hold the particles


	// host data
	glm::vec3 *m_hPosition;  // particle positions
	glm::vec3 *m_hVelocity;  // particle velocities

	// device data
	glm::vec3 *m_dPosition;
	glm::vec3 *m_dVelocity;
	glm::vec3 *m_dSortedPosition;
	glm::vec3 *m_dSortedVelocity;


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

