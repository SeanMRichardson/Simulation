#pragma once

#include <vector>
#include <glm/gtc/type_ptr.hpp>

#include "Molecule.h"
#include "Shader.h"

enum ParticleSystemType { BOX };
enum Wall { WALL_LEFT, WALL_RIGHT, WALL_FAR, WALL_NEAR };

#define DENSITY 0.5f

class MoleculeSystem
{
public:
	MoleculeSystem(ParticleSystemType pType, int maxParticles, int meshWidth, glm::vec3 origin = glm::vec3(0));
	~MoleculeSystem();

	void GenerateMolecules(ParticleSystemType pType, glm::vec3 origin);
	void AddMolecule(Molecule m);
	void KillMolecule();

	void Update(float deltaTime, glm::vec3* vertices, glm::vec3* normals);
	void Render(glm::mat4 proj, glm::mat4 view);

	float GetVertexHeight(glm::vec3* vertices, float x, float z);
	glm::vec3 GetVertexNormal(glm::vec3* normals, float x, float z);

	bool CheckMoleculeCollisionWithTerrain(Molecule m, glm::vec3 normal, float height = 0.0f);
	bool CheckMoleculeCollisionWithWall(Molecule m, Wall w);
	bool CheckMoleculeMoleculeCollision(Molecule* m1, Molecule* m2);

	void HandleTerrainCollision(std::vector<Molecule>& molecules, glm::vec3 normal, float height, int index);
	void HandleWallCollision(std::vector<Molecule>& molecules, int index);
	void HandleMoleculeCollision(std::vector<Molecule>& molecules, int index);

	float BarryCentric(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec2 pos);
	void Reset();

private:
	int m_meshWidth;
	float m_dampingFactor = 0.9f;

	long m_numParticles;
	long m_maxParticles;
	std::vector<Molecule> m_molecules;

	Shader* m_shader;

	GLuint m_vao, m_vbo;

	glm::vec3 GetWallDirection(Wall w);
	glm::vec3 GetPointOnWall(Wall w);

};

