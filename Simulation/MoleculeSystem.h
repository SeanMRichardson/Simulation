#pragma once

#include <vector>
#include <glm/gtc/type_ptr.hpp>

#include "Molecule.h"
#include "Shader.h"

enum ParticleSystemType { BOX };

#define DENSITY 1.0f

class MoleculeSystem
{
public:
	MoleculeSystem(ParticleSystemType pType, int maxParticles, int meshWidth, glm::vec3 origin = glm::vec3(0));
	~MoleculeSystem();

	void GenerateMolecules(ParticleSystemType pType, glm::vec3 origin);
	void AddMolecule(Molecule m);
	void KillMolecule();

	void Update(float deltaTime, glm::vec3* vertices);
	void Render(glm::mat4 proj, glm::mat4 view);

	float GetVertexHeight(glm::vec3* vertices, float x, float z, int width);

	void Reset();

private:
	int m_meshWidth;

	long m_numParticles;
	long m_maxParticles;
	std::vector<Molecule> m_molecules;
	
	Shader* m_shader;

	GLuint m_vao, m_vbo;
	
};

