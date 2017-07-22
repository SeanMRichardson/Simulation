#include "stdafx.h"
#include "MoleculeSystem.h"


MoleculeSystem::MoleculeSystem(ParticleSystemType pType, int maxParticles, int meshWidth, glm::vec3 origin): m_maxParticles(maxParticles), m_meshWidth(meshWidth)
{
	GenerateMolecules(pType, origin);

	m_shader = new Shader("Shaders/particleVert.glsl", "Shaders/particleFrag.glsl");
	if (!m_shader->LinkProgram())
	{
		std::cout << "ERROR: linking shader program" << std::endl;
	}

	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_vbo);

	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, m_numParticles * sizeof(Molecule), &m_molecules.front(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,sizeof(Molecule), (void*)sizeof(int));
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


MoleculeSystem::~MoleculeSystem()
{
	m_molecules.clear();
}

void MoleculeSystem::GenerateMolecules(ParticleSystemType pType, glm::vec3 origin)
{
	switch (pType)
	{
	case BOX:
		int size = (int)std::cbrt(m_maxParticles);
		m_numParticles = size*size*size;
		for (int z = 0; z < size; z++)
		{
			for (int y = 0; y < size; y++)
			{
				for (int x = 0; x < size; x++)
				{
					AddMolecule(Molecule(glm::vec3(x*DENSITY + origin.x, y*DENSITY + origin.y, z*DENSITY + origin.z)));
				}
			}
		}
		break;
	}
}

void MoleculeSystem::AddMolecule(Molecule m)
{
	m_molecules.push_back(m);
}

void MoleculeSystem::KillMolecule()
{
	m_molecules.erase(m_molecules.begin());
}


void MoleculeSystem::Update(float deltaTime, glm::vec3* vertices)
{
	for(int i = 0; i < m_molecules.size(); i++)
	{
		auto compute_acc = [&](const glm::vec3& pos)
		{
			glm::vec3& acc = (m_molecules[i].GetForce() + m_molecules[i].GetImpulseForce()) * m_molecules[i].GetInverseMass();	// apply external forces to the system	
			return acc;
		};

		// first order integration
		glm::vec3 p1 = m_molecules[i].GetPosition();
		glm::vec3 v1 = m_molecules[i].GetVelocity();
		glm::vec3 a1 = compute_acc(p1);

		m_molecules[i].SetPosition(m_molecules[i].GetPosition() + (v1 * deltaTime));
		m_molecules[i].SetVelocity(m_molecules[i].GetVelocity() + (a1 * deltaTime));

		m_molecules[i].SetImpulseForce(glm::vec3(0));
	}
}

void MoleculeSystem::Render(glm::mat4 proj, glm::mat4 view)
{
	glBindVertexArray(m_vao);

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_numParticles * sizeof(Molecule), &m_molecules[0]);

	glUseProgram(m_shader->GetProgram());
	glUniformMatrix4fv(glGetUniformLocation(m_shader->GetProgram(), "matProjection"), 1, GL_FALSE, glm::value_ptr(proj));
	glUniformMatrix4fv(glGetUniformLocation(m_shader->GetProgram(), "matModelview"), 1, GL_FALSE, glm::value_ptr(view));

	glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	glEnable(GL_PROGRAM_POINT_SIZE);

	glDrawArrays(GL_POINTS, 0, m_numParticles);

	glBindVertexArray(0);
	glDisable(m_shader->GetProgram());
}

float MoleculeSystem::GetVertexHeight(glm::vec3* vertices, float x, float z, int width)
{
	int offset = (x*m_meshWidth) + z;
	return vertices[offset].y;
}


void MoleculeSystem::Reset()
{

}
