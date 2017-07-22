#include "stdafx.h"
#include "Molecule.h"

std::atomic<int> Molecule::s_id;

Molecule::Molecule() :  id(++s_id),
						m_position(glm::vec3(0.0f)), 
						m_velocity(glm::vec3(0.0f)),
						m_colour(glm::vec4(1.0f)),
						m_mass(1.0f),
						m_radius(1.0f),
						m_force(glm::vec3(0, -9.81, 0))
{
	m_inverseMass = 1 / m_mass;
}

Molecule::Molecule(glm::vec3 position) :	m_position(position),
											m_velocity(glm::vec3(0.0f)),
											m_colour(glm::vec4(1.0f)),
											m_mass(1.0f),
											m_radius(1.0f),
											m_force(glm::vec3(0, -9.81, 0))
{
	m_inverseMass = 1 / m_mass;
}

Molecule::~Molecule()
{
}

void Molecule::Update(float deltaTime)
{

}