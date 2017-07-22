#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <atomic>

class Molecule
{
public:
	int id;
	Molecule();
	Molecule(glm::vec3 position);
	~Molecule();

	void Update(float deltaTime);

	glm::vec3 GetPosition() const { return m_position; }
	glm::vec3 GetVelocity() { return m_velocity; }
	glm::vec3 GetAcceleration() { return m_acceleration; }
	glm::vec3 GetForce() { return m_force; }
	glm::vec3 GetImpulseForce() { return m_impulseForce; }
	float GetInverseMass() { return m_inverseMass; }

	void SetPosition(glm::vec3 position) { m_position = position;  }
	void SetVelocity(glm::vec3 velocity) { m_velocity = velocity; }
	void AddForce(glm::vec3 force) { m_force += force; }
	void AddImpuseForce(glm::vec3 force) { m_impulseForce += force; }
	void SetImpulseForce(glm::vec3 force) { m_impulseForce = force; }


	inline bool	operator==(const Molecule &other) const { return (other.id == id) ? true : false; };
	inline bool	operator!=(const Molecule &other)const { return (other.id == id) ? false : true; };


private:
	glm::vec3 m_position;
	glm::vec3 m_velocity;
	glm::vec3 m_acceleration;
	glm::vec3 m_force;
	glm::vec3 m_impulseForce;

	glm::vec4 m_colour;

	float m_mass;
	float m_inverseMass;
	float m_radius;

protected:
	static std::atomic<int> s_id;
};

