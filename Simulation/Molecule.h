#pragma once
#include "defines.h"

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <atomic>
#include <math.h>

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
	glm::vec3 GetForce() { return m_force + m_impulseForce; }
	glm::vec3 GetImpulseForce() { return m_impulseForce; }
	glm::vec3 GetContactpoint() { return m_contactPoint; }
	glm::vec3 GetContactNormal() { return m_contactNormal; }

	float GetInverseMass() { return m_inverseMass; }
	float GetRadius() { return m_radius; }
	float GetPenetrationDepth() { return m_penetrationDepth; }


	void SetPosition(glm::vec3 position) { m_position = position; }
	void SetVelocity(glm::vec3 velocity) { m_velocity = velocity; }
	void AddForce(glm::vec3 force) { m_force += force; }
	void AddImpulseForce(glm::vec3 force) { m_impulseForce += force; }
	void SetImpulseForce(glm::vec3 force) { m_impulseForce = force; }

	void SetContactPoint(glm::vec3 contact) { m_contactPoint = contact; }
	void SetContactNormal(glm::vec3 normal) { m_contactNormal = normal; }
	void SetPenetrationDepth(float depth) { m_penetrationDepth = depth; }

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

	glm::vec3 m_contactPoint;
	glm::vec3 m_contactNormal;
	float m_penetrationDepth;

protected:
	static std::atomic<int> s_id;
};

