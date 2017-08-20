#ifndef MOLECULE_SYSTEM_H
#define MOLECULE_SYSTEM_H

#include "glm/glm.hpp"
#include "GL/glew.h"

#include <vector_types.h>
#include <helper_math.h>

// the set of parameters used for the simulation, shared between device and host .... so both can access the values
struct SimulationParameters
{
	int numberOfParticles; // number of particles in the simulation

	float mass; // the mass of each particle
	float3 gravity; // the system grivity
	float globalDamping; // a multiplcation factor used to make the simuation more releastic 1.0f - no damping, 0.5f all velocities with be halved
	float particleRadius; // the size of each particle
	

	float3 gridSize; // the dimensions of the uniform grid
	GLuint numCells; // the number of cells in the uniform grid
	float3 cellSize; // the size of each cell in the uniform grid

	GLuint maxParticlesPerCell; // the maximum number of particles that can fit in a cell of the uniform grid

	float smoothingRadius; // used by SPH to determine how close other partciles need to be to affect each other
	
	float restDensity; // the density of a particle, used in SPH
	float gasConstant; // used in SPH in calulating the pressures acting on a particle
	float viscosityCoefficient; // how viscous the fluid is, used by SPH in calculating the forces on a particle

	float spring; // the spring force applied when particles collide
	float damping; // the damping that occurs between particles on a collision
	float shear; // the shear force applied when particles collide
	float attraction; // the attraction force between particles
	float boundaryDamping; // the damping that occurs when a particle hits a wall
};


extern "C" void setParameters(SimulationParameters *hostParams);
#endif