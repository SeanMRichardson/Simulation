#ifndef MOLECULE_SYSTEM_H
#define MOLECULE_SYSTEM_H

#include "glm/glm.hpp"
#include "GL/glew.h"

#include <vector_types.h>
#include <helper_math.h>



struct SimulationParameters
{
	int numberOfParticles;

	float3 gravity;
	float globalDamping;
	float particleRadius;
	float smoothingRadius;

	float3 gridSize;
	GLuint numCells;
	float3 cellSize;

	GLuint maxParticlesPerCell;

	float mass;
	float particleDensity;
	float restDensity;
	float gasConstant;
	float viscosityCoefficient;

	float spring;
	float damping;
	float shear;
	float attraction;
	float boundaryDamping;
};


extern "C" void setParameters(SimulationParameters *hostParams);
#endif