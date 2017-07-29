#include "stdafx.h"
#include "MoleculeSystem.h"

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_gl_interop.h>


extern "C" void integrateVelocity(glm::vec3* pos, glm::vec3*vel, GLuint numberOfParticles, float deltaTime, glm::vec3* vertices);
extern "C" void calculateHash(GLuint *gridParticleHash, GLuint *gridParticleIndex, glm::vec3 *pos, int numParticles);
extern "C" void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);
extern "C" void reorderDataAndFindCellStart(GLuint *cellStart, GLuint *cellEnd, glm::vec3 *sortedPos, glm::vec3 *sortedVel, GLuint *gridParticleHash, GLuint *gridParticleIndex, glm::vec3 *oldPos, glm::vec3 *oldVel, GLuint numParticles, GLuint numCells);
extern "C" void collide(glm::vec3 *newVel, glm::vec3 *sortedPos, glm::vec3 *sortedVel, GLuint  *gridParticleIndex, GLuint *cellStart, GLuint  *cellEnd, GLuint numParticles, GLuint numCells);

MoleculeSystem::MoleculeSystem(ParticleSystemType pType, int maxParticles, int meshWidth, glm::vec3* vertices, glm::vec3 origin) : m_maxParticles(maxParticles), m_vertices(vertices)
{
	// grid configuration
	m_numberOfGridCells = GRID_SIZE * GRID_SIZE * GRID_SIZE;
	m_parameters.particleRadius = 1.0f / GRID_SIZE;
	m_parameters.gridSize = make_float3((float)GRID_SIZE, (float)GRID_SIZE, (float)GRID_SIZE);

	// each grid set to the size of a particle
	float cellSize = m_parameters.particleRadius * 2.0f;
	m_parameters.cellSize = make_float3(cellSize, cellSize, cellSize);


	// simulation parameters
	m_parameters.spring = 0.5f;
	m_parameters.damping = 0.02f;
	m_parameters.shear = 0.1f;
	m_parameters.attraction = 0.0f;
	m_parameters.boundaryDamping = -0.5f;
	m_parameters.gravity = make_float3(0.0f, -1.0f, 0.0f);
	m_parameters.globalDamping = 1.0f;

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");

	// copy parameters to constant memory
	setParameters(&m_parameters);

	m_shader = new Shader("Shaders/particleVert.glsl", "Shaders/particleFrag.glsl");
	if (!m_shader->LinkProgram())
	{
		std::cout << "ERROR: linking shader program" << std::endl;
	}

	// generate our initial set of particles
	GenerateMolecules(NUM_PARTICLES, origin);

	

}


MoleculeSystem::~MoleculeSystem()
{
	delete[] m_hPosition;
	delete[] m_hVelocity;
	delete[] m_hCellStart;
	delete[] m_hCellEnd;
}

/*
Generate a set of particles at the specified origin point
*/
void MoleculeSystem::GenerateMolecules(GLuint numberOfParticles, glm::vec3 origin)
{
	m_numParticles = numberOfParticles;

	// allocate host storage
	m_hPosition = new glm::vec3[m_numParticles];
	m_hVelocity = new glm::vec3[m_numParticles];
	memset(m_hPosition, 0, m_numParticles *  sizeof(glm::vec3));
	memset(m_hVelocity, 0, m_numParticles *  sizeof(glm::vec3));

	m_hCellStart = new GLuint[m_numberOfGridCells];
	memset(m_hCellStart, 0, m_numberOfGridCells * sizeof(GLuint));

	m_hCellEnd = new GLuint[m_numberOfGridCells];
	memset(m_hCellEnd, 0, m_numberOfGridCells * sizeof(GLuint));

	// allocate device data
	unsigned int memSize = sizeof(glm::vec3) * m_numParticles;

	checkCudaErrors(cudaMalloc((void **)&m_dVelocity, memSize));

	checkCudaErrors(cudaMalloc((void **)&m_dSortedPosition, memSize));
	checkCudaErrors(cudaMalloc((void **)&m_dSortedVelocity, memSize));

	checkCudaErrors(cudaMalloc((void **)&m_dGridParticleHash, m_numParticles * sizeof(GLuint)));
	checkCudaErrors(cudaMalloc((void **)&m_dGridParticleIndex, m_numParticles * sizeof(GLuint)));

	checkCudaErrors(cudaMalloc((void **)&m_dCellStart, m_numberOfGridCells * sizeof(GLuint)));
	checkCudaErrors(cudaMalloc((void **)&m_dCellEnd, m_numberOfGridCells * sizeof(GLuint)));

	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_vbo);

	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, &m_hPosition[0], GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_posvbo_resource, m_vbo, cudaGraphicsMapFlagsNone));

	// initialise the grid with particles
	float jitter = m_parameters.particleRadius*0.01f;
	GLuint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
	GLuint gridSize[3];
	gridSize[0] = gridSize[1] = gridSize[2] = s;
	initGrid(gridSize, m_parameters.particleRadius*2.0f, jitter, m_numParticles, origin);

	checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_posvbo_resource));
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_numParticles * sizeof(glm::vec3), m_hPosition);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_posvbo_resource, m_vbo, cudaGraphicsMapFlagsNone));

	checkCudaErrors(cudaMemcpy((char *)m_dVelocity, m_hVelocity, m_numParticles * sizeof(glm::vec3), cudaMemcpyHostToDevice));
}

void MoleculeSystem::AddMolecule(Molecule m)
{
	m_molecules.push_back(m);
}

void MoleculeSystem::KillMolecule()
{
	m_molecules.erase(m_molecules.begin());
}


void MoleculeSystem::Update(float deltaTime, glm::vec3* vertices, glm::vec3* normals)
{
	glm::vec3 *dPos;

	void *ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_posvbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, m_cuda_posvbo_resource));

	dPos = (glm::vec3*) ptr;
	// update constants
	setParameters(&m_parameters);

	

	integrateVelocity(dPos, m_dVelocity, m_numParticles, deltaTime, vertices);

	getLastCudaError("");

	calculateHash(m_dGridParticleHash, m_dGridParticleIndex, dPos, m_numParticles);

	// sort particles based on hash
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(m_dCellStart, m_dCellEnd, m_dSortedPosition, m_dSortedVelocity, m_dGridParticleHash, m_dGridParticleIndex, dPos, m_dVelocity, m_numParticles, m_numberOfGridCells);

	// process collisions
	collide(m_dVelocity, m_dSortedPosition,	m_dSortedVelocity, m_dGridParticleIndex, m_dCellStart, m_dCellEnd, m_numParticles, m_numberOfGridCells);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_posvbo_resource, 0));
}

void MoleculeSystem::Render(glm::mat4 proj, glm::mat4 view)
{
	glBindVertexArray(m_vao);

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	//glBufferSubData(GL_ARRAY_BUFFER, 0, m_numParticles * sizeof(glm::vec3), m_hPosition);

	glUseProgram(m_shader->GetProgram());
	glUniformMatrix4fv(glGetUniformLocation(m_shader->GetProgram(), "matProjection"), 1, GL_FALSE, glm::value_ptr(proj));
	glUniformMatrix4fv(glGetUniformLocation(m_shader->GetProgram(), "matModelview"), 1, GL_FALSE, glm::value_ptr(view));

	glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	glEnable(GL_PROGRAM_POINT_SIZE);

	glDrawArrays(GL_POINTS, 0, m_numParticles);

	glBindVertexArray(0);

	glDisable(m_shader->GetProgram());
}

void MoleculeSystem::BroadphaseCollisionDetection()
{
	m_BroadphaseCollisionPairs.clear();

	// create the octree for broadphase collisions
	Octree* tree = new Octree(glm::vec3(0, 0, 0), m_molecules, MESH_WIDTH/2);
	tree->BuildCollisionPairs(&m_BroadphaseCollisionPairs, tree->GetRootNode());
}

void MoleculeSystem::NarrowphaseCollisionDetection(glm::vec3* vertices, glm::vec3* normals, float deltaTime)
{
	if (m_BroadphaseCollisionPairs.size() > 0)
	{
		// Iterate over all possible collision pairs and perform accurate collision detection
		for (size_t i = 0; i < m_BroadphaseCollisionPairs.size(); ++i)
		{
			CollisionPair& cp = m_BroadphaseCollisionPairs[i];


			HandleMoleculeCollision(&cp);
		}
	}
}

void MoleculeSystem::HandleTerrainCollision(std::vector<Molecule>& molecules, glm::vec3 normal, float height, int index)
{
	if (CheckMoleculeCollisionWithTerrain(molecules[index], normal, height))
	{
		normal = normalize(normal);

		molecules[index].SetContactNormal(normalize(molecules[index].GetVelocity()));
		molecules[index].SetContactPoint(molecules[index].GetPosition() + (molecules[index].GetContactNormal()*molecules[index].GetRadius()));

		float y = GetVertexHeight(m_vertices, m_molecules[index].GetContactpoint().x, m_molecules[index].GetContactpoint().z);

		glm::vec3 point = glm::vec3(molecules[index].GetPosition().x, y, molecules[index].GetPosition().z);

		float depth = glm::length(molecules[index].GetContactpoint() - point);

		molecules[index].SetPenetrationDepth(depth);

		float contactVel = dot(molecules[index].GetVelocity(), normal);

		if (contactVel > 0)
			return;

		float e = 1.0f;
		float j = -(1.0f + e) * contactVel;
		j /= molecules[index].GetInverseMass();

		glm::vec3 impulse = j*normal;

		molecules[index].SetVelocity(m_molecules[index].GetInverseMass() * impulse * m_dampingFactor);
		
		if (molecules[index].GetPenetrationDepth() > 0)
		{
			const float k_slop = 0.1f;
			const float percent = 0.8f;
			glm::vec3 correction = glm::max(depth - k_slop, 0.0f) / molecules[index].GetInverseMass() * percent * normal;

			//molecules[index].AddForce(glm::vec3(0, 9.81, 0));
			molecules[index].SetPosition(molecules[index].GetPosition() + molecules[index].GetInverseMass() * correction);
		}
	}
}

void MoleculeSystem::HandleWallCollision(std::vector<Molecule>& molecules, int index)
{
	Wall walls[4] = { WALL_LEFT,  WALL_RIGHT,  WALL_FAR,  WALL_NEAR };

	for (int k = 0; k < 4; k++)
	{
		if (CheckMoleculeCollisionWithWall(molecules[index], walls[k]))
		{
			glm::vec3 wallNormal = normalize(GetWallDirection(walls[k]));

			float contactVel = dot(molecules[index].GetVelocity(), wallNormal);

			if (contactVel > 0)
				return;

			float e = 1.0f;
			float j = -(1.0f + e) * contactVel;
			j /= molecules[index].GetInverseMass();

			glm::vec3 impulse = j*wallNormal;

			molecules[index].SetVelocity(molecules[index].GetVelocity() + molecules[index].GetInverseMass() * impulse * m_dampingFactor);
			break;
		}
	}
}

void MoleculeSystem::HandleMoleculeCollision(CollisionPair* cp)
{
	if (CheckMoleculeMoleculeCollision(cp->pObjectA, cp->pObjectB))
	{
		glm::vec3 relativeVelocity = cp->pObjectB->GetVelocity() - cp->pObjectA->GetVelocity();
		glm::vec3 collisionNormal = cp->pObjectB->GetPosition() - cp->pObjectA->GetPosition();

		float velocityAlongNormal = dot(relativeVelocity, collisionNormal);

		if (velocityAlongNormal <= 0)
		{
			float e = 0.1f;
			float j = -(1.0f + e) * velocityAlongNormal;
			j /= (cp->pObjectA->GetInverseMass() + cp->pObjectB->GetInverseMass());

			glm::vec3 impulse = j*collisionNormal;

			cp->pObjectA->SetVelocity(-cp->pObjectA->GetInverseMass() * impulse);
			cp->pObjectB->SetVelocity(cp->pObjectB->GetInverseMass() * impulse);
		}
	}
}

float MoleculeSystem::GetVertexHeight(glm::vec3* vertices, float x, float z)
{
	float gridSquareSize = MESH_WIDTH / (MESH_WIDTH - 1);
	int gridX = floor(x / gridSquareSize);
	int gridZ = floor(z / gridSquareSize);

	if (gridX >= MESH_WIDTH - 1 || gridZ >= MESH_WIDTH - 1 || gridX < 0 || gridZ < 0)
		return 0;

	float xCoord = fmod(x, gridSquareSize)/gridSquareSize;
	float zCoord = fmod(z, gridSquareSize)/gridSquareSize;

	float height;

	if (xCoord <= (1 - zCoord))
	{
		int offset1 = (gridX*MESH_WIDTH) + gridZ;
		int offset2 = ((gridX +1)*MESH_WIDTH) + gridZ;
		int offset3 = (gridX*MESH_WIDTH) + gridZ +1;
		height = BarryCentric(glm::vec3(0,vertices[offset1].y,0), glm::vec3(1, vertices[offset2].y, 0), glm::vec3(0, vertices[offset3].y, 1), glm::vec2(xCoord, zCoord));
	}
	else
	{
		int offset1 = ((gridX + 1)*MESH_WIDTH) + gridZ;
		int offset2 = ((gridX +1)*MESH_WIDTH) + gridZ + 1;
		int offset3 = (gridX*MESH_WIDTH) + gridZ + 1;
		height = BarryCentric(glm::vec3(1, vertices[offset1].y, 0), glm::vec3(1, vertices[offset2].y, 1), glm::vec3(0, vertices[offset3].y, 1), glm::vec2(xCoord, zCoord));
	}

	return height;
}

glm::vec3 MoleculeSystem::GetVertexNormal(glm::vec3* normals, float x, float z)
{
	float gridSquareSize = MESH_WIDTH / (MESH_WIDTH - 1);
	int gridX = floor(abs(x) / gridSquareSize);
	int gridZ = floor(abs(z) / gridSquareSize);

	/*if (gridX >= m_meshWidth - 1 || gridZ >= m_meshWidth - 1 || gridX < 0 || gridZ < 0)
		return glm::vec3(0,0,0);*/

	float xCoord = fmod(x, gridSquareSize) / gridSquareSize;
	float zCoord = fmod(z, gridSquareSize) / gridSquareSize;

	glm::vec3 normal;

	if (xCoord <= 1 - zCoord)
	{
		int offset1 = (gridX*MESH_WIDTH) + gridZ;

		normal = normals[offset1];
	}
	else
	{
		int offset1 = ((gridX + 1)*MESH_WIDTH) + gridZ;

		normal = normals[offset1];
	}

	return normal;
}

bool MoleculeSystem::CheckMoleculeCollisionWithTerrain(Molecule m, glm::vec3 planeNormal, float height)
{
	glm::vec3 pointOnPlane = glm::vec3(m.GetPosition().x, height, m.GetPosition().z);

	glm::vec3 vector = m.GetPosition() - pointOnPlane;

	float magnitude = glm::length(vector);
	float normalMagnitude = glm::length(planeNormal);

	float cosAngle = dot(vector, planeNormal) / magnitude* normalMagnitude;

	float distance = cosAngle*magnitude;

	if (distance - m.GetRadius() <= height)
	{
		return true;
	}
	return false;
}

bool MoleculeSystem::CheckMoleculeCollisionWithWall(Molecule m, Wall w)
{
	glm::vec3 pointOnWall = GetPointOnWall(w);// = glm::vec3(m.GetPosition().x, height, m.GetPosition().z);

	glm::vec3 wallNormal = GetWallDirection(w);

	glm::vec3 vector = m.GetPosition() - pointOnWall;

	float magnitude = glm::length(vector);
	float normalMagnitude = glm::length(wallNormal);

	float cosAngle = dot(vector, wallNormal) / magnitude* normalMagnitude;

	float distance = cosAngle*magnitude;

	if (distance <= 0)
	{
		return true;
	}
	return false;
}

glm::vec3 MoleculeSystem::GetWallDirection(Wall w)
{
	switch (w)
	{
	case  WALL_LEFT:
		return glm::vec3(1, 0, 0);
	case  WALL_RIGHT:
		return glm::vec3(-1, 0, 0);
	case  WALL_NEAR:
		return glm::vec3(0, 0, 1);
	case  WALL_FAR:
		return glm::vec3(0, 0, -1);
	}
}

glm::vec3 MoleculeSystem::GetPointOnWall(Wall w)
{
	switch (w)
	{
	case  WALL_LEFT:
		return glm::vec3(0, 0, (MESH_WIDTH - 1) / 2);
	case  WALL_RIGHT:
		return glm::vec3(MESH_WIDTH -1, 0, (MESH_WIDTH - 1) / 2);
	case  WALL_NEAR:
		return glm::vec3((MESH_WIDTH - 1) /2, 0, 0);
	case  WALL_FAR:
		return glm::vec3((MESH_WIDTH - 1) /2, 0, MESH_WIDTH -1);
	}
}

bool MoleculeSystem::CheckMoleculeMoleculeCollision(Molecule* m1, Molecule* m2)
{
	glm::vec3 vector = m2->GetPosition() - m1->GetPosition();

	float distance = glm::length(vector);
	float radii = m1->GetRadius() + m2->GetRadius();

	return distance <= radii;
}

float MoleculeSystem::BarryCentric(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec2 pos) {
	float det = (p2.z - p3.z) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.z - p3.z);
	float l1 = ((p2.z - p3.z) * (pos.x - p3.x) + (p3.x - p2.x) * (pos.y - p3.z)) / det;
	float l2 = ((p3.z - p1.z) * (pos.x - p3.x) + (p1.x - p3.x) * (pos.y - p3.z)) / det;
	float l3 = 1.0f - l1 - l2;
	return l1 * p1.y + l2 * p2.y + l3 * p3.y;
}

void MoleculeSystem::Reset()
{

}

void MoleculeSystem::initGrid(GLuint *size, float spacing, float jitter, GLuint numParticles, glm::vec3 origin)
{
	srand(1973);

	for (GLuint z = 0; z<size[2]; z++)
	{
		for (GLuint y = 0; y<size[1]; y++)
		{
			for (GLuint x = 0; x<size[0]; x++)
			{
				GLuint i = (z*size[1] * size[0]) + (y*size[0]) + x;

				if (i < numParticles)
				{
					m_hPosition[i] = glm::vec3((spacing * x) + m_parameters.particleRadius - 1.0f + (rand()*2.0f - 1.0f)*jitter + origin.x, (spacing * y) + m_parameters.particleRadius - 1.0f + (rand()*2.0f - 1.0f)*jitter + origin.y, (spacing * z) + m_parameters.particleRadius - 1.0f + (rand()*2.0f - 1.0f)*jitter+origin.z);
					m_hVelocity[i] = glm::vec3(0.0f);
				}
			}
		}
	}
}
