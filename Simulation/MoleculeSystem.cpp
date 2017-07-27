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
					AddMolecule(Molecule(glm::vec3(x*DENSITY + origin.x - 0.5f, y*DENSITY + origin.y, z*DENSITY + origin.z - 0.5)));
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


void MoleculeSystem::Update(float deltaTime, glm::vec3* vertices, glm::vec3* normals)
{
	BroadphaseCollisionDetection();
	NarrowphaseCollisionDetection(vertices, normals, deltaTime);
	
	m_dampingFactor = pow(0.85f, deltaTime);
	for (int i = 0; i < m_molecules.size(); i++)
	{
		auto compute_acc = [&](const glm::vec3& pos)
		{
			glm::vec3& acc = m_molecules[i].GetForce() * m_molecules[i].GetInverseMass();	// apply external forces to the system	
			return acc;
		};

		// first order integration
		glm::vec3 p1 = m_molecules[i].GetPosition();
		glm::vec3 v1 = m_molecules[i].GetVelocity();
		glm::vec3 a1 = compute_acc(p1);

		// second order integration
		glm::vec3 p2 = p1 + v1 * deltaTime / 2.0f;
		glm::vec3 v2 = v1 + a1 * deltaTime / 2.0f;
		glm::vec3 a2 = compute_acc(p2);// compute the acceleration at the second point

									   // third order integration
		glm::vec3 p3 = p1 + v2 * deltaTime / 2.0f;
		glm::vec3 v3 = v1 + a2 * deltaTime / 2.0f;
		glm::vec3 a3 = compute_acc(p3);// compute the acceleration at the third point

									   // fourth order integration
		glm::vec3 p4 = p1 + v3 * deltaTime;
		glm::vec3 v4 = v1 + a3 * deltaTime;
		glm::vec3 a4 = compute_acc(p4);// compute the acceleration at the fourth point

		m_molecules[i].SetVelocity(m_molecules[i].GetVelocity() * m_dampingFactor);

		m_molecules[i].SetPosition(p1 + ((v1 + (v2 * 2.0f) + (v3 * 2.0f) + v4) * deltaTime) / 6.0f);
		m_molecules[i].SetVelocity(v1 + ((a1 + (a2 * 2.0f) + (a3 * 2.0f) + a4) * deltaTime) / 6.0f);

		m_molecules[i].SetImpulseForce(glm::vec3(0));

		float height = GetVertexHeight(vertices, m_molecules[i].GetPosition().x, m_molecules[i].GetPosition().z);
		glm::vec3 normal = GetVertexNormal(normals, m_molecules[i].GetPosition().x, m_molecules[i].GetPosition().z);
		HandleWallCollision(m_molecules, i);
		HandleTerrainCollision(m_molecules, normal, height, i);
		
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

void MoleculeSystem::BroadphaseCollisionDetection()
{
	m_BroadphaseCollisionPairs.clear();

	// create the octree for broadphase collisions
	Octree* tree = new Octree(glm::vec3(0, 0, 0), m_molecules, m_meshWidth/2);
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

		float contactVel = dot(molecules[index].GetVelocity(), normal);

		if (contactVel > 0)
			return;

		float e = 1.0f;
		float j = -(1.0f + e) * contactVel;
		j /= molecules[index].GetInverseMass();

		glm::vec3 impulse = j*normal;

		molecules[index].SetVelocity(m_molecules[index].GetVelocity() + m_molecules[index].GetInverseMass() * impulse * m_dampingFactor);
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
			float e = 0.01f;
			float j = -(1.0f + e) * velocityAlongNormal;
			j /= (cp->pObjectA->GetInverseMass() + cp->pObjectB->GetInverseMass());

			glm::vec3 impulse = j*collisionNormal;

			cp->pObjectA->SetVelocity(cp->pObjectA->GetVelocity() - cp->pObjectA->GetInverseMass() * impulse * m_dampingFactor);
			cp->pObjectB->SetVelocity(cp->pObjectB->GetVelocity() + cp->pObjectB->GetInverseMass() * impulse * m_dampingFactor);
		}
	}
}

float MoleculeSystem::GetVertexHeight(glm::vec3* vertices, float x, float z)
{
	float gridSquareSize = m_meshWidth / (m_meshWidth - 1);
	int gridX = floor(x / gridSquareSize);
	int gridZ = floor(z / gridSquareSize);

	if (gridX >= m_meshWidth - 1 || gridZ >= m_meshWidth - 1 || gridX < 0 || gridZ < 0)
		return 0;

	float xCoord = fmod(x, gridSquareSize)/gridSquareSize;
	float zCoord = fmod(z, gridSquareSize)/gridSquareSize;

	float height;

	if (xCoord <= (1 - zCoord))
	{
		int offset1 = (gridX*m_meshWidth) + gridZ;
		int offset2 = ((gridX +1)*m_meshWidth) + gridZ;
		int offset3 = (gridX*m_meshWidth) + gridZ +1;
		height = BarryCentric(glm::vec3(0,vertices[offset1].y,0), glm::vec3(1, vertices[offset2].y, 0), glm::vec3(0, vertices[offset3].y, 1), glm::vec2(xCoord, zCoord));
	}
	else
	{
		int offset1 = ((gridX + 1)*m_meshWidth) + gridZ;
		int offset2 = ((gridX +1)*m_meshWidth) + gridZ + 1;
		int offset3 = (gridX*m_meshWidth) + gridZ + 1;
		height = BarryCentric(glm::vec3(1, vertices[offset1].y, 0), glm::vec3(1, vertices[offset2].y, 1), glm::vec3(0, vertices[offset3].y, 1), glm::vec2(xCoord, zCoord));
	}

	return height;
}

glm::vec3 MoleculeSystem::GetVertexNormal(glm::vec3* normals, float x, float z)
{
	float gridSquareSize = m_meshWidth / (m_meshWidth - 1);
	int gridX = floor(abs(x) / gridSquareSize);
	int gridZ = floor(abs(z) / gridSquareSize);

	/*if (gridX >= m_meshWidth - 1 || gridZ >= m_meshWidth - 1 || gridX < 0 || gridZ < 0)
		return glm::vec3(0,0,0);*/

	float xCoord = fmod(x, gridSquareSize) / gridSquareSize;
	float zCoord = fmod(z, gridSquareSize) / gridSquareSize;

	glm::vec3 normal;

	if (xCoord <= 1 - zCoord)
	{
		int offset1 = (gridX*m_meshWidth) + gridZ;

		normal = normals[offset1];
	}
	else
	{
		int offset1 = ((gridX + 1)*m_meshWidth) + gridZ;

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

	if (distance <= height)
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
		return glm::vec3(0, 0, (m_meshWidth - 1) / 2);
	case  WALL_RIGHT:
		return glm::vec3(m_meshWidth-1, 0, (m_meshWidth - 1) / 2);
	case  WALL_NEAR:
		return glm::vec3((m_meshWidth - 1) /2, 0, 0);
	case  WALL_FAR:
		return glm::vec3((m_meshWidth - 1) /2, 0, m_meshWidth-1);
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
