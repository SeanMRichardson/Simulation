#pragma once
#include "Molecule.h"
#include <glm\glm.hpp>
#include <vector>

struct CollisionPair	//Forms the output of the broadphase collision detection
{
	Molecule* pObjectA;
	Molecule* pObjectB;
};

struct Node
{
	glm::vec3 position;
	glm::vec3 maxPos;
	glm::vec3 minPos;
	float _halfSize;

	Node* children[2][2][2];
	std::vector<Molecule> objects;
	Node* parent;
};

class Octree
{
public:
	Octree(glm::vec3 position, std::vector<Molecule> molecules, int halfSize = 100);
	~Octree();

	Node* GetRootNode() { return _root; }

	bool CheckBoundaries(Molecule* obj, Node* root);
	void CalculateChildren(Node* root);
	void InsertObject(Node& node, Molecule* obj);

	void PassParentObject(std::vector<CollisionPair>* colpairs, Node* node, std::vector<Molecule>* childNodeObjects);

	void BuildCollisionPairs(std::vector<CollisionPair>* colpairs, Node* node);
private:
	Node* _root;

};

