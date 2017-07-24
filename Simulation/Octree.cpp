#include "stdafx.h"
#include "Octree.h"


Octree::Octree(glm::vec3 position, std::vector<Molecule> molecules, int halfSize)
{
	// create a new node
	_root = new Node();
	_root->parent = NULL;

	//set the position and size
	_root->position = position;
	_root->_halfSize = float(halfSize);

	// define the size of the root node in world space
	_root->maxPos = glm::vec3(position.x + halfSize, position.y + halfSize, position.z + halfSize);
	_root->minPos = glm::vec3(position.x - halfSize, position.y - halfSize, position.z - halfSize);

	//get all the objects inside the node
	for (auto obj : molecules)
	{
		//add the objects to the root node
		InsertObject(*_root, &obj);
	}
}

Octree::~Octree()
{
	delete _root;
}

void Octree::InsertObject(Node& node, Molecule* obj)
{
	// calculate the children of the current node
	CalculateChildren(&node);

	// check if the child contains any of the objects
	bool inChild = false;
	for each(Node* n in node.children)
	{
		if (n != NULL)
		{
			if (CheckBoundaries(obj, n))
			{
				//recursivly add the objects to the node if they fit
				InsertObject(*n, obj);
				inChild = true;
			}
		}
	}
	if (!inChild)
		node.objects.push_back(*obj); // if not in any child add to the parent node
}

// checks if an object is inside a node
bool Octree::CheckBoundaries(Molecule* obj, Node* node)
{
	float radius = obj->GetRadius();
	glm::vec3 pos = obj->GetPosition();

	if (pos.x + radius <= node->maxPos.x
		&& pos.x - radius >= node->minPos.x
		&& pos.y + radius <= node->maxPos.y
		&& pos.y - radius >= node->minPos.y
		&& pos.z + radius <= node->maxPos.z
		&& pos.z - radius >= node->minPos.z)
	{
		return true;
	}
	return false;
}

void Octree::CalculateChildren(Node* root)
{
	// only ever 8 children per node
	if (root->children != NULL)
	{
		//re-calculate the size for the new nodes
		glm::vec3 child_dims = (root->maxPos - root->minPos) * 0.5f;

		// create the eight new nodes at offsetted positions
		for (int k = 0; k < 2; k++)
		{
			for (int j = 0; j < 2; j++)
			{
				for (int i = 0; i < 2; i++)
				{
					glm::vec3 pos = child_dims * 0.5f;

					root->children[i][j][k] = new Node;
					root->children[i][j][k]->parent = root;
					root->children[i][j][k]->position.x = pos.x + (i *child_dims.x);
					root->children[i][j][k]->position.y = pos.y + (j * child_dims.y);
					root->children[i][j][k]->position.z = pos.z + (k * child_dims.z);

					root->children[i][j][k]->_halfSize = root->_halfSize * 0.5f;

					root->children[i][j][k]->minPos = glm::vec3(
						i *child_dims.x,
						j *child_dims.y,
						k *child_dims.z);

					root->children[i][j][k]->minPos = glm::vec3(
						child_dims.x + i *child_dims.x,
						child_dims.y + j *child_dims.y,
						child_dims.z + k *child_dims.z);


					for (int l = 0; l < 2; l++)
					{
						for (int m = 0; m < 2; m++)
						{
							for (int n = 0; n < 2; n++)
							{
								root->children[i][j][k]->children[n][m][l] = NULL;
							}
						}
					}
				}
			}
		}
	}
}

// add the collision pair between parent node and children nodes
void Octree::PassParentObject(std::vector<CollisionPair>* colpairs, Node* node, std::vector<Molecule>* childNodeObjects)
{
	if (node != NULL)
	{
		Molecule *m_pObj1, *m_pObj2;
		if (node->objects.size() > 0)
		{
			//loop through all the objects in the node
			for (size_t i = 0; i < node->objects.size() - 1; ++i)
			{
				m_pObj1 = &node->objects[i];
				for (size_t j = 0; j < childNodeObjects->size(); ++j)
				{
					m_pObj2 = &(*childNodeObjects)[j];


					float bothRadii = m_pObj2->GetRadius() + m_pObj1->GetRadius();
					float bothRadii_sq = bothRadii * bothRadii;

					float distance_sq = pow(glm::length(m_pObj2->GetPosition() - m_pObj1->GetPosition()), 2);

					if (distance_sq <= bothRadii_sq)
					{
						CollisionPair cp;
						cp.pObjectA = m_pObj1;
						cp.pObjectB = m_pObj2;
						colpairs->push_back(cp);
					}
				}
			}
		}
		PassParentObject(colpairs, node->parent, childNodeObjects);
	}
}

// build all of the other collision pairs 
void Octree::BuildCollisionPairs(std::vector<CollisionPair>* colpairs, Node* node)
{
	Molecule *m_pObj1, *m_pObj2;

	if (node != NULL)
	{

		if (node->objects.size() > 0)
		{
			PassParentObject(colpairs, node->parent, &node->objects);

			//loop through all the objects in the node
			for (size_t i = 0; i < node->objects.size() - 1; ++i)
			{
				m_pObj1 = &node->objects[i];
				for (size_t j = i + 1; j < node->objects.size(); ++j)
				{
					m_pObj2 = &node->objects[j];

					float bothRadii = m_pObj2->GetRadius() + m_pObj1->GetRadius();
					float bothRadii_sq = bothRadii * bothRadii;

					float distance_sq = pow(glm::length(m_pObj2->GetPosition() - m_pObj1->GetPosition()),2);

					if (distance_sq <= bothRadii_sq)
					{
						CollisionPair cp;
						cp.pObjectA = m_pObj1;
						cp.pObjectB = m_pObj2;
						colpairs->push_back(cp);
					}
				}
			}

		}
	}
	for (int k = 0; k < 2; k++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int i = 0; i < 2; i++)
			{
				if (node->children[i][j][k])
					BuildCollisionPairs(colpairs, node->children[i][j][k]);
			}
		}
	}
}
