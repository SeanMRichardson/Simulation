#pragma once
#include <GL/glew.h>
#include <GLFW\glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera
{
public:
	Camera(void) {
		yaw = 0.0f;
		pitch = 0.0f;
		roll = 0.0f;
	};

	Camera(float pitch, float yaw, glm::vec3 position) {
		this->pitch = pitch;
		this->yaw = yaw;
		this->roll = 0;
		this->position = glm::vec4(position, 1.0);
	}

	~Camera(void) {};

	void UpdateCamera(GLFWwindow* window, float msec = 10.0f);
	void ProcessInput(GLFWwindow* window, float msec);
	//Builds a view matrix for the current camera variables, suitable for sending straight
	//to a vertex shader (i.e it's already an 'inverse camera matrix').
	glm::mat4 BuildViewMatrix();

	//Gets position in world space
	glm::vec3 GetPosition() const { return position; }
	//Sets position in world space
	void	SetPosition(glm::vec3 val) { position = glm::vec4(val,1); }

	//Gets yaw, in degrees
	float	GetYaw()   const { return yaw; }
	//Sets yaw, in degrees
	void	SetYaw(float y) { yaw = y; }

	//Gets pitch, in degrees
	float	GetPitch() const { return pitch; }
	//Sets pitch, in degrees
	void	SetPitch(float p) { pitch = p; }

	glm::vec3 GetDirection() { return direction; }

	//Gets roll, in degrees
	float GetRoll() const { return roll; }
	//Sets roll, in degrees
	void SetRoll(float r) { roll = r; }
protected:
	float	yaw;
	float	pitch;
	float	roll;
	glm::vec4 position;
	glm::vec3 direction;
};

