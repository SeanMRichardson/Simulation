#include "stdafx.h"
#include "Camera.h"


/*
Polls the camera for keyboard / mouse movement.
Should be done once per frame! Pass it the msec since
last frame (default value is for simplicities sake...)
*/
void Camera::UpdateCamera(GLFWwindow* window, float msec)
{
	double x;
	double y;
	glfwGetCursorPos(window, &x, &y);

	//Update the mouse by how much
	pitch -= (x);
	yaw -= (y);

	//Bounds check the pitch, to be between straight up and straight down ;)
	pitch = glm::min(pitch, 90.0f);
	pitch = glm::max(pitch, -90.0f);

	if (yaw <0) {
		yaw += 360.0f;
	}
	if (yaw > 360.0f) {
		yaw -= 360.0f;
	}

	ProcessInput(window, msec);
}

/*
Generates a view matrix for the camera's viewpoint. This matrix can be sent
straight to the shader...it's already an 'inverse camera' matrix.
*/
glm::mat4 Camera::BuildViewMatrix() {
	//Why do a complicated matrix inversion, when we can just generate the matrix
	//using the negative values ;). The matrix multiplication order is important!
	glm::mat4 translation;
	translation = glm::rotate(translation, -pitch, glm::vec3(1, 0, 0)) *
		glm::rotate(translation, -yaw, glm::vec3(0, 1, 0)) *
		glm::rotate(translation, -roll, glm::vec3(0, 0, 1)) *
		glm::translate(translation, -glm::vec3(position));
	return translation;
};

void Camera::ProcessInput(GLFWwindow* window, float msec)
{
	float speed = 0.1;

	msec *= 5.0f;

	glm::mat4 rotationMat;

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		glm::rotate(rotationMat, yaw, glm::vec3(0, 1, 0));
		position += rotationMat * glm::vec4(0.0f, 0.0f, -1.0f, 1.0f) * msec * speed;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		glm::rotate(rotationMat, yaw, glm::vec3(0, 1, 0));
		position -= rotationMat * glm::vec4(0.0f, 0.0f, -1.0f, 1.0f) * msec * speed;
	}

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		glm::rotate(rotationMat, yaw, glm::vec3(0, 1, 0));
		position += rotationMat * glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f) * msec * speed;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		glm::rotate(rotationMat, yaw, glm::vec3(0, 1, 0));
		position -= rotationMat * glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f) * msec * speed;
	}

	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		roll++;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		roll--;
	}

	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
		position.y += msec * speed;
	}
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		position.y -= msec * speed;
	}
}