// Simulation.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#include "Shader.h"

#define MESHWIDTH 16
#define MESHHEIGHT 16

void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xPos, double yPos);
void scrollCallback(GLFWwindow* window, double xOffset, double yOffest);


const GLint WIDTH = 800, HEIGHT = 600;

Shader* shader;

glm::vec3 lightPos(1.2f, 1.0f, 2.0f);

glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, -3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
float fov = 45.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main()
{
	// init glfw
	glfwInit();

	// set the version
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

	// use new opengl and maximise compatibility
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// forward compatibility
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	// set fixed
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// create the window
	GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "My Window", nullptr, nullptr);

	


	if (nullptr == window)
	{
		std::cout << "failed to create window" << std::endl;
		glfwTerminate();

		return EXIT_FAILURE;
	}

	// attach window
	glfwMakeContextCurrent(window);

	// get the actual window size
	int screenWidth, screenHeight;
	glfwGetFramebufferSize(window, &screenWidth, &screenHeight);

	//// camera bindings
	glfwSetCursorPosCallback(window, mouseCallback);
	glfwSetScrollCallback(window, scrollCallback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// use modern approach to retrieve function pointers etc.
	glewExperimental = GL_TRUE;

	// and initialise
	if (GLEW_OK != glewInit())
	{

		std::cout << "Failed to init GLEW" << std::endl;
		return EXIT_FAILURE;
	}

	// set viewport
	glViewport(0, 0, screenWidth, screenHeight);

	//store geometry vertices
	int numVerts = MESHWIDTH*MESHHEIGHT;
	glm::vec3* vertices = new glm::vec3[numVerts];
	int indexCount = (MESHWIDTH)*(MESHHEIGHT) * 6;
	GLuint* indices = new GLuint[indexCount];

	std::ifstream file("Textures/Kilamanjaro.Raw", std::ios::binary);
	if (!file) {
		return EXIT_FAILURE;
	}

	numVerts = MESHWIDTH * MESHHEIGHT;
	vertices = new glm::vec3[numVerts];
	
	unsigned char * data = new unsigned char[numVerts];
	file.read((char *)data, numVerts * sizeof(unsigned char));
	file.close();

	for (int x = 0; x < MESHWIDTH; ++x)
	{
		for (int z = 0; z < MESHHEIGHT; ++z)
		{
			int offset = (x * MESHWIDTH) + z;
			vertices[offset] = glm::vec3(x, data[offset], -z);
		}
	}

	GLuint numIndices = 0;
	bool tri = true;

	for (int x = 0; x < MESHWIDTH - 1; ++x)
	{
		for (int z = 0; z < MESHHEIGHT - 1; ++z)
		{
			long a = (x * (MESHWIDTH)) + z;
			long b = ((x + 1) * (MESHWIDTH)) + z;
			long c, d;
			c = ((x + 1) * (MESHWIDTH)) + (z + 1);
			d = (x * (MESHWIDTH)) + (z + 1);


			if (tri)
			{
				indices[numIndices++] = c;
				indices[numIndices++] = b;
				indices[numIndices++] = a;

				indices[numIndices++] = a;
				indices[numIndices++] = d;
				indices[numIndices++] = c;
			}
			else
			{
				indices[numIndices++] = b;
				indices[numIndices++] = a;
				indices[numIndices++] = d;

				indices[numIndices++] = d;
				indices[numIndices++] = c;
				indices[numIndices++] = b;
			}
			tri = !tri;
		}
	}


	glEnable(GL_DEPTH_TEST);

	shader = new Shader("Shaders/vertex.glsl", "Shaders/fragment.glsl");
	if (!shader->LinkProgram())
	{
		return 0;
	}

	// create a vao
	GLuint vao;
	glGenVertexArrays(1, &vao);
	// create vbo
	GLuint vbo;
	glGenBuffers(1, &vbo);
	GLuint ebo;
	glGenBuffers(1, &ebo);

	glBindVertexArray(vao);

	// bind the buffer as an aaray buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//store the buffer data
	glBufferData(GL_ARRAY_BUFFER, numVerts * sizeof(glm::vec3), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount*sizeof(GLuint), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	// simulation loop
	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);

		// reset the color
		glClearColor(0, 0, 0, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// draw stuff
		glUseProgram(shader->GetProgram());

		glm::mat4 proj;
		proj = glm::perspective(glm::radians(45.0f), (float)screenWidth / (float)screenHeight, 0.1f, 100.0f);

		glUniformMatrix4fv(glGetUniformLocation(shader->GetProgram(), "proj"), 1, GL_FALSE, glm::value_ptr(proj));

		glm::mat4 view;
		view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);		

		glUniformMatrix4fv(glGetUniformLocation(shader->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(view));

		glUniform3f(glGetUniformLocation(shader->GetProgram(), "lightColour"), 1.0f, 1.0f, 1.0f);
		glUniform3f(glGetUniformLocation(shader->GetProgram(), "lightPos"), lightPos.x, lightPos.y, lightPos.z);
		glUniform3f(glGetUniformLocation(shader->GetProgram(), "viewPos"), cameraPos.x, cameraPos.y, cameraPos.z);
		
		glBindVertexArray(vao);

		glm::mat4 model;

		model = glm::translate(model, glm::vec3(0,0,0));
		float angle = 20.0f * 0;
		model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
		glUniformMatrix4fv(glGetUniformLocation(shader->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(model));

		//glDrawArrays(GL_TRIANGLES, 0, 6);
		glDrawElements(GL_TRIANGLES, indexCount * sizeof(GLuint), GL_UNSIGNED_INT, 0);

		// write to buffer
		glfwSwapBuffers(window);
		// get events
		glfwPollEvents();
	}

	delete[] vertices;
	delete[] indices;

	delete shader;
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ebo);
	// tidy up
	glfwTerminate();

	return EXIT_SUCCESS;
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float cameraSpeed = 10.0f * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;

}

void mouseCallback(GLFWwindow* window, double xPos, double yPos)
{
	if (firstMouse)
	{
		lastX = xPos;
		lastY = yPos;
		firstMouse = false;
	}

	float xoffset = xPos - lastX;
	float yoffset = lastY - yPos;
	lastX = xPos;
	lastY = yPos;

	float sensitivity = 0.1f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	/*if (pitch > 89.0f)
		pitch = 89.0f;*/
	/*if (pitch < -89.0f)
		pitch = -89.0f;*/
	
	glm::vec3 front;
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraFront = glm::normalize(front);
}

void scrollCallback(GLFWwindow* window, double xOffset, double yOffest)
{
	if (fov >= 1.0f && fov <= 45.0f)
		fov -= yOffest;
	if (fov <= 1.0f)
		fov = 1.0f;
	if (fov >= 45.0f)
		fov = 45.0f;
}