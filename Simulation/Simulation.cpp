// Simulation.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include <stdio.h>
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
#include "Camera.h"

#define MESHWIDTH 257
#define MESHHEIGHT 257

void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xPos, double yPos);
void scrollCallback(GLFWwindow* window, double xOffset, double yOffest);

int init();
void SetUp();
void GenerateVertices(glm::vec3* vertices, GLbyte* data);
GLuint GenerateIndices(GLuint* indices, GLuint numIndices);
GLbyte* ReadHeightData(char* string, int numVerts);

const GLint WIDTH = 800, HEIGHT = 600;

GLFWwindow *window;
int screenWidth, screenHeight;

Shader* shader;

glm::vec3 lightPos(1.2f, 1.0f, 2.0f);

Camera camera(glm::vec3(0, 20, 20));
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

glm::vec3* vertices;
GLuint* indices;
int numVerts;
GLint numIndices;
int indexCount;
GLbyte* data;

GLuint vao, vbo, ebo;

int main()
{
	if (!init())
	{
		return EXIT_FAILURE;
	}
	
	data = ReadHeightData("Textures/Kilamanjaro.Raw", numVerts);
	GenerateVertices(vertices, data);
	numIndices = GenerateIndices(indices, numIndices);

	//int num = sizeof(indices);
	glEnable(GL_DEPTH_TEST);

	shader = new Shader("Shaders/vertex.glsl", "Shaders/fragment.glsl");
	if (!shader->LinkProgram())
	{
		return 0;
	}

	SetUp();

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
		proj = glm::perspective(glm::radians(camera.Zoom), (float)screenWidth / (float)screenHeight, 0.1f, 10000.0f);

		glUniformMatrix4fv(glGetUniformLocation(shader->GetProgram(), "proj"), 1, GL_FALSE, glm::value_ptr(proj));

		glm::mat4 view;
		view = camera.GetViewMatrix();		

		glUniformMatrix4fv(glGetUniformLocation(shader->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(view));

		glUniform3f(glGetUniformLocation(shader->GetProgram(), "lightColour"), 1.0f, 1.0f, 1.0f);
		glUniform3f(glGetUniformLocation(shader->GetProgram(), "lightPos"), lightPos.x, lightPos.y, lightPos.z);
		glUniform3f(glGetUniformLocation(shader->GetProgram(), "viewPos"), camera.Position.x, camera.Position.y, camera.Position.z);
		
		glBindVertexArray(vao);

		glm::mat4 model;

		model = glm::translate(model, glm::vec3(0,0,0));
		float angle = 20.0f * 0;
		model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
		glUniformMatrix4fv(glGetUniformLocation(shader->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(model));

		//glDrawArrays(GL_TRIANGLES, 0, 6);
		glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);

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
		camera.ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, deltaTime);

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

	camera.ProcessMouseMovement(xoffset, yoffset);
}

void scrollCallback(GLFWwindow* window, double xOffset, double yOffest)
{
	camera.ProcessMouseScroll(yOffest);
}

int init()
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
	window = glfwCreateWindow(WIDTH, HEIGHT, "My Window", nullptr, nullptr);

	if (nullptr == window)
	{
		std::cout << "failed to create window" << std::endl;
		glfwTerminate();

		return EXIT_FAILURE;
	}

	// attach window
	glfwMakeContextCurrent(window);

	// get the actual window size
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
	numVerts = MESHWIDTH*MESHHEIGHT;
	vertices = new glm::vec3[numVerts];
	indexCount = (MESHWIDTH)*(MESHHEIGHT) * 6;
	indices = new GLuint[indexCount];
	numIndices = 0;

	numVerts = MESHWIDTH * MESHHEIGHT;
	vertices = new glm::vec3[numVerts];

	return 0;
}

void SetUp()
{
	// create a vertex array
	glGenVertexArrays(1, &vao);
	// create vertex buffer
	glGenBuffers(1, &vbo);
	//create element buffer
	glGenBuffers(1, &ebo);

	// bind the vertex array
	glBindVertexArray(vao);

	// bind the buffer as an aaray buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//store the buffer data
	glBufferData(GL_ARRAY_BUFFER, numVerts * sizeof(glm::vec3), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(GLuint), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void GenerateVertices(glm::vec3* vertices, GLbyte* data)
{
	for (int x = 0; x < MESHWIDTH; ++x)
	{
		for (int z = 0; z < MESHHEIGHT; ++z)
		{
			if ((x < MESHWIDTH) && (z < MESHHEIGHT))
			{
				int offset = (x * MESHWIDTH) + z;
				vertices[offset] = glm::vec3(x, abs(data[offset]), z);
			}
		}
	}
}

GLuint GenerateIndices(GLuint* indices, GLuint numIndices)
{
	bool tri = false;

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
	return numIndices;
}

GLbyte* ReadHeightData(char* string, int numVerts)
{
	FILE *f = NULL;

	f = fopen(string, "rb");
	if (f == NULL)
	{
		std::cout << "Error: could not find/open file" << std::endl;
	}

	GLbyte* data = new GLbyte[numVerts];

	fread(data, 1, numVerts, f);

	int result = ferror(f);
	if (result)
	{
		std::cout << "Error: could not read file" << std::endl;
	}

	fclose(f);
	return data;
}