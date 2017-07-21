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


// includes, cuda
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_gl.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>

#include "Shader.h"
#include "Camera.h"

#define MESHWIDTH 257
#define MESHHEIGHT 257

void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xPos, double yPos);
void scrollCallback(GLFWwindow* window, double xOffset, double yOffest);

int init();
void SetUp();
void GenerateVertices(float3* vertices, GLbyte* data);
GLuint GenerateIndices(GLuint* indices, GLuint numIndices);
GLbyte* ReadHeightData(char* string, int numVerts);

extern "C" void CalculateVertices(float3* vertices, GLbyte* data, int width, int height);

const GLint WIDTH = 800, HEIGHT = 600;

GLFWwindow *window;
int screenWidth, screenHeight;

Shader* shader;

glm::vec3 lightPos(1.2f, 1.0f, 2.0f);

Camera camera(glm::vec3(0, 10, 20));
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

float3* vertices;
GLuint* indices;
int numVerts;
GLint numIndices;
int indexCount;
GLbyte* data;

GLuint vao, vbo, ebo;
struct cudaGraphicsResource *vboResource, *eboResource;

extern "C" void Add(int* a, int* b, int* c);

int main(int argc, char **argv)
{
	if (init() != EXIT_SUCCESS)
	{
		return EXIT_FAILURE;
	}
	//store geometry vertices
	numVerts = MESHWIDTH*MESHHEIGHT;
	//vertices = new float3[numVerts];
	indexCount = (MESHWIDTH)*(MESHHEIGHT) * 6;
	indices = new GLuint[indexCount];
	numIndices = 0;

	data = ReadHeightData("Textures/Kilamanjaro.Raw", numVerts);
	GLbyte* device_data;

	checkCudaErrors(cudaMalloc((void **)&device_data, numVerts * sizeof(GLbyte)));
	checkCudaErrors(cudaMemcpy((void*)device_data, (void*)data, numVerts, cudaMemcpyHostToDevice));

	float3* device_vertices;

	int spectrumSize = MESHWIDTH*MESHHEIGHT * sizeof(float3);
	checkCudaErrors(cudaMalloc((void **)&device_vertices, spectrumSize));
	vertices = (float3*)malloc(spectrumSize);

	GenerateVertices(device_vertices, device_data);
	//GenerateVertices(vertices, data);
	checkCudaErrors(cudaMemcpy((void*)vertices, (void*)device_vertices, spectrumSize, cudaMemcpyDeviceToHost));

	cudaFree(device_data);
	cudaFree(device_vertices);

	numIndices = GenerateIndices(indices, numIndices);

	glEnable(GL_DEPTH_TEST);

	shader = new Shader("Shaders/vertex.glsl", "Shaders/fragment.glsl");
	if (!shader->LinkProgram())
	{
		return 0;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		printf("[%s]\n", argv[0]);
		printf("   Does not explicitly support -device=n in OpenGL mode\n");
		printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
		printf(" > %s -device=n -qatest\n", argv[0]);
		printf("exiting...\n");

		exit(EXIT_SUCCESS);
	}

	findCudaGLDevice(argc, (const char **)argv);

	SetUp();

	size_t num_bytes;

	// calculate slope for shading
	checkCudaErrors(cudaGraphicsMapResources(1, &vboResource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&data, &num_bytes, vboResource));

	checkCudaErrors(cudaGraphicsUnmapResources(1, &vboResource, 0));
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
	free(vertices);
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

	return EXIT_SUCCESS;
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
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&vboResource, vbo, cudaGraphicsMapFlagsWriteDiscard));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(GLuint), indices, GL_STATIC_DRAW);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&eboResource, ebo, cudaGraphicsMapFlagsWriteDiscard));

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	
}

void GenerateVertices(float3* vertices, GLbyte* data)
{
	CalculateVertices(vertices, data, MESHWIDTH, MESHHEIGHT);
	/*for (int x = 0; x < MESHWIDTH; ++x)
	{
		for (int z = 0; z < MESHHEIGHT; ++z)
		{
			if ((x < MESHWIDTH) && (z < MESHHEIGHT))
			{
				int offset = (x* MESHWIDTH) + z;
				vertices[offset].x = x;
				vertices[offset].y = abs(data[offset]);
				vertices[offset].z = -z;

				printf("Offset = %d ", offset);
				printf("%d ", x);
				printf("%d ", abs(data[offset])*0.2);
				printf("%d \n", z);
			}
		}
	}*/
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

	int bytesRead = fread(data, 1, numVerts, f);

	int result = ferror(f);
	if (result)
	{
		std::cout << "Error: could not read file" << std::endl;
	}

	fclose(f);
	return data;
}