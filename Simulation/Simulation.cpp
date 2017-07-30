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

#include <helper_math.h>

#include "Shader.h"
#include "Camera.h"
#include "MoleculeSystem.h"


#define START_X 0
#define START_Y 0
#define START_Z 0

//const size_t NUM_PARTICLES = 1000;

MoleculeSystem* m_system;

void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xPos, double yPos);
void scrollCallback(GLFWwindow* window, double xOffset, double yOffest);

int init();
void SetUp();
void GenerateVertices(glm::vec3* vertices, GLbyte* data);
//GLuint GenerateIndices(GLuint* indices, GLuint numIndices);
void GenerateIndices(GLuint* indices, GLuint numIndices, int width, int height);
GLbyte* ReadHeightData(char* string, int numVerts);

extern "C" void CalculateVertices(glm::vec3* vertices, GLbyte* data, int width, int height);
extern "C" void CalculateIndices(GLuint* indices, GLint numIndices, int width, int height);
extern "C" void CalculateNormals(glm::vec3* normals, GLuint* indices, glm::vec3* vertices, int width, int height);

const GLint WIDTH = 800, HEIGHT = 600;

GLFWwindow *window;
int screenWidth, screenHeight;

Shader* shader;

glm::vec3 lightPos(1.2f, 1.0f, 2.0f);

Camera camera(glm::vec3(START_X - 5.0f, START_Y, START_Z), glm::vec3(0, 1, 0), 10, 0);
//Camera camera(glm::vec3(0,0,0));

float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

glm::vec3* vertices;
glm::vec3* normals;
GLuint* indices;
int numVerts;
GLint numIndices;
int indexCount;
GLbyte* data;

GLuint vao, vbo, nbo, ebo;
struct cudaGraphicsResource *vboResource, *nboResource, *eboResource;
bool play = false;

int main(int argc, char **argv)
{
	if (init() != EXIT_SUCCESS)
	{
		return EXIT_FAILURE;
	}
	
	findCudaGLDevice(argc, (const char **)argv);

	// the number of vertices in the terrain file
	numVerts = MESH_WIDTH * MESH_HEIGHT;

	// the number of indices in the terrain file
	indexCount = MESH_WIDTH * MESH_HEIGHT * 6;
	
	numIndices = 0;

	// read the raw data from the terrain file
	data = ReadHeightData("Textures/Kilamanjaro.Raw", numVerts);
	GLbyte* device_data;

	checkCudaErrors(cudaMalloc((void **)&device_data, numVerts * sizeof(GLbyte)));
	checkCudaErrors(cudaMemcpy((void*)device_data, (void*)data, numVerts, cudaMemcpyHostToDevice));

	glm::vec3* device_vertices;

	// allocate storage on device for vertex data
	int spectrumSize = numVerts * sizeof(glm::vec3);
	checkCudaErrors(cudaMalloc((void **)&device_vertices, spectrumSize));
	vertices = (glm::vec3*)malloc(spectrumSize);

	GenerateVertices(device_vertices, device_data);
	//GenerateVertices(vertices, data);

	// read vertices data back from cuda
	checkCudaErrors(cudaMemcpy((void*)vertices, (void*)device_vertices, spectrumSize, cudaMemcpyDeviceToHost));

	cudaFree(device_data);
	//cudaFree(device_vertices);

	GLuint* device_indices;
	checkCudaErrors(cudaMalloc((void **)&device_indices, indexCount * sizeof(GLuint)));
	indices = (GLuint*)malloc(indexCount * sizeof(GLuint));

	GenerateIndices(device_indices, indexCount, MESH_WIDTH, MESH_HEIGHT);
	checkCudaErrors(cudaMemcpy((void*)indices, (void*)device_indices, indexCount * sizeof(GLuint), cudaMemcpyDeviceToHost));

	glm::vec3* device_normals;
	checkCudaErrors(cudaMalloc((void **)&device_normals, spectrumSize));
	normals = (glm::vec3*)malloc(spectrumSize);


	checkCudaErrors(cudaMemcpy((void*)device_vertices, (void*)vertices, numVerts, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy((void*)device_indices, (void*)indices, indexCount, cudaMemcpyHostToDevice));

	CalculateNormals(device_normals, device_indices, device_vertices, MESH_WIDTH, MESH_HEIGHT);

	checkCudaErrors(cudaMemcpy((void*)normals, (void*)device_normals, spectrumSize, cudaMemcpyDeviceToHost));

	
	cudaFree(device_indices);
	cudaFree(device_normals);

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

	

	SetUp();



	m_system = new MoleculeSystem(MESH_WIDTH, vertices, glm::vec3(START_X, START_Y, START_Z));

	

	// simulation loop
	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);

		glm::mat4 proj, view;
		//draw height data
		{
			// reset the color
			glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glEnable(GL_PROGRAM_POINT_SIZE);

			// draw stuff
			glUseProgram(shader->GetProgram());


			proj = glm::perspective(glm::radians(camera.Zoom), (float)screenWidth / (float)screenHeight, 0.1f, 10000.0f);

			glUniformMatrix4fv(glGetUniformLocation(shader->GetProgram(), "proj"), 1, GL_FALSE, glm::value_ptr(proj));

			view = camera.GetViewMatrix();

			glUniformMatrix4fv(glGetUniformLocation(shader->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(view));

			glUniform3f(glGetUniformLocation(shader->GetProgram(), "lightColour"), 1.0f, 1.0f, 1.0f);
			glUniform3f(glGetUniformLocation(shader->GetProgram(), "lightPos"), lightPos.x, lightPos.y, lightPos.z);
			glUniform3f(glGetUniformLocation(shader->GetProgram(), "viewPos"), camera.Position.x, camera.Position.y, camera.Position.z);

			glBindVertexArray(vao);

			glm::mat4 model;

			model = glm::translate(model, glm::vec3(0, 0, 0));
			float angle = 20.0f * 0;
			model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
			glUniformMatrix4fv(glGetUniformLocation(shader->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(model));

			glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
		}

		// particle system
		{
			if (play)
			{
				m_system->Update(0.03, device_vertices, normals);
			}
			m_system->Render(proj, view, HEIGHT, camera.Zoom);
		}
		
		// write to buffer
		glfwSwapBuffers(window);
		// get events
		glfwPollEvents();
	}
	cudaFree(device_vertices);
	//delete[] vertices;
	free(vertices);
	delete[] indices;

	delete shader;
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &nbo);
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
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		play = true;
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
	//create normal buffer
	glGenBuffers(1, &nbo);
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

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,  0, (void*)0);
	glEnableVertexAttribArray(0);

	// bind the buffer as an aaray buffer
	glBindBuffer(GL_ARRAY_BUFFER, nbo);
	//store the buffer data
	glBufferData(GL_ARRAY_BUFFER, numVerts * sizeof(glm::vec3), normals, GL_STATIC_DRAW);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&nboResource, nbo, cudaGraphicsMapFlagsWriteDiscard));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	
}

void GenerateVertices(glm::vec3* vertices, GLbyte* data)
{
	CalculateVertices(vertices, data, MESH_WIDTH, MESH_HEIGHT);
}

void GenerateIndices(GLuint* indices, GLuint numIndices, int width, int height)
{
	CalculateIndices(indices, numIndices, width, height);
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

	size_t bytesRead = fread(data, 1, numVerts, f);

	int result = ferror(f);
	if (result)
	{
		std::cout << "Error: could not read file" << std::endl;
	}

	fclose(f);
	return data;
}