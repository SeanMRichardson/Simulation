// Simulation.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

const GLint WIDTH = 800, HEIGHT = 600;

bool LoadShaderFile(std::string from, std::string& into)
{
	std::ifstream file;
	std::string temp;

	std::cout << "Loading Shader Text From " << from << std::endl << std::endl;
	file.open(from.c_str());
	if (!file.is_open())
	{
		std::cout << "File does not exist!" << std::endl;
		return false;
	}
	while (!file.eof())
	{
		std::getline(file, temp);
		into += temp + "\n";
	}

	file.close();
	std::cout << into << std::endl << std::endl;
	std::cout << "Loaded shader text!" << std::endl << std::endl;
	return true;
}

GLuint GenerateShader(std::string from, GLenum type)
{
	std::cout << "Compiling Shader..." << std::endl;

	std::string load;
	if (!LoadShaderFile(from, load))
	{
		std::cout << "Compiling Failed!" << std::endl;
		return 0;
	}

	GLuint shader = glCreateShader(type);

	const char* chars = load.c_str();
	glShaderSource(shader, 1, &chars, NULL);
	glCompileShader(shader);

	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

	if (status == GL_FALSE)
	{
		std::cout << "Compiling Failed" << std::endl;
		char error[512];
		glGetInfoLogARB(shader, sizeof(error), NULL, error);
		std::cout << error;
		return 0;
	}

	std::cout << "Compiling Success" << std::endl << std::endl;
	return shader;
}

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

	// get the actual window size
	int screenWidth, screenHeight;
	glfwGetFramebufferSize(window, &screenWidth, &screenHeight);

	if (nullptr == window)
	{
		std::cout << "failed to create window" << std::endl;
		glfwTerminate();

		return EXIT_FAILURE;
	}

	// attach window
	glfwMakeContextCurrent(window);

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
	float vertices[] =
	{
		-0.5f,-0.5f,0.0f,
		 0.5f,-0.5f,0.0f,
		 0.f , 0.5f,0.0f
	};

	// create vbo
	GLuint vbo;
	glGenBuffers(1, &vbo);

	// create a vao
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// bind the buffer as an aaray buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	//store the buffer data
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	
	GenerateShader("Shaders/fragment.glsl", GL_FRAGMENT_SHADER);

	GLuint shaderProg;
	shaderProg = glCreateProgram();

	GLuint vert = GenerateShader("Shaders/vertex.glsl", GL_VERTEX_SHADER);
	GLuint frag = GenerateShader("Shaders/fragment.glsl", GL_FRAGMENT_SHADER);
	glAttachShader(shaderProg, vert);
	glAttachShader(shaderProg, frag);

	glLinkProgram(shaderProg);

	GLint success;
	GLchar infoLog[512];

	glGetProgramiv(shaderProg, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(shaderProg, 512, NULL, infoLog);
		std::cout << infoLog << std::endl;
	}

	glUseProgram(shaderProg);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(0);

	// simulation loop
	while (!glfwWindowShouldClose(window))
	{
		// get events
		glfwPollEvents();

		// reset the color
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// draw stuff
		glUseProgram(shaderProg);
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		
		// write to buffer
		glfwSwapBuffers(window);
	}

	glDeleteShader(vert);
	glDeleteShader(frag);

	// tidy up
	glfwTerminate();

	return EXIT_SUCCESS;

}

