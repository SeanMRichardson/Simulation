#pragma once
#include <GL\glew.h>
#include <iostream>
#include <string>
#include <fstream>

#define SHADER_VERTEX	0
#define SHADER_FRAGMENT	1
#define SHADER_GEOMETRY 2

class Shader
{
public:
	Shader(std::string vfILE, std::string fFile, std::string gFile = "");
	~Shader(void);

	GLuint GetProgram() { return program; } // get the shader program instance
	bool LinkProgram(); // link the program
protected:

	bool LoadShaderFile(std::string from, std::string& into); // load shader data from a file
	GLuint GenerateShader(std::string from, GLenum type); // create a shader instance

	GLuint objects[3]; // share objects, fragment, vertex and geometry
	GLuint program; // the program

	bool loadFailed; // flag to indicate success/failure
};

