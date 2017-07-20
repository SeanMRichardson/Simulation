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

	GLuint GetProgram() { return program; }
	bool LinkProgram();
protected:
	void SetDefaultAttributes();
	bool LoadShaderFile(std::string from, std::string& into);
	GLuint GenerateShader(std::string from, GLenum type);

	GLuint objects[3];
	GLuint program;

	bool loadFailed;
};

