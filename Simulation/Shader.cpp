#include "stdafx.h"
#include "Shader.h"

// Constructor
Shader::Shader(std::string vFile, std::string fFile, std::string gFile)
{
	// create the program
	program = glCreateProgram();

	// add shader instances
	objects[SHADER_VERTEX] = GenerateShader(vFile, GL_VERTEX_SHADER);
	objects[SHADER_FRAGMENT] = GenerateShader(fFile, GL_FRAGMENT_SHADER);
	objects[SHADER_GEOMETRY] = 0;

	// add geometry shader instance
	if (!gFile.empty())
	{
		objects[SHADER_GEOMETRY] = GenerateShader(gFile, GL_GEOMETRY_SHADER);
		glAttachShader(program, objects[SHADER_GEOMETRY]);
	}

	// attach fragment and vertex shader
	glAttachShader(program, objects[SHADER_VERTEX]);
	glAttachShader(program, objects[SHADER_FRAGMENT]);
}

// clean up
Shader::~Shader(void)
{
	for (int i = 0; i < 3; ++i)
	{
		glDetachShader(program, objects[i]);
		glDeleteShader(objects[i]);
	}
	glDeleteProgram(program);
}

// create a shader instance
GLuint Shader::GenerateShader(std::string from, GLenum type)
{
	// load the shader from a file
	std::string load;
	if (!LoadShaderFile(from, load))
	{
		std::cout << "Compiling Failed!" << std::endl;
		loadFailed = true;
		return 0;
	}

	// create the shader
	GLuint shader = glCreateShader(type);

	// compile the shader
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
		loadFailed = true;
		return 0;
	}

	std::cout << "Compiling Success" << std::endl << std::endl;
	loadFailed = false;
	return shader;
}

// load shader data from a file
bool Shader::LoadShaderFile(std::string from, std::string& into)
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
		getline(file, temp);
		into += temp + "\n";
	}

	file.close();
	std::cout << into << std::endl << std::endl;
	std::cout << "Loaded shader text!" << std::endl << std::endl;
	return true;
}

// link a program
bool Shader::LinkProgram()
{
	if (loadFailed)
	{
		return false;
	}
	glLinkProgram(program);


	GLint code;
	glGetProgramiv(program, GL_LINK_STATUS, &code);


	if (code == GL_FALSE)
	{
		std::cout << "Linking Failed" << std::endl;
		char error[512];
		glGetProgramInfoLog(program, sizeof(error), NULL, error);
		std::cout << error;
		loadFailed = true;
		return 0;
	}
	return code == GL_TRUE ? true : false;
}
