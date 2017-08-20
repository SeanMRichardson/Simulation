#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
	// the vertex position for colouring
	FragPos = vec3(model * vec4(aPos, 1.0));
	Normal = mat3(transpose(inverse(model)))*aNormal;

	// set the viewing direction for the vertex
	gl_Position = proj*view * vec4(FragPos, 1.0);
	//set the point size for the mesh
	gl_PointSize = 4.0f;
}