#version 330

uniform mat4 matModelview;
uniform mat4 matProjection;

layout(location = 0) in vec4 vVertex;
layout(location = 1) in vec4 vColor;

out vec4 outColor;

uniform float pointScale;
uniform float pointRadius;

void main()
{
	// get the viewing direction
	vec4 eyePos = matModelview * vVertex;
	gl_Position = matProjection * eyePos;

	//get the object colour
	outColor = vColor;

	// find the distance from the eye to the object
	float dist = length(eyePos.xyz);
	float att = inversesqrt(0.1f*dist);

	// calculate the correct viewing scale for the GL_POINT 
	gl_PointSize = pointRadius * (pointScale / dist);
}