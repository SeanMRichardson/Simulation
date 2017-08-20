#version 330

in vec4 outColor;

out vec4 vFragColor;

void main() 
{
	// render the GL_Point as a sphere
	vec3 N;
	// get the position of a vertex
	N.xy = gl_PointCoord* 2.0 - vec2(1.0); 
	float mag = dot(N.xy, N.xy);
	if (mag > 1.0)
		discard; // kill pixels outside circle
	N.z = sqrt(1.0 - mag);

	//specify directional light direction 
	vec3 lightDir = vec3(0.25, -1.0, -0.25);
	// calculate diffuse lighting
	float diffuse = max(0.0, dot(lightDir, N));
	//apply the lighting to the vertex
	vFragColor = vec4(0, 1, 1, 1) * diffuse;
}