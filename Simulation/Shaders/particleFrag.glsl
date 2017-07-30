#version 330

in vec4 outColor;

out vec4 vFragColor;

void main() 
{
	vec3 N;
	N.xy = gl_PointCoord* 2.0 - vec2(1.0);
	float mag = dot(N.xy, N.xy);
	if (mag > 1.0)
		discard; // kill pixels outside circle
	N.z = sqrt(1.0 - mag);

	vec3 lightDir = vec3(0.25, -1.0, -0.25);
	// calculate lighting
	float diffuse = max(0.0, dot(lightDir, N));

	vFragColor = vec4(0, 1, 1, 1) * diffuse;
}