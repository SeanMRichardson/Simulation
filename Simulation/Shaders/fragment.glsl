#version 330 core

out vec4 FragColour;

uniform vec3 lightColour;
uniform vec3 lightPos;
uniform vec3 viewPos;

in vec3 FragPos;
in vec3 Normal;

void main()
{
	// create the ambient light
	float ambientStrength = 0.1f;
	vec3 ambient = ambientStrength * lightColour;

	// create the directional light direction
	vec3 norm = normalize(Normal);
	vec3 lightDir = vec3(0.25, 1.0, -0.25);

	// calculate the diffuse light
	float diff = max(dot(norm, lightDir), 0.0f);
	vec3 diffuse = diff * lightColour;

	// calculate the specular component
	float specularStrength = 0.5f;
	vec3 viewDir = normalize(viewPos - FragPos);
	vec3 reflectDir = reflect(-lightDir, norm);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 4);
	vec3 specular = specularStrength * spec * lightColour;

	// apply the lighting calculations
	vec3 result = (ambient + diffuse + specular) * vec3(0.5f);
	FragColour =  vec4(result,1.0);
}