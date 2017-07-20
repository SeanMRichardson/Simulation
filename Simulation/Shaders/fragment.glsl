#version 330 core

out vec4 FragColour;

uniform vec3 lightColour;
uniform vec3 lightPos;
uniform vec3 viewPos;

in vec3 FragPos;
in vec3 Normal;

void main()
{
	float ambientStrength = 0.1f;
	vec3 ambient = ambientStrength * lightColour;

	vec3 norm = normalize(Normal);
	vec3 lightDir = vec3(0.0, 1.0, 0.0);//normalize(lightPos - FragPos);

	float diff = max(dot(norm, lightDir), 0.0f);
	vec3 diffuse = diff * lightColour;

	float specularStrength = 0.5f;
	vec3 viewDir = normalize(viewPos - FragPos);
	vec3 reflectDir = reflect(-lightDir, norm);

	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 4);
	vec3 specular = specularStrength * spec * lightColour;

	vec3 result = (/*ambient +*/ diffuse /*+ specular*/) * vec3(1.0f, 0.6f, 0.4f);

	FragColour =  vec4(result,1.0);
}