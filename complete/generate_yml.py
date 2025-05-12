from jinja2 import Environment, FileSystemLoader
import os

# Set the directory for the Jinja2 template
template_dir = os.path.abspath('./complete')  # Use current directory for template
env = Environment(loader=FileSystemLoader(template_dir))

# Load the Jinja2 template
template = env.get_template('compose_50.yml')

# Render the template with any necessary variables (if needed)
output = template.render(FLWR_VERSION='1.17.0')

# Write the rendered output to a new file
with open('docker-compose-50.yml', 'w') as f:
    f.write(output)

print("docker-compose.yml file generated successfully!")