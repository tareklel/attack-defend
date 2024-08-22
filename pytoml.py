def convert_requirements_to_poetry(req_file, poetry_file):
    with open(req_file, "r") as reqs, open(poetry_file, "a") as poetry:
        poetry.write("[tool.poetry.dependencies]\n")
        poetry.write('python = "^3.8"\n')  # Adjust Python version as necessary

        for line in reqs:
            line = line.strip()
            if line and not line.startswith("#"):
                package, version = line.split("==")
                poetry.write(f'{package} = "{version}"\n')


# Example usage
convert_requirements_to_poetry("requirements.txt", "pyproject.toml")