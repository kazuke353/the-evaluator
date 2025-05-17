import random
from jinja2 import Template

# Define prompt templates
prompt_templates = [
    Template("Solve the following problem: {{ problem_description }}"),
    Template("Write a function to {{ problem_description }}"),
    Template("Implement a solution for: {{ problem_description }}"),
]

def generate_prompt(problem_description: str) -> str:
    template = random.choice(prompt_templates)
    return template.render(problem_description=problem_description)

if __name__ == "__main__":
    problem_description = "calculate the factorial of a number"
    print(generate_prompt(problem_description))
