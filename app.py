import streamlit as st
import requests

st.title("Benchmark Tool")

# API endpoint URL
url = "http://localhost:8000"

# Function to fetch problems from the API
def fetch_problems():
    response = requests.get(f"{url}/problems")
    return response.json()

# Function to fetch models from the API
def fetch_models():
    response = requests.get(f"{url}/models")
    return response.json()

# Function to generate prompt for a problem
def generate_prompt(problem_name):
    response = requests.get(f"{url}/generate_prompt/{problem_name}")
    return response.json()["prompt"]

# Function to score code for a problem
def score_code(problem_name, code):
    response = requests.post(f"{url}/score_code/{problem_name}", json={"code": code})
    return response.json()

# Main app logic
problems = fetch_problems()
models = fetch_models()

selected_problem = st.selectbox("Select a problem", [problem["name"] for problem in problems])
selected_model = st.selectbox("Select a model", [model["name"] for model in models["data"]])

if st.button("Generate Prompt"):
    prompt = generate_prompt(selected_problem)
    st.write(prompt)

code = st.text_area("Enter your code here")

if st.button("Score Code"):
    score_result = score_code(selected_problem, code)
    st.write("Score:", score_result["score"])
    st.write("Code Quality:", score_result["code_quality"])
