import openai

# Setup OpenAI API key
openai.api_key = "your api_key"

def generate_diagram_code(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        generated_code = response['choices'][0]['message']['content'].strip()
        return generated_code
    except Exception as e:
        return f"Error: {e}"
