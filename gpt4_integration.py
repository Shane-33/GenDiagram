import openai

# Setup OpenAI API key
openai.api_key = "your api_key"

def generate_diagram_code(prompt):
    try:
        system_prompt = ""
        with open('prompt.txt', 'r', encoding='utf-8') as file:
            # Read the entire content of the file
            system_prompt = file.read()

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Now given the user specification of a system, output the final intermediary language, mermaid, and matplotlib code:\n" + prompt + "\nTake a deep breath. Think step by step."}
            ]
        )
        generated_code = response['choices'][0]['message']['content'].strip()
        return generated_code
    except Exception as e:
        return f"Error: {e}"
