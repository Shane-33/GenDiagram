# diagram_app.py
import nltk
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graphviz import Source
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import altair as alt
from gpt4_integration import generate_diagram_code

# Additional necessary imports
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu_score
from rouge_score import rouge_scorer
import difflib
import re
import seaborn as sns
import ast
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Download NLTK data
nltk.download('punkt')

# Setup for Bokeh
output_notebook()


# Helper functions for diagram relevance validation
def semantic_similarity(reference, candidate):
    doc_ref = nlp(reference)
    doc_cand = nlp(candidate)
    return doc_ref.similarity(doc_cand)

def compute_bleu_score(reference, candidate):
    return nltk.translate.bleu_score.sentence_bleu([reference.split()], candidate.split())

def compute_rouge_score(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure

def token_similarity(prompt, generated_code):
    prompt_tokens = set(word_tokenize(prompt.lower()))
    generated_tokens = set(word_tokenize(generated_code.lower()))
    common_tokens = prompt_tokens.intersection(generated_tokens)
    similarity = len(common_tokens) / (len(prompt_tokens) + len(generated_tokens) - len(common_tokens))
    return similarity

def tfidf_similarity(prompt, generated_code):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([prompt, generated_code])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def sequence_similarity(prompt, generated_code):
    similarity = difflib.SequenceMatcher(None, prompt, generated_code).ratio()
    return similarity

def refined_validate_diagram_relevance(prompt, generated_code):
    bleu_score = compute_bleu_score(prompt, generated_code)
    rouge_score = compute_rouge_score(prompt, generated_code)
    sem_sim = semantic_similarity(prompt, generated_code)
    
    len_ratio = len(generated_code) / len(prompt)
    overall_similarity = ((bleu_score + rouge_score) * len_ratio + sem_sim) / 3.0
    
    if overall_similarity > 0.5:
        return True, bleu_score, rouge_score, sem_sim, overall_similarity
    else:
        return False, bleu_score, rouge_score, sem_sim, overall_similarity


# UI functions
def gpt4_diagram_interface(api_key):
    prompt = st.text_area("Enter your prompt for GPT-4:", height=150)
    if st.button("Generate Diagram Code"):
        try:
            generated_code = generate_diagram_code(api_key, prompt)
            
            is_relevant, bleu_score, rouge_score, sem_sim, overall_similarity = refined_validate_diagram_relevance(prompt, generated_code)
            
            st.text_area("Generated Code:", height=300, value=generated_code)
            st.text(f"BLEU Score: {bleu_score:.4f}")
            st.text(f"ROUGE Score: {rouge_score:.4f}")
            st.text(f"Semantic Similarity: {sem_sim:.4f}")
            st.text(f"Overall Similarity: {overall_similarity:.4f}")
            
            if is_relevant:
                st.text("Code Accuracy: Yes")
            else:
                st.text("Code Accuracy: No")
                
            scores = [bleu_score, rouge_score, sem_sim, overall_similarity]
            labels = ['BLEU Score', 'ROUGE Score', 'Semantic Similarity', 'Overall Similarity']
            
            plt.figure(figsize=(10, 5))
            sns.barplot(x=labels, y=scores)
            plt.title('Refined Evaluation Scores')
            plt.ylim(0, 1)
            st.pyplot(plt)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Streamlit UI configuration and styling
st.set_page_config(layout="wide", page_title="AI Diagram and Plot Creator", page_icon="📊")
st.markdown("""
    <style>
        body { 
            color: #fff;
            background-color: #0e1117;
        }
        .st-bw {
            background-color: #161b22;
        }
        .st-b3 {
            padding: 0.25rem 1rem;
        }
        button {
            border-radius: 20px;
            border: 1px solid #1f6feb;
            background-color: #1f6feb;
            color: #fff;
            padding: 8px 16px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# Main content
st.title("AI Diagram and Plot Creator")
st.subheader("Create various diagrams and plots effortlessly using multiple libraries.")

# Add this section for entering the OpenAI API key
st.sidebar.subheader("OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

# Sidebar navigation
nav_options = ["GPT-4 Diagram Generation", "Documentation"]
selected_nav = st.sidebar.radio("Navigation", nav_options)

if selected_nav == "GPT-4 Diagram Generation":
    gpt4_diagram_interface(api_key)  # Pass the api_key argument here
elif selected_nav == "Documentation":
    st.sidebar.title("About")
    st.sidebar.info("This app allows you to create diagrams and plots using various libraries and tools. Select the type of diagram or plot from the dropdown menu, provide the necessary code, and generate visualizations.")
    st.sidebar.header("GPT-4 Diagram Generation")
    st.sidebar.markdown("Use this section to generate diagram code based on your prompt and evaluate its relevance.")


# Example codes with added diagram types
example_codes = {
    "Mermaid": {
        "Sequence Diagram": "sequenceDiagram\n    Alice->>+John: Hello John, how are you?\n    John-->>-Alice: Great!",
        "Flowchart": "graph TD;\n    A-->B;\n    A-->C;\n    B-->D;\n    C-->D;",
        "Gantt Chart": "gantt\ntitle A Gantt Diagram\ndateFormat  YYYY-MM-DD\nsection Section\nA task           :a1, 2014-01-01, 30d\nAnother task     :after a1  , 20d"
    },
    "DOT": {
        "Directed Graph": "digraph G {\n    A -> B\n    B -> C\n    C -> D\n    D -> A\n}",
        "Undirected Graph": "graph G {\n    A -- B\n    B -- C\n    C -- D\n    D -- A\n}",
        "Graph with Attributes": "graph G {\n    node [shape=circle]\n    A -- B\n    B -- C\n    C -- D\n    D -- A\n}"
    },
    "Matplotlib": {
        "Sine Wave": 'np.sin(x)',
        "Cosine Wave": 'np.cos(x)',
        "Tangent Wave": 'np.tan(x)'
    },
    "NetworkX": {
    "Simple Network": "G = nx.barabasi_albert_graph(20, 2)\npos = nx.spring_layout(G)\nnx.draw_networkx(G, pos=pos, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, alpha=0.5)", 
    "Directed Network": "G = nx.DiGraph()\nG.add_edge('A', 'B')\nG.add_edge('A', 'C')\nG.add_edge('B', 'C')\nG.add_edge('C', 'A')\npos = nx.spring_layout(G)\nnx.draw_networkx(G, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, alpha=0.5)",
    "Weighted Network": "G = nx.Graph()\nG.add_edge('A', 'B', weight=0.6)\nG.add_edge('A', 'C', weight=0.2)\nG.add_edge('C', 'D', weight=0.1)\nG.add_edge('D', 'A', weight=0.7)\npos = nx.spring_layout(G)\nnx.draw_networkx(G, pos, with_labels=True, edge_color='k', node_size=500, edge_cmap=plt.cm.Blues)"
    },

    "Bokeh": {
        "Simple Line Plot": "p = figure(title='Simple Line Plot', x_axis_label='x', y_axis_label='y')\np.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2)\nshow(p)",
        "Scatter Plot": "p = figure(title='Scatter Plot', x_axis_label='x', y_axis_label='y')\np.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=15, color='navy', alpha=0.5)\nshow(p)"
    },
    "Altair": {
        "Simple Bar Chart": "alt.Chart(alt.Data(values=[{'x': 'A', 'y': 5}, {'x': 'B', 'y': 3}, {'x': 'C', 'y': 6}])).mark_bar().encode(x='x:N', y='y:Q')",
        "Line Chart": "alt.Chart(alt.Data(values=[{'x': 1, 'y': 5}, {'x': 2, 'y': 3}, {'x': 3, 'y': 6}, {'x': 4, 'y': 7}])).mark_line().encode(x='x:Q', y='y:Q')"
    }
}




documentation_links = {
    "Mermaid": "[Mermaid Documentation](https://mermaid-js.github.io/mermaid/#/)",
    "DOT": "[Graphviz Documentation](https://graphviz.org/documentation/)",
    "Matplotlib": "[Matplotlib Documentation](https://matplotlib.org/stable/contents.html)",
    "NetworkX": "[NetworkX Documentation](https://networkx.org/documentation/stable/)",
    "Bokeh": "[Bokeh Documentation](https://docs.bokeh.org/en/latest/)",
    "Altair": "[Altair Documentation](https://altair-viz.github.io/)"
}



# Diagram type selection
diagram_type = st.selectbox("Type", ["Mermaid", "DOT", "Matplotlib", "NetworkX", "Bokeh", "Altair"])


if diagram_type in documentation_links:
    st.sidebar.markdown(documentation_links[diagram_type])

# Handle Mermaid diagrams
if diagram_type == "Mermaid":
    mermaid_code = st.text_area("Mermaid code", height=200, value=example_codes["Mermaid"]["Flowchart"])
    
    # Calculate the number of lines in the Mermaid code
    num_lines = mermaid_code.count('\n') + 1  # Counting the number of new lines and adding one for the last line
    
    # Set the height dynamically based on the number of lines
    height = 120 * num_lines  # Adjust this multiplier as needed to fit your content
    
    if mermaid_code:
        # Embed Mermaid diagram directly using components.html
        st.components.v1.html(f"""
        <div class="mermaid">
        {mermaid_code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
        mermaid.initialize({{startOnLoad:true}});
        </script>
        """, height=height)

elif diagram_type == "DOT":
    dot_code = st.text_area("DOT code", height=200, value=example_codes["DOT"]["Directed Graph"])
    if dot_code:
        try:
            # Display DOT graph using Graphviz
            st.graphviz_chart(dot_code)
        except Exception as e:
            st.error(f"Error: {e}")

elif diagram_type == "Matplotlib":
    function_input = st.selectbox("Function", options=list(example_codes["Matplotlib"].values()))
    x = np.linspace(-10, 10, 400)
    try:
        y = eval(function_input)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to generate the plot. Error: {str(e)}")
        st.info("Please ensure the function is defined correctly. For example, use 'np.sin(x)' for a sine wave.")


elif diagram_type == "NetworkX":
    nx_code = st.text_area("NetworkX code", height=200, value=next(iter(example_codes["NetworkX"].values())))
    if st.button('Generate NetworkX Diagram'):
        # Clear previous matplotlib figures to prevent accumulation in Streamlit
        plt.clf()  # Clear figure
        plt.cla()  # Clear axes
        plt.close()  # Close a figure window

        # Execute the NetworkX code
        # Create a new figure and axes explicitly for the plot
        fig, ax = plt.subplots()
        
        # Create a sample graph
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])
        
        # Compute the spring layout
        pos = nx.spring_layout(G)
        
        # Draw the graph using spring layout
        nx.draw(G, pos, with_labels=True, ax=ax)
        
        # Modify the exec environment to include the newly created fig and ax
        local_vars = {"plt": plt, "fig": fig, "ax": ax, "nx": nx, "pos": pos}
        exec(nx_code, globals(), local_vars)
        
        # Use st.pyplot to display the figure
        st.pyplot(fig)



elif diagram_type == "Bokeh":
    bokeh_code = st.text_area("Bokeh code", height=200, value=next(iter(example_codes["Bokeh"].values())).split("\nshow(p)")[0])
    if st.button('Generate Bokeh Plot'):
        # Execute the Bokeh plot code
        exec(bokeh_code)
        p = eval(bokeh_code.split('=')[0].strip())
        st.bokeh_chart(p, use_container_width=True)

elif diagram_type == "Altair":
    altair_code = st.text_area("Altair code", height=200, value=next(iter(example_codes["Altair"].values())))
    if st.button('Generate Altair Chart'):
        # Evaluate and display the Altair chart
        chart = eval(altair_code)
        st.altair_chart(chart, use_container_width=True)


# Display examples in the sidebar
for example_type, examples in example_codes.items():
    if diagram_type == example_type:
        for title, code in examples.items():
            st.sidebar.write(f"**{title}**")
            st.sidebar.code(code)

# Implementing tooltips and modal for help
with st.sidebar:
    if st.button("Help"):
        st.markdown("Here's how to use the application effectively...")
