import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graphviz import Source
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import altair as alt


# Setup for Bokeh
output_notebook()

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
        "Simple Network": "G = nx.fast_gnp_random_graph(5, 0.5)\npos = nx.spring_layout(G)\nnx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, alpha=0.5)",
        "Directed Network": "G = nx.DiGraph()\nG.add_edge('A', 'B')\nG.add_edge('A', 'C')\nG.add_edge('B', 'C')\nG.add_edge('C', 'A')\npos = nx.spring_layout(G)\nnx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, alpha=0.5)"
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
st.title("Diagram and Plot Creator")
st.header("Select the type and enter the definition or expression:")

# Diagram type selection
diagram_type = st.selectbox("Type", ["Mermaid", "DOT", "Matplotlib", "NetworkX", "Bokeh", "Altair"])

# Sidebar content and documentation links
st.sidebar.title("About")
st.sidebar.text("This app allows creating diagrams and plots using various libraries and tools.")

documentation_links = {
    "Mermaid": "[Mermaid Documentation](https://mermaid-js.github.io/mermaid/#/)",
    "DOT": "[Graphviz Documentation](https://graphviz.org/documentation/)",
    "Matplotlib": "[Matplotlib Documentation](https://matplotlib.org/stable/contents.html)",
    "NetworkX": "[NetworkX Documentation](https://networkx.org/documentation/stable/)",
    "Bokeh": "[Bokeh Documentation](https://docs.bokeh.org/en/latest/)",
    "Altair": "[Altair Documentation](https://altair-viz.github.io/)"
}

if diagram_type in documentation_links:
    st.sidebar.markdown(documentation_links[diagram_type])

# Handle each diagram type
if diagram_type == "Mermaid":
    mermaid_code = st.text_area("Mermaid code", height=200, value=example_codes["Mermaid"]["Flowchart"])
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
        """, height=300)
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
    try:
        x = np.linspace(-10, 10, 400)
        y = eval(function_input)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error: {e}")

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
        # Modify the exec environment to include the newly created fig and ax
        local_vars = {"plt": plt, "fig": fig, "ax": ax}
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





