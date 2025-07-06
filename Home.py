import streamlit as st
import streamlit_extras as stx
from pandas import read_csv
from pandas import DataFrame
import numpy as np

# ----  Backend ---- #
from ml_backend.data_handler import process_file
import ml_backend.draw_graph as dg
from ml_backend.model import BinaryNeuronalNetwork
from ml_backend.open_ai_chat import ask_open_ai, add_message

# ---- Global Widgets ---- #
from ml_backend.global_widgets import initialize


# ---- Persistent vars --- #

persistent_vars = [
    "train_data",
    "test_data",
    "neurons_like",
    "hyperparameters",
    "model",
    "messages",
    "messages_ai",
]

for key in persistent_vars:
    if key not in st.session_state:
        st.session_state[key] = None


persistent_boolean_vars = ["is_training", "ai_is_thinking"]

for key in persistent_boolean_vars:
    if key not in st.session_state:
        st.session_state[key] = False


# ---- Functions ---- #


def get_neuron_chart(neurons, layers):
    node_list = dg.get_nodes(neurons, layers)
    graph = dg.get_graph(node_list)
    chart = dg.draw_graph(graph)

    st.session_state.neurons_like = [
        len(n) for n in node_list
    ]  # Save the NN's architecture

    return chart


def ask_ai(prompt: str, context: tuple):
    return ask_open_ai(
        {"role": "user", "content": f"Prompt: {prompt}\n Context: {context}"}
    )


# ---- Introduction ---- #

# region


def introduction():
    st.image("assets/Logo.png")
    st.write("")
    st.markdown(
        "Project made for easily handle **:rainbow[binary neuronal networks!]**"
    )

    st.divider()

    st.markdown("""
    ## :green[How to Use B1naryForge]
    Upload a `.csv` file with your inputs and expected outputs, select your hyperparameters, and start training.
                    """)

    st.expander("Data sample").dataframe(
        read_csv("sample_data/sample_training_data.csv")
    )

    st.markdown(
        """Each row represents an example, each column a feature 
        and the las column **:red[must]** be the expected output.
        """
    )


# endregion

# ---- Upload file ---- #

# region


def upload_file(key):
    file = st.file_uploader(
        "**Upload your File**", ".csv", accept_multiple_files=False, key=key
    )

    if file:
        result = process_file(file)

        if isinstance(result, str):
            st.error(result)
        else:
            df, X, Y = result
            st.success("File uploaded")
            st.expander("Data preview", True).dataframe(df.head())

            return X, Y

    return None


# endregion

# ---- Training ---- #

# region


def train_model():
    st.markdown("""
    ## :green[Upload your CSV]
    **Tips:**  
    - Make sure your .csv file has the expected output as the last column.  
    - Check that there are no empty or incorrect values in your data.  
    - If you have questions, see the example in "Data sample" above.
    
    """)

    st.session_state.train_data = upload_file(key="train_model_upload")

    st.divider()


# endregion


# ---- Chat ---- #


# region
def chat():
    with st.expander("**:green[Need some help?]** Ask our AI"):
        message_col, clear_col = st.columns([0.9, 0.1])
        message = message_col.chat_input(
            "Ask something", disabled=st.session_state.ai_is_thinking
        )
        button = clear_col.button(
            icon=":material/delete_sweep:",
            label="",
        )

        if button:
            st.session_state.messages = None
            st.session_state.messages_ai = None
        if message:
            add_message((message, "user"))
            st.session_state.ai_is_thinking = True
            ask_ai(message, st.session_state.hyperparameters)

        if st.session_state.messages:
            for msg, sender in st.session_state.messages:
                with st.chat_message(sender):
                    st.markdown(msg)


# endregion
# ---- Hyperparameters ---- #

# region


def hyperparameters():
    st.markdown("""
    ## :green[Hyperparameters]            
    Here you can adjust hyperparameters to train your model. Take care, because with
    a bad choice, your model may not learn well.  
    Below, you can see a graphic that helps you understand if the model is
    learning. If the cost didn't go down, try changing your hyperparameters.
    
    
                
                """)

    # Draw select sliders and show the NN's representation
    chat()
    learning_col, cycles_col, neurons_col, layers_col = st.columns(4)

    learning_rate = learning_col.select_slider(
        "Learning rate",
        options=[round(x, 3) for x in np.linspace(0.001, 0.5, num=20)],
        value=0.5,
    )

    cycles = cycles_col.select_slider(
        "Cycles",
        options=[int(x) for x in np.linspace(100, 50000, num=500)],
        value=2000,
    )

    neurons = neurons_col.select_slider(
        "Neurons (approximal)",
        options=[int(x) for x in np.linspace(3, 50, num=43)],
        help="This is an approximation and may contain errors due to structural limitations of the NN.",
        value=5,
    )

    layers = layers_col.select_slider(
        "Layers",
        options=[int(x) for x in np.linspace(2, 10, num=8)],
        value=2,
    )

    if layers > 6 and neurons > 20:
        st.warning(
            "This amount of neurons and layers could be fatal for training performance or may cause your browser to freeze."
        )

    st.plotly_chart(get_neuron_chart(neurons, layers))

    st.session_state.hyperparameters = (learning_rate, cycles, neurons, layers)


# endregion

# ---- Start training ---- #

# region


def start_training():
    _, button_col = st.columns([0.7, 0.3], vertical_alignment="center")
    cost_widget = st.empty()

    start_button = button_col.button(
        "Start training",
        use_container_width=True,
        disabled=True
        if st.session_state.train_data
        is None  # FIXME (It does not work, solution: Threads)
        else st.session_state.is_training,
    )

    if start_button:
        st.session_state.is_training = True
        st.session_state.model = BinaryNeuronalNetwork(
            X=st.session_state.train_data[0],
            Y=st.session_state.train_data[1],
            learning_rate=st.session_state.hyperparameters[0],
            cycles=st.session_state.hyperparameters[1],
            neurons_like=st.session_state.neurons_like,
            cost_widget=cost_widget,
        )
        st.session_state.model.train()
        st.session_state.is_training = False
        st.toast("Model trained", icon=":material/network_intelligence:")


# endregion


# ---- Test model ---- #

# region


def test_model():
    st.markdown("""
    ## :green[Test your model]
    
When uploading a CSV file to test your model, make sure that the file
does **:red[not]** include the expected output (target) column. 
The model will generate predictions for these inputs, 
so the last column should contain only input features, 
not the actual results you want to predict.
                
""")

    result = upload_file(key="test_model_upload")

    if result:
        X, Y = result
        st.session_state.test_data = np.vstack((X, Y))


# endregion


# ---- Test model button and results ---- #

# region


def test_button_results():
    _, button_col = st.columns([0.7, 0.3], vertical_alignment="center")

    test_button = button_col.button(
        "Start test",
        use_container_width=True,
        disabled=st.session_state.model is None or st.session_state.test_data is None,
    )

    if test_button:
        result = st.session_state.model.predict(st.session_state.test_data)

        st.dataframe(DataFrame({"Predictions": result.flatten()}))


# endregion


# ---- App's start ---- #
def main():
    initialize()
    introduction()
    train_model()
    hyperparameters()
    start_training()
    test_model()
    test_button_results()


if __name__ == "__main__":
    main()
    print("Run the project in the terminal with: streamlit run Home.py")
