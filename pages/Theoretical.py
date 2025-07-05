import streamlit as st

# ---- Global Widgets ---- #
from ml_backend.global_widgets import initialize


# ---- Explanation ---- #

# region


def explanation():
    # Title
    st.title(":red[Theoretical part] <Working on this>")

    # Theoretical part

    st.markdown(
        r"""
    These aspects are technical, and I only recommend reading
    if you have a basic understanding of neural networks. However, 
    it is not necessary to be a professional to understand the concepts presented here.

    In this project, I have utilized several statistical methods. Below are some of them.  
    The explanations are short and are references to better understand 
    the techniques I have implemented.
    If you want a better understanding you can see the project in the GitHub repository.

    ### Logistic Regression

    Logistic regression is a statistical method
    used to determine outcomes where the output variable is binary, for example,
    `True` or `False`, `1` or `0`. It estimates the probability
    that a given input is True or False.

    The function behind logistic regression is the
    sigmoid function, which converts the output 
    between 0 and 1, making it perfect for probabilities.

    $$
    P(y = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
    $$

    $$
    \hat{y} = \sigma(W^\top X + b), \quad \text{where } \sigma(z) = \frac{1}{1 + e^{-z}}
    $$

    For example, if our inputs are `1` and `0`,
    and we want to do the logical (XOR) operation,
    logistic regression can be trained to approximate the result.

    If you want to explore more about Boolean operators, visit [this resource](https://www.codecademy.com/resources/blog/what-is-boolean-logic/).

    Each neuron in this neural network is implemented using 
    the sigmoid activation function and can be trained to 
    predict the probability associated with specific input data.

    ### Cost Function

    The cost function is a fundamental component in logistic regression. 
    It allows us to evaluate how well the model is doing it. If the cost `J` is large,
    it indicates that the model is not learning.

    $y^{(i)}$ represents the true output (label), and $\hat{y}^{(i)}$ denotes
    the predicted output from the neural network. 
    $$
    J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
    $$

    Where:

    - $m$ is the number of training samples.
    - $y^{(i)}$ is the actual label for sample $i$.
    - $\hat{y}^{(i)}$ is the predicted probability for sample $i$.

    ---
    """
    )


# endregion

# ---- Code ---- #


# region
def code():
    st.markdown("""
## :green[Code]
Below you can explore the model implementation in Python, Rust, and as mathematical notation.

""")

    python_tab, rust_tab, math_tab = st.tabs(["Python", "Rust", "Math"])

    with open("assets/sample_code/model.py", "r") as f:
        python_code = f.read()

    python_tab.code(
        python_code,
        language="python",
    )

    with open("assets/sample_code/model.rs", "r") as f:
        rust_code = f.read()

    rust_tab.code(
        rust_code,
        language="rust",
    )

    with open("assets/sample_code/model.md", "r") as f:
        math_content = f.read()

    math_tab.markdown(math_content)


# endregion


# Draw page

explanation()
code()
