import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space


def initialize():
    st.logo("assets/Logo.png")
    sidebar()


# ---- Sidebar ---- #

# region


def sidebar():
    side_bar = st.sidebar

    with side_bar:
        add_vertical_space(30)

    side_bar.divider()
    side_bar.link_button(
        "Repository",
        type="secondary",
        icon=":material/star:",
        use_container_width=True,
        url="https://github.com/Jhojan00/Bynar1Forge",
    )
    side_bar.link_button(
        "Github",
        type="secondary",
        url="https://github.com/Jhojan00",
        icon=":material/favorite:",
        use_container_width=True,
    )
    side_bar.link_button(
        "LinkedIn",
        type="secondary",
        url="https://www.linkedin.com/in/jhojan-alfredo-aguilera-sanchez-60480a303/",
        icon=":material/business_center:",
        use_container_width=True,
    )


# endregion
