import streamlit as st

st.title("Test App")

st.markdown('<div class="edit-btn-anchor"></div>', unsafe_allow_html=True)
if st.button("✏️ Edit Last Query", key="edit_btn"):
    st.write("Clicked!")

st.markdown("""
<style>
div[data-testid="stMarkdownContainer"]:has(.edit-btn-anchor) {
    display: none;
}
div[data-testid="element-container"]:has(.edit-btn-anchor) + div[data-testid="element-container"] {
    position: fixed !important;
    bottom: 85px !important;
    right: 20px !important;
    z-index: 999 !important;
    width: auto !important;
}
</style>
""", unsafe_allow_html=True)

for i in range(10):
    st.write(f"Line {i}")
    
st.chat_input("Type here...")
