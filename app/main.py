"""Streamlit entry point: `streamlit run app/main.py`"""

import streamlit as st

from app.core import (
    DEFAULT_MODEL_DIR,
    classify_paragraphs,
)


@st.cache_resource
def get_classifier():
    from app.core import load_classifier

    return load_classifier(DEFAULT_MODEL_DIR)


st.set_page_config(page_title="HR Policy Document Classifier")
st.title("HR Policy Document Classifier")
st.write("HR administrators use this tool to automatically classify internal policy documents (e.g., \"leave policy\", \"expense claim\") for quick retrieval and departmental routing.")

uploaded = st.file_uploader("Upload an HR policy .txt file", type=["txt"])

if uploaded is not None:
    try:
        text = uploaded.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("The uploaded file must contain UTF-8 text.")
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
    else:
        if not text.strip():
            st.warning("The uploaded file is empty.")
        else:
            try:
                tokenizer, model = get_classifier()
                results = classify_paragraphs(text, tokenizer, model)
                for result in results:
                    st.markdown(f"**{result['category']}**")
                    st.write(result["text"])
            except FileNotFoundError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Classification failed: {exc}")
