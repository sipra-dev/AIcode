# legal_AI_frontend.py

import os
import tempfile

import streamlit as st

from legalAI_backend import build_pipeline



# PAGE CONFIG


st.set_page_config(
    page_title="Legal AI Analyzer",
    layout="wide"
)


# TITLE


st.title("⚖️ Legal AI Analyzer")

st.write(
    "Upload any legal PDF document"
)


# FILE UPLOAD


uploaded_file = st.file_uploader(
    "Upload Legal PDF",
    type=["pdf"]
)


# ANALYSIS


if uploaded_file is not None:

    st.success(
        "PDF Uploaded Successfully"
    )

    if st.button("Analyze Document"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.read()
            )

            temp_pdf_path = tmp_file.name

        try:

            with st.spinner(
                "Analyzing Legal Document..."
            ):

                result = build_pipeline(
                    temp_pdf_path
                )

            st.success(
                "Analysis Completed"
            )

            
            # IPC
        

            st.markdown("## IPC Sections")

            st.write(
                result["ipc_sections"]
            )

            
            # INCIDENT
            

            st.markdown("## Incident")

            st.write(
                result["incident"]
            )

            
            # VICTIM AGE
            

            st.markdown("## Victim Age")

            st.write(
                result["victim_age"]
            )

            
            # PLACE
            

            st.markdown(
                "## Place Of Occurrence"
            )

            st.write(
                result[
                    "place_of_occurrence"
                ]
            )

            
            # SUMMARY
            

            st.markdown("## Summary")

            st.write(
                result["summary"]
            )

            
            # JUDGEMENT
            

            st.markdown("## Judgement")

            st.write(
                result["judgement"]
            )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

        finally:

            if os.path.exists(
                temp_pdf_path
            ):

                os.remove(
                    temp_pdf_path
                )
