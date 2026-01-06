"""Demo application for FED-MED system."""

import streamlit as st
import sys
sys.path.append('..')

from src.model.load_model import load_model_and_tokenizer
from src.model.inference import generate_response
from src.safety.guardrails import MedicalGuardrails


def main():
    """Main demo application."""
    st.title("🏥 FED-MED: Federated Medical AI")
    st.markdown("*Privacy-preserving medical AI trained across multiple hospitals*")
    
    # Initialize
    guardrails = MedicalGuardrails()
    
    # Sidebar
    st.sidebar.header("Settings")
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
    max_length = st.sidebar.slider("Max Length", 128, 1024, 512)
    
    # Main interface
    st.header("Medical Query Assistant")
    query = st.text_area(
        "Enter your medical question:",
        placeholder="E.g., What are the common symptoms of diabetes?",
        height=100
    )
    
    if st.button("Generate Response", type="primary"):
        if query:
            with st.spinner("Generating response..."):
                # TODO: Load model and generate
                response = "This is a placeholder response. Integrate the actual model here."
                
                # Apply safety checks
                safety_check = guardrails.check_response(response)
                
                if safety_check['is_safe']:
                    response = guardrails.add_disclaimer(response)
                    st.success("Response generated successfully")
                    st.markdown("### Response:")
                    st.write(response)
                else:
                    st.error("Response blocked by safety guardrails")
                    st.warning(f"Blocked patterns: {safety_check['blocked_patterns']}")
        else:
            st.warning("Please enter a question first.")
    
    # Information
    st.markdown("---")
    st.markdown("""
    ### About FED-MED
    This system uses federated learning to train medical AI models across multiple hospitals
    while preserving patient privacy. The model never sees raw patient data from any hospital.
    """)


if __name__ == "__main__":
    main()
