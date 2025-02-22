import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from io import BytesIO
from keras.models import load_model
import numpy as np

model = load_model('your_model2.keras')

def main():
    st.set_page_config(
        page_title="Sleep Apnea Severity Prediction",
        page_icon="🌙",
        layout="wide",
    )

    st.title("Sleep Apnea Severity Prediction")
    st.header("Enter Personal Details")
    name = st.text_input("Name:")
    age = st.number_input("Age:")
    
    gender = st.selectbox("Gender:", ["Male", "Female"])

    st.header("Upload ECG Data (TXT file)")
    uploaded_file = st.file_uploader("Choose a file", type="txt")


    if st.button("Predict and Generate PDF"):
        if not uploaded_file:
            st.warning("Please upload a valid TXT file.")
            return


        ecg_data = pd.read_csv(uploaded_file, header=None, names=["ECG"])
        input_data = preprocess_data(ecg_data)
        predictions = model.predict(input_data)

        prediction_result = "Moderate Sleep Apnea" if predictions[0, 0] > 0.5 else "Normal Sleep"

        st.header("ECG Plot")
        fig, ax = plt.subplots(figsize=(4,2.5))
        ax.plot(ecg_data["ECG"])
        ax.set_xlabel("Time", fontsize=5)
        ax.set_ylabel("ECG Signal",fontsize=5)
        ax.tick_params(axis='both', labelsize=3)  
        st.pyplot(fig)


        st.header("Prediction Result")
        st.success(f"{name}, based on the analysis, you have {prediction_result}.")

        # Generate PDF report
        st.header("Generate PDF Report")

        # Create PDF document
        pdf_filename = f"{name}_Sleep_Apnea_Report.pdf"
        pdf_stream = BytesIO()
        pdf = canvas.Canvas(pdf_stream)

        pdf.setFont("Helvetica", 12)
        pdf.drawString(100, 750, f"Name: {name}")
        pdf.drawString(100, 730, f"Age: {age}")
        pdf.drawString(100, 710, f"Gender: {gender}")

        pdf.drawString(100, 670, "ECG Plot")
        img_filename = "ecg_plot.png"
        plt.savefig(img_filename, bbox_inches="tight", pad_inches=0.1)
        pdf.drawInlineImage(img_filename, 100, 400, width=400, height=200)

        pdf.drawString(100, 350, "Prediction Result")
        pdf.drawString(100, 330, f"{name}, based on the analysis, you have {prediction_result}.")

        pdf.save()


        with open(pdf_filename, "wb") as f:
            f.write(pdf_stream.getvalue())

        st.success(f"PDF Report generated: [{pdf_filename}]")

def preprocess_data(ecg_data):


    if len(ecg_data) > 3000:
        ecg_data = ecg_data[:3000]
    elif len(ecg_data) < 3000:
        # Pad with zeros if the data is shorter than 3000
        ecg_data = np.pad(ecg_data, (0, 3000 - len(ecg_data)))
        
    return ecg_data.values.reshape(1, 3000, 1)

if __name__ == "__main__":
    main()
