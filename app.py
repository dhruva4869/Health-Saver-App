from dotenv import load_dotenv
import streamlit as st
import os
import time
import pickle
import textwrap
from IPython.display import Markdown
import google.generativeai as genai
from streamlit_option_menu import option_menu
import os
import PyPDF2
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration,T5Tokenizer
from fpdf import FPDF


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model=genai.GenerativeModel("gemini-pro") 


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

chat = model.start_chat(history=[])

def get_gemini_response(question):
    response=chat.send_message(question,stream=True)
    return response




st.set_page_config (
    page_title="Health Saver",
    layout="wide",
    page_icon="💓",
    initial_sidebar_state="expanded"
)


working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f'{working_dir}/Health Saver/code-files/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/Health Saver/code-files/heart_disease_model.sav', 'rb'))

st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content .stButton button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content .stRadio label {
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Explore")
with st.sidebar:
        page = option_menu(
        menu_title='Health Saver',
        options=['Homepage', 'Diabetes Prediction', 'Heart Disease Prediction', 'Other Disease Predictors', 
                    'AI Chatbot', 'Medical Report Summarizer', 'Medical Video Summarizer'],
        icons=['house', 'activity', 'heart', 'activity', 'robot', 'book', 'youtube'],
        menu_icon='cast',
        default_index=0,
        orientation="vertical"
    )
if page == "Homepage":
    st.title("🌟 Welcome to the Health Saver App")
    st.write("Use the sidebar to navigate between different functionalities.")
    st.image("https://wallpapercave.com/wp/wp12234331.jpg", caption="Health Assistant & Q&A Application")
    st.write("""
        The "Health Saver Web application" is a groundbreaking healthcare tool that predicts the likelihood of diseases 
             like diabetes and Heart diseases, enabling users to take proactive measures for better health. 
             Additionally, it provides a chatbot that users can use to ask the Google Gemini AI, about any possible
             doubts and concerns regarding their heatlh.
        By leveraging advanced technology and analyzing personal health data, and 
             medical history, the Health Saver web application empowers users to make informed choices for disease 
             prevention and management.
    """)
    st.markdown("""
        ### Available Features
        - **Diabetes Prediction 💹**: Predict the likelihood of diabetes based on various health metrics.
        - **Heart Disease Prediction 💖**: Assess the risk of heart disease using medical parameters.
        - **Other Disease Predictors**: Check the predictors for many other diseases like Autism and Kidney on our Github Link.
        - **AI Chatbot 🤖**: Get answers to your questions powered by Google Gemini.
    """)

elif page == "Diabetes Prediction":
    st.title('Diabetes Prediction using ML')
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level (Plasma glucose concentration a 2 hours in an oral glucose tolerance test)')
    with col1:
        BloodPressure = st.text_input('Blood Pressure value (Diastolic blood pressure (mm Hg))')
    with col2:
        SkinThickness = st.text_input('Skin Thickness value (Triceps skin fold thickness (mm))')
    with col1:
        Insulin = st.text_input('Insulin Level (2-Hour serum insulin (mu U/ml))')
    with col2:
        BMI = st.text_input('BMI value')
    with col1:
        Age = st.text_input('Age of the Person')
    
    diab_diagnosis = ''
        
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "0.5", Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    if diab_diagnosis: st.success(diab_diagnosis) 
    else: st.error(diab_diagnosis)


elif page == "Heart Disease Prediction":
    st.title('Heart Disease Prediction using ML')
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input('Age')
        sex = st.selectbox(
                'Sex',
                [(0, 'Female'), (1, 'Male')],
                format_func=lambda x: x[1],
                index=0
            )[0]  
        cp = st.selectbox(
                'Chest Pain (Category 0, 1, 2, 3 types)',
                [(0, 'Typical angina'), (1, 'Atypical angina'), (2, 'Non-anginal pain'), (3, 'Asymptomatic')],
                format_func=lambda x: x[1],
                index=0
            )[0]  
        trestbps = st.text_input('Resting Blood Pressure')
        fbs = st.selectbox(
                'Fasting Blood Sugar > 120 mg/dl',
                [(0, 'No'), (1, 'Yes')],
                format_func=lambda x: x[1],
                index=0
            )[0]  

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        thal = st.text_input('Inherited blood disorder? (0 = normal; 1 = fixed defect; 2 = reversible defect)')

        show_advanced = st.checkbox('Show advanced parameters')
        if show_advanced:
            chol = st.text_input('Serum Cholesterol in mg/dl', value='200')
            restecg = st.selectbox(
                'Resting Electrocardiographic results',
                [(0, 'Normal'), (1, 'ST-T wave abnormality'), (2, 'Left ventricular hypertrophy by the criteria of Estes')],
                format_func=lambda x: x[1],
                index=0
            )[0]  
            exang = st.selectbox(
                'Exercise Induced Angina',
                [(0, 'No'), (1, 'Yes')],
                format_func=lambda x: x[1],
                index=0
            )[0]  
            oldpeak = st.text_input('ST depression induced by exercise', value='1.8')
            slope = st.selectbox(
                'Slope of the peak exercise ST segment',
                [(0, 'Upslope (Going up)'), (1, 'Flat'), (2, 'Downslope (Going down)')],
                format_func=lambda x: x[1],
                index=0
            )[0]  
            ca = st.text_input('Major vessels colored by fluoroscopy (0, 1, 2, 3)', value='0')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        age_numeric = float(age)
        sex_numeric = int(sex)
        cp_numeric = int(cp)
        trestbps_numeric = float(trestbps)
        fbs_numeric = int(fbs)
        thalach_numeric = float(thalach)
        thal_numeric = int(thal)
        if show_advanced:
            chol_numeric = float(chol)
            restecg_numeric = float(restecg)
            exang_numeric = float(exang)
            oldpeak_numeric = float(oldpeak)
            slope_numeric = float(slope)
            ca_numeric = float(ca)
            input_data = [age_numeric, sex_numeric, cp_numeric, trestbps_numeric, chol_numeric, fbs_numeric, restecg_numeric, thalach_numeric, exang_numeric, oldpeak_numeric, slope_numeric, ca_numeric, thal_numeric]
        else:
            input_data = [age_numeric, sex_numeric, cp_numeric, trestbps_numeric, 250.0, fbs_numeric, 1.0, thalach_numeric, 1.0, 1.8, 1.0, 0.0, thal_numeric]
        heart_prediction = heart_disease_model.predict([input_data])
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is predicted to have heart disease'
        else:
            heart_diagnosis = 'The person is predicted to not have heart disease'

    if heart_diagnosis:
        st.success(heart_diagnosis)
    else:
        st.error('Please input all required fields.')

elif page == "Other Disease Predictors":
    st.header("Website part under development")
    st.write("For the code can visit the following link")
    st.write("Thank you for your understanding")
    st.link_button(label="Autism Predictor", url="https://github.com/dhruva4869", help="Under development")
    st.link_button(label="Kidney Disease Predictor", url="https://github.com/dhruva4869", help="Under development")

elif page == "AI Chatbot":
    st.header("MedBot 🤖")
    st.write("Your personal AI Medical Chatbot, fetching the answers to your queries, instantly with the help of Google Gemini")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    input=st.text_input("Input: ",key="input")
    submit=st.button("Ask the question")

    if submit and input:
        response=get_gemini_response(input)
        st.session_state['chat_history'].append(("You", input))
        for chunk in response:
            st.session_state['chat_history'].append(("Bot", chunk.text))
    st.subheader("The Chat History is: ")
    with st.spinner('Loading..'):
        time.sleep(2)
    for role, text in st.session_state['chat_history']:
        isCode = False
        if role == "You":
            st.markdown(f'<div style=" overflow:scroll; float: right; clear: both; margin: 1em; padding: 1em; background-color: #0047ab; border-radius: 10px; animation: fadeIn 0.5s ease; ">User: {text}</div>', unsafe_allow_html=True)
            isCode = any(word in text for word in ("code", "python" "C++", "Python", "Code"))
        elif role == "Bot":
            
            if isCode:
                st.code({text})
            else:
                st.markdown(f'<div style="overflow:scroll; float: left; clear: both; margin: 1em; padding: 1em; background-color: #3ded97; border-radius: 10px; animation: fadeIn 0.5s ease; color: black;">AI: {text}</div>', unsafe_allow_html=True)

elif page == "Medical Report Summarizer":
    modelP = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    st.header("Medical Report Summarizer App")
    st.write("Upload a PDF document and get a summarized version of its content.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    

    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        if text:
            # Tokenize and summarize the text
            inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1000, truncation=True)
            outputs = modelP.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.text_area("Summarized Text", summary, height=150)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            pdf.multi_cell(0, 10, summary)

            pdf_output = pdf.output(dest='S').encode('latin1')

            st.download_button(
                label="Export Report",
                data=pdf_output,
                file_name="Report.pdf",
                mime='application/pdf'
            )
        else:
            st.warning("Could not extract text from the PDF.")
    else:
        st.info("Please upload a PDF file.")


elif page == "Medical Video Summarizer":
    modelP = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    st.title("Medical Video Summarizer (English)")
    st.write("Summarizes not only Medical videos, but any normal videos in general. Here try it out yourself")
    
    video_id = st.text_input("Enter a video link")
    id_ = video_id.split("watch?v=")
    with st.spinner('Fetching and processing transcript...'):
        complete_transcript = ""
        try:
            if id_ and video_id: 
                transcript_ = YouTubeTranscriptApi.get_transcript(id_[-1])
                complete_transcript = ""
                # st.write(transcript_)
                N = len(transcript_)
                for i in range(N):
                    complete_transcript += (transcript_[i]["text"]+" ")
                # st.write(len(transcript_))
                # st.write(complete_transcript)
        except Exception as e:
            st.write("Subtitles are not available at the moment for this video. We are sorry for the inconvenience")

        
        if complete_transcript:
            def split_into_chunks(text, max_length=512):
                return [text[i:i + max_length] for i in range(0, len(text), max_length)]
            
            text_chunks = split_into_chunks(complete_transcript)
            summaries = []
            for chunk in text_chunks:
                inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", truncation=True)
                outputs = modelP.generate(
                    inputs, 
                    max_length=150,
                    length_penalty=3.0,
                    num_beams=6,
                    early_stopping=True
                )
                chunk_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                summaries.append(chunk_summary)
            
            full_summary = "\n\n".join(summaries)

            st.text_area("Summarized Text", full_summary, height=300)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, full_summary)

            pdf_output = pdf.output(dest='S').encode('latin1')

            st.download_button(
                label="Export Report",
                data=pdf_output,
                file_name="Report.pdf",
                mime='application/pdf'
            )
        else:
            st.error("Enter a valid URL or Text")

            

st.markdown("""
    <style>
    .reportview-container .main footer {visibility: hidden;}
    .footer {position: fixed; left: 0; bottom: 0; width: 100%; background-color: #121212; text-align: center; padding: 10px; font-size: small; color: #ff47ab;}
    </style>
    <div class="footer">
    <p>© 2024 Dhruva. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
