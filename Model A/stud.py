import streamlit as st
import pandas as pd
import pickle
#import statsmodels.api as sm


st.title('STUDENT PERFORMANCE INDEX PREDICTOR')

st.caption('This model was built on the dataset below, to help predict possible performance outcome' \
' of students.')
st.caption("1. Hours Studied: The numbers of hours spent on studying")
st.caption("2. Previous Scores: The previous examination score")
st.caption("3. Extracurricular Activities: Yes/No, showing if the student to part in extracurricular activities"
           )
st.caption("4. Sleep Hours: The numbers of hours spent on sleeping")
st.caption("5. Sample Question Papers Practiced: The numbers of sample questions attempted by the student ")
st.caption("6. Performance Index: The Independent variable, The ratio to how possible the student can attain success in the coming examination.")


df = pd.read_csv('Student_Performance.csv')
st.dataframe(df)




with open ('Lperformance.pkl', 'rb') as lm_pick:
    lme = pickle.load(lm_pick)


with open('label.pkl', 'rb') as label:
    labeler = pickle.load(label)






st.header("PREDICTION TAB")
st.caption("Input the variables to make the prediction")

with st.form('Form'):
    hours = st.number_input("Hours Studied")
    previous = st.number_input('Previous Score')
    extra = st.selectbox("Extracurricular Activities", ('Yes', 'No'))
    sleep = st.number_input('Sleep Hours')
    sample = st.number_input("Sample Question Papers Practiced")
    L_submitted = st.form_submit_button('Model A')
    


if L_submitted:
    features = pd.DataFrame({
        'Hours_studied' : [hours],
        'Previous_Score' : [previous],
        'Extracurricular_activities' : [extra],
        'Sleep_Hours' : [sleep],
        'Practiced_Questions' : [sample]
    })
    features['Extracurricular_activities'] = labeler.transform(features['Extracurricular_activities'])
    '''reL = {}
    for i in features.columns:
        reL[i] = labeler.transform(features[i])
    '''
    #sleR = pd.DataFrame(reL)
    prediction = lme.predict(features)
    st.write(f'Predicted Performnce Index:  {prediction[0]}')



