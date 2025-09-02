import numpy as np
import streamlit as st 
import pandas as pd
import joblib 
import shap 
import matplotlib.pyplot as plt
st.title('Prediction of Cardiovascular Disease in Middle-Aged and Elderly Patients with Diabetes Mellitus')

st.info('''This application is built based on data from the China Health and Retirement Longitudinal Study (CHARLS)
and incorporates an XGBoost prediction model. It is designed to assess the risk probability of middle-aged and elderly
diabetic patients developing cardiovascular diseases.\n
To use the application, users can enter relevant clinical and health-related information via the input panel on the left. 
After clicking the "Predict" button, they will receive personalized results, including the predicted probability of developing
the disease and corresponding visualization outcomes.''')
df = pd.read_csv("X_test.csv")
model = joblib.load('XGB.pkl')


## 'srh', 'adlab_c', 'hibpe', 'lunge', 'dyslipe', 'kidneye', 'digeste',
##       'asthmae', 'memrye', 'mdact_c', 'hospital', 'retire', 'wrist_pain',
##       'chest_pain' 
with st.sidebar:
  st.header("patient-related information")

  
  hibpe = st.selectbox(
    "Has any doctor ever told you that you have hypertension?",
    (0, 1),
    index=0,
    placeholder='No = 0,Yes = 1',
  )

  
  lunge =  st.selectbox(
    "Have you been diagnosed with Chronic lung diseases, such as chronic bronchitis ,emphysema ( excluding tumors, or cancer) by a doctor",
    (0, 1),
    index=0,
    placeholder='No = 0,Yes = 1',
  )

  
  dyslipe = st.selectbox(
  '''Have you been diagnosed with Dyslipidemia (elevation of low density lipoprotein,
  triglycerides (TGs),and total cholesterol, or a low high density lipoprotein level) by
  a doctor?''',
    (0, 1),
    index=0,
    placeholder='No = 0,Yes = 1',
  )

  kidneye = st.selectbox(
  '''Have you been diagnosed with Kidney disease (except for tumor or cancer) by a
  doctor?''',
  (0, 1),
  index=0,
  placeholder='No = 0,Yes = 1',
  )

  digeste = st.selectbox(
  '''Have you been diagnosed with Stomach or other digestive diseases (except for tumor or cancer) by a doctor?''',
    (0, 1),
    index=0,
    placeholder='No = 0,Yes = 1',
  )

  asthmae = st.selectbox(
  '''Have you been diagnosed with Asthma by a doctor?''',
    (0, 1),
    index=0,
    placeholder='No = 0,Yes = 1',
  )
  

  memrye =  st.selectbox(
  '''Have you been diagnosed with Memory-related disease (such as dementia, brain
  atrophy, and Parkinson’s disease) by a doctor?''',
  (0, 1),
  index=0,
  placeholder='No = 0,Yes = 1',
  )

  mdact_c = st.selectbox(
  '''Do you usually do moderate exercise?
  (Note: Moderate exercise includes carrying light items, 
  riding a bike at a regular pace, or other activities making 
  your breathing slightly faster than usual.)''',
  (0, 1),
  index=0,
  placeholder='No = 0,Yes = 1',
  )
 
  hospital =st.selectbox(
  '''Have you received inpatient care in the past year''',
  (0, 1),
  index=0,
  placeholder='No = 0,Yes = 1',
  )

  retire =st.selectbox(
  '''Are you retired?''',
  (0, 1),
  index=0,
  placeholder='No = 0,Yes = 1',
  )

  wrist_pain = st.selectbox(
  '''Do you often have wrist pain?''',
  (0, 1),
  index=0,
  placeholder='No = 0,Yes = 1',
  )

  chest_pain = st.selectbox(
  '''Do you often have chest pain?''',
  (0, 1),
  index=0,
  placeholder='No = 0,Yes = 1',
  )
  
  srh = st.selectbox(
    "Your Self-Reported Health Status Score(1–5 correspond to Very Good, Good, Fair, Poor, and Very Poor (in order).)",
    (1, 2, 3, 4 ,5),
    index=0,
    placeholder="Select you score.",
  )
  srh_encoder = np.zeros(4, dtype=np.int32).reshape(1,-1)
  if srh > 1:
    srh_encoder[0,srh-2] = 1
  st.write(srh_encoder)
  adlab_c = st.selectbox(
    '''How many of the following daily living activities do you have difficulty with?
  (Note: Daily living activities include: using the toilet, feeding yourself,
   dressing yourself, controlling bowel and bladder movements, getting in and out of bed, bathing yourself)''',
    (0, 1, 2 , 3 , 4 , 5 ,6),
    index=0,
    placeholder='''Options: 0 item (no difficulty) / 1 item 
    / 2 items / 3 items / 4 items / 5 items / 6 items.''',
  )
  adlab_c_encoder = np.zeros(6, dtype=np.int32).reshape(1,-1)
  if adlab_c > 0 :
    adlab_c_encoder[0,adlab_c-1] = 1
values = [hibpe, lunge, dyslipe, kidneye, digeste,
asthmae, memrye, mdact_c, hospital, retire, wrist_pain,chest_pain ]
input_values1 = np.array([values])
input_values2 = np.concatenate([input_values1,srh_encoder], axis=1)
input_values = np.concatenate([input_values2,adlab_c_encoder], axis=1)

if st.button("Predict",width="stretch"):
  predicted_class = model.predict(input_values)[0]
  predicted_proba = model.predict_proba(input_values)[0]

  df_proba = pd.DataFrame(predicted_proba).T
  df_proba.columns =['Disease probability','No Disease probability']
  df_proba.rename(columns={0:'Disease',
                          1:'No Disease'})
  st.subheader('Predicted Result')
  st.dataframe(df_proba['Disease probability'],
            column_config={
            'Disease probability':st.column_config.ProgressColumn(
              'Disease probability',
              format='%f',
              width = 'medium',
              min_value =0,
              max_value =1),
            })
  if predicted_class == 1:
    st.write(f'''Based on the model assessment, you have a high risk of developing cardiovascular disease, 
    with a predicted probability of{predicted_proba[0]:.1f} .To better protect your health,
    it is recommended that you consult a doctor in the cardiology or endocrinology department 
    as soon as possible for further professional examinations and interventions.''' )
  if predicted_class == 0:
    st.write(f'''Based on the model assessment, you have a low risk of developing cardiovascular disease, 
    with a predicted probability of{predicted_proba[0]:.1f}.''' )
    

