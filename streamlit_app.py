
import streamlit as st
import pandas as pd
import joblib 
import shap
import matplotlib.pyplot as plt
st.title('Prediction of Cardiovascular Disease in Middle-Aged and Elderly Patients with Diabetes Mellitus')

st.info('This app is based on machine learning model')
df = pd.read_csv("X_test.csv")
model = joblib.load('XGB.pkl')


## 'srh', 'adlab_c', 'hibpe', 'lunge', 'dyslipe', 'kidneye', 'digeste',
##       'asthmae', 'memrye', 'mdact_c', 'hospital', 'retire', 'wrist_pain',
##       'chest_pain' 
with st.sidebar:
  st.header("patient-related information")
  srh = st.selectbox(
    "Your Self-Reported Health Status Score(1–5 correspond to Very Good, Good, Fair, Poor, and Very Poor (in order).)",
    (1, 2, 3, 4 ,5),
    index=None,
    placeholder="Select you score.",
  )
  
  adlab_c = st.selectbox(
    '''How many of the following daily living activities do you have difficulty with?
  (Note: Daily living activities include: using the toilet, feeding yourself,
   dressing yourself, controlling bowel and bladder movements, getting in and out of bed, bathing yourself)''',
    (0, 1, 2 , 3 , 4 , 5 ,6),
    index=None,
    placeholder='''Options: 0 item (no difficulty) / 1 item 
    / 2 items / 3 items / 4 items / 5 items / 6 items.''',
  )
  
  
  hibpe = st.selectbox(
    "Has any doctor ever told you that you have hypertension?",
    (0, 1),
    index=None,
    placeholder='No = 0,Yes = 1',
  )

  
  lunge =  st.selectbox(
    "Have you been diagnosed with Chronic lung diseases, such as chronic bronchitis ,emphysema ( excluding tumors, or cancer) by a doctor",
    (0, 1),
    index=None,
    placeholder='No = 0,Yes = 1',
  )

  
  dyslipe = st.selectbox(
  '''Have you been diagnosed with Dyslipidemia (elevation of low density lipoprotein,
  triglycerides (TGs),and total cholesterol, or a low high density lipoprotein level) by
  a doctor?''',
    (0, 1),
    index=None,
    placeholder='No = 0,Yes = 1',
  )

  kidneye = st.selectbox(
  '''Have you been diagnosed with Kidney disease (except for tumor or cancer) by a
  doctor?''',
  (0, 1),
  index=None,
  placeholder='No = 0,Yes = 1',
  )

  digeste = st.selectbox(
  '''Have you been diagnosed with Stomach or other digestive diseases (except for tumor or cancer) by a doctor?''',
    (0, 1),
    index=None,
    placeholder='No = 0,Yes = 1',
  )

  asthmae = st.selectbox(
  '''Have you been diagnosed with Asthma by a doctor?''',
    (0, 1),
    index=None,
    placeholder=''No = 0,Yes = 1',
  )
  

  memrye =  st.selectbox(
  '''Have you been diagnosed with Memory-related disease (such as dementia, brain
  atrophy, and Parkinson’s disease) by a doctor?''',
  (0, 1),
  index=None,
  placeholder='No = 0,Yes = 1',
  )

  mdact_c = st.selectbox(
  '''Do you usually do moderate exercise?
  (Note: Moderate exercise includes carrying light items, 
  riding a bike at a regular pace, or other activities making 
  your breathing slightly faster than usual.)''',
  (0, 1),
  index=None,
  placeholder='No = 0,Yes = 1',
  )
 
  hospital =st.selectbox(
  '''Have you received inpatient care in the past year''',
  (0, 1),
  index=None,
  placeholder='No = 0,Yes = 1',
  )

  retire =st.selectbox(
  '''Are you retired?''',
  (0, 1),
  index=None,
  placeholder='No = 0,Yes = 1',
  )

  wrist_pain = st.selectbox(
  '''Do you often have wrist pain?''',
  (0, 1),
  index=None,
  placeholder='No = 0,Yes = 1',
  )

  chest_pain = st.selectbox(
  '''Do you often have chest pain?''',
  (0, 1),
  index=None,
  placeholder='No = 0,Yes = 1',
  )


st.write(type(adlab_c))
