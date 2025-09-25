import numpy as np
import streamlit as st 
import pandas as pd
import joblib 
import shap 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# Main title for the app
st.title('Prediction of Cardiovascular Disease in Middle-Aged and Elderly Patients with Diabetes Mellitus')

# introduce the app and the method to use it
st.info('''This app, built on CHARLS data with an integrated XGBoost prediction model, assesses the 
cardiovascular disease risk probability of middle-aged and elderly diabetic patients.
To use it, users enter relevant clinical/health info via the left input panel; 
clicking "Predict" provides personalized results, including disease risk probability and related visualizations.''')

# get the column name for input data
X_train = pd.read_csv("X_train.csv")
feature_names = X_train.columns.tolist()
stand_scaler = StandardScaler()
X_train['Age'] = stand_scaler.fit_transform(X_train['Age'].to_frame())
max_scaler = MinMaxScaler()
columns_to_normalize = ['Self Reported Health Status','ADL Score']
X_train[columns_to_normalize] = max_scaler.fit_transform(X_train[columns_to_normalize])

# Age 	Self Reported Health Status 	ADL Score 	Hypertension 	Dyslipidemia 	Kidney disease 	Hospital 	Chest pain


#load trained model
model = joblib.load('rf.pkl')

# Age 	Self Reported Health Status 	ADL Score 	Hypertension 	Dyslipidemia 	Kidney disease 	Hospital 	Chest pain

# set siderbar and select box
with st.sidebar:
  st.header("patient-related information")

  
  #select box
  age = st.slider("How old are you?", 45, 100, 1)

  ##Self-Reported Health Status Score
  srh = st.selectbox(
    "Your Self-Reported Health Status Score(1–5 correspond to Very Good, Good, Fair, Poor, and Very Poor (in order).)",
    (1, 2, 3, 4 ,5),
    index=0,
    placeholder="Select you score.",
  )

  ## daily living activities
  adlab_c = st.selectbox(
    '''How many of the following daily living activities do you have difficulty with?
  (Note: Daily living activities include: using the toilet, feeding yourself,
   dressing yourself, controlling bowel and bladder movements, getting in and out of bed, bathing yourself)''',
    (0, 1, 2 , 3 , 4 , 5 ,6),
    index=0,
    placeholder='''Options: 0 item (no difficulty) / 1 item 
    / 2 items / 3 items / 4 items / 5 items / 6 items.''',
  )
  
  # hypertension
  hibpe = st.selectbox(
    "Has any doctor ever told you that you have hypertension?",
    (0, 1),
    index=0,
    placeholder='No = 0,Yes = 1',
  )
  
  ##  Dyslipidemia
  dyslipe = st.selectbox(
  '''Have you been diagnosed with Dyslipidemia (elevation of low density lipoprotein,
  triglycerides (TGs),and total cholesterol, or a low high density lipoprotein level) by
  a doctor?''',
    (0, 1),
    index=0,
    placeholder='No = 0,Yes = 1',
  )
  
  # Kidney disease
  kidneye = st.selectbox(
  '''Have you been diagnosed with Kidney disease (except for tumor or cancer) by a
  doctor?''',
  (0, 1),
  index=0,
  placeholder='No = 0,Yes = 1',
  )
  
  ##inpatient care in the past year
  hospital =st.selectbox(
  '''Have you received inpatient care in the past year''',
  (0, 1),
  index=0,
  placeholder='No = 0,Yes = 1',
  )

  ##chest_pain
  chest_pain = st.selectbox(
  '''Do you often have chest pain?''',
  (0, 1),
  index=0,
  placeholder='No = 0,Yes = 1',
  )

# merge all the input data
values = [age,srh,adlab_c,hibpe,dyslipe, kidneye, hospital,chest_pain ]
input_values_raw = np.array([values])
input_values = pd.DataFrame(input_values_raw,columns = feature_names)
input_values['Age'] = stand_scaler.transform(input_values['Age'].to_frame())
input_values[columns_to_normalize] = max_scaler.transform(input_values[columns_to_normalize])

# set button for predict
if st.button("Predict",width="stretch"):
  predicted_class = model.predict(input_values)[0]   #get class
  predicted_proba = model.predict_proba(input_values)[0] #get probability

  df_proba = pd.DataFrame(predicted_proba).T   #transpose

  # turn the ndarray to dataframe
  df_proba.columns =['Disease probability','No Disease probability']
  df_proba.rename(columns={0:'Disease',
                          1:'No Disease'})

  # visualize the probability
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

  # give some advice for user
  if predicted_class == 0:
    st.write(f'''Based on the model assessment, you have a high risk of developing cardiovascular disease, 
    with a predicted probability of {100 * predicted_proba[0]:.1f}%.To better protect your health,
    it is recommended that you consult a doctor in the cardiology or endocrinology department 
    as soon as possible for further professional examinations and interventions.''' )
  if predicted_class == 1:
    st.write(f'''Based on the model assessment, you have a low risk of developing cardiovascular disease, 
    with a predicted probability of {100* predicted_proba[0]:.1f}.%''' )


  # SHAP explain
  st.subheader("SHAP Force Plot Explanation")
  explainer_shap = shap.TreeExplainer(model)
  shap_values =explainer_shap.shap_values(pd.DataFrame(input_values,columns = feature_names))

  shap.force_plot(explainer_shap.expected_value,shap_values,input_values,matplotlib=True)

  plt.savefig('shap_force_plot.png', bbox_inches='tight',dpi =1600)
  st.image('shap_force_plot.png',caption = 'SHAP Force Plot Explanation')
    

