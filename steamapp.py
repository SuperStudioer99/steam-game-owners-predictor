import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np

#Loading our best model (Gradient Boosting Regressor)
with open('GradientBoosting.pkl', 'rb') as file:
  model = pickle.load(file)

st.set_page_config(page_title="Steam Predictor", layout="wide", page_icon="🎮")
st.title("🎮 Steam Game Predictor", anchor=False)
st.subheader("Enter game data to see its estimated owners and features influences", anchor=False)

MAX_VALS = {
    'recommendations': 287673,
    'years': 29,
    'playtime': 359665,
    'positive': 1739980,
    'negative': 294520,
    'age': 18
}

col1, col2 = st.columns(2)

with col1:
    recommendations = st.number_input("📝 Recommendations", min_value=0, value=0, step=1, max_value=MAX_VALS['recommendations'])
    years = st.number_input("⏳ Years Since Release", min_value=0, value=0, step=1, max_value=MAX_VALS['years'])
    avg_playtime = st.number_input("⏱️ Avg Playtime Forever (minutes)", min_value=0, value=0, step=1, max_value=MAX_VALS['playtime'])

with col2:
    positive = st.number_input("👍 Positive Reviews", min_value=0, value=0, step=1,max_value= MAX_VALS['positive'])
    negative = st.number_input("👎 Negative Reviews", min_value=0, value=0, step=1, max_value= MAX_VALS['negative'])
    req_age = st.number_input("🔞 Required Age", min_value=0, value=0, step=1, max_value=MAX_VALS['age'])

st.write("---")

if st.button("🚀 Calculate Estimated Owners", use_container_width=True):
    rec_val = min(recommendations, MAX_VALS['recommendations'])
    years_val = min(years, MAX_VALS['years'])
    play_val = min(avg_playtime, MAX_VALS['playtime'])
    pos_val = min(positive, MAX_VALS['positive'])
    neg_val = min(negative, MAX_VALS['negative'])
    age_val = min(req_age, MAX_VALS['age'])

    yearly_rec = rec_val / (years_val + 1)
    total_reviews = pos_val + neg_val
    review_score = pos_val / (total_reviews + 1)

    input_df = pd.DataFrame([[
        yearly_rec, play_val, years_val, rec_val, age_val, neg_val, review_score
    ]], columns=model.feature_names_in_)

    pred_sqrt = model.predict(input_df)[0]
    final_val = int(round(max(0, pred_sqrt) ** 2))

    st.balloons()
    st.markdown(f"### The estimated number of owners is:")
    st.metric(label="Predicted Owner Count", value=f"{final_val:,} owners")

    st.write("---")
    st.subheader("🔍 Influence on Owner Count", anchor=False)

    explainer = shap.TreeExplainer(model)
    shap_results = explainer(input_df)
    base_val = shap_results.base_values[0]
    total_sqrt = base_val + np.sum(shap_results.values[0])

    owner_impacts = []
    for i in range(len(model.feature_names_in_)):
        s_val = shap_results.values[0][i]
        val_without = (total_sqrt - s_val) ** 2
        impact = int(round(final_val - val_without))
        owner_impacts.append(impact)

    impact_df = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Impact (Owners)': owner_impacts
    }).sort_values(by='Impact (Owners)', ascending=True)

    impact_df['Color'] = ['#FF4B4B' if x < 0 else '#29b5e8' for x in impact_df['Impact (Owners)']]

    st.bar_chart(data=impact_df, x='Feature', y='Impact (Owners)', color="Color")
    st.dataframe(impact_df[['Feature', 'Impact (Owners)']].sort_values(by='Impact (Owners)', ascending=False))
