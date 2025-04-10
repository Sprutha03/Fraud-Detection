# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # ----------------------
# # Page setup
# st.set_page_config(page_title="Fraud Detection", layout="wide")
# st.markdown("""
#     <style>
#         .main {
#             background-color: #f9fafb;
#         }
#         .stButton>button {
#             background-color: #ff4b4b;
#             color: white;
#             border-radius: 8px;
#             padding: 0.5em 1em;
#         }
#         .stTextInput>div>input {
#             border-radius: 10px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.title("🛡️ AI-powered Credit Card Fraud Detection")
# st.write("Enter transaction details to check if it's **Legitimate** or **Fraudulent**.")

# # ----------------------
# # Load dataset
# try:
#     data = pd.read_csv("creditcard.csv")

#     # Data balancing
#     legit = data[data.Class == 0]
#     fraud = data[data.Class == 1]
#     legit_sample = legit.sample(n=len(fraud), random_state=2)
#     data = pd.concat([legit_sample, fraud], axis=0)

#     # Train model
#     X = data.drop(columns="Class", axis=1)
#     y = data["Class"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

#     model = LogisticRegression()
#     model.fit(X_train, y_train)

#     st.success(f"Model trained ✅ | Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")

#     # ----------------------
#     # Input section
#     st.markdown("### 🧮 Enter all 30 features (V1 to V28, Time, Amount) as comma-separated values:")
#     user_input = st.text_input("Example: 0.1, -1.2, ..., 20.5, 100.00")

#     if st.button("🧪 Run Prediction"):
#         try:
#             values = np.array(user_input.split(','), dtype=np.float64)
#             if len(values) != X.shape[1]:
#                 st.warning(f"Expected {X.shape[1]} values, but got {len(values)}. Please check your input.")
#             else:
#                 prediction = model.predict(values.reshape(1, -1))
#                 if prediction[0] == 0:
#                     st.success("✅ This is a **Legitimate transaction**.")
#                 else:
#                     st.error("⚠️ This is a **Fraudulent transaction**.")
#         except:
#             st.error("❌ Invalid input. Please enter valid numeric values separated by commas.")

# except FileNotFoundError:
#     st.error("❌ 'creditcard.csv' file not found. Please upload it below:")

#     uploaded_file = st.file_uploader("Upload creditcard.csv", type="csv")
#     if uploaded_file is not None:
#         with open("creditcard.csv", "wb") as f:
#             f.write(uploaded_file.read())
#         st.success("File uploaded successfully! Please refresh to re-run the app.")


# app.py
import streamlit as st
import numpy as np
import joblib

# ----------------------
# Page setup
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f9fafb;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stTextInput>div>input {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ AI-powered Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's **Legitimate** or **Fraudulent**.")

# ----------------------
# Load pre-trained model
try:
    model = joblib.load("fraud_model.pkl")
    st.success("Model loaded ✅ | Accuracy: 0.96")  # Replace 0.96 with your actual accuracy
    
    # Input section
    st.markdown("### 🧮 Enter all 30 features (V1 to V28, Time, Amount) as comma-separated values:")
    user_input = st.text_input("Example: 0.1, -1.2, ..., 20.5, 100.00")

    if st.button("🧪 Run Prediction"):
        try:
            values = np.array(user_input.split(','), dtype=np.float64)
            if len(values) != model.n_features_in_:
                st.warning(f"Expected {model.n_features_in_} values, but got {len(values)}. Please check your input.")
            else:
                prediction = model.predict(values.reshape(1, -1))
                if prediction[0] == 0:
                    st.success("✅ This is a **Legitimate transaction**.")
                else:
                    st.error("⚠️ This is a **Fraudulent transaction**.")
        except:
            st.error("❌ Invalid input. Please enter valid numeric values separated by commas.")

except FileNotFoundError:
    st.error("❌ Model file missing. Please contact support.")