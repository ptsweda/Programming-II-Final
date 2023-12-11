import streamlit as st 
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Page Configuration
st.set_page_config(
    page_title ="LinkedIn Probability",
    layout = "wide",
    initial_sidebar_state="expanded")

st.image('linkedin1.jpg')
# Title of the App
st.title("Who's A User?")
st.subheader("Predicting :blue[LinkedIn Users] with Machine Learning")
st.write("Sometimes it seems like the whole world uses LinkedIn. Of course, while it is a popular social networking site and useful for marketing purposes, not everyone uses it. Our marketing analytics team has built the following app to help evaluate options for promoting the business on different mediums by predicting whether someone uses LinkedIn.")
st.header("", divider='blue')       

# Load the dataset
s = pd.read_csv("social_media_usage.csv")
# Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

# Create a new dataframe called "ss"
ss = pd.DataFrame({"Income": np.where(s["income"] > 9, np.nan, s["income"]),
                   "Education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
                   "Parent": np.where(s["par"] == 1, 1,0),
                   "Married": np.where(s["marital"] == 1, 1,0),
                   "Female": np.where(s["gender"] == 2 ,1,0),
                   "Age": np.where(s["age"] > 98, np.nan, s["age"]),
                   "sm_li": s["web1h"].apply(clean_sm)})
ss = ss.dropna()
# Create a target vector (y) and feature set (X)
y = ss["sm_li"]
X = ss[["Income", "Education", "Parent", "Married", "Female", "Age"]]

# Split the data into training and test sets. Hold out 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X,
                         y,                            
                         stratify=y,                     # same number of target in training & test set
                         test_size=0.2,                 # hold out 20% of the data for testing
                         random_state=987)              # set for reproducibility

# Instantiate a logistic regression model and set class_weight to balanced
# Initialize algorithm
lr = LogisticRegression(class_weight='balanced')
# Fit the algorithm to training data
lr.fit(X_train, y_train)


#Sidebar Questions/Widgets
st.sidebar.title("Demographics")
st.sidebar.caption("Please answer the following questions to receive a probility prediction of a person's LinkedIn account status")
with st.sidebar:
    gen = st.radio("Gender", ["Male", "Female"], index=0)
    mar = st.radio("Martial Status", ["Not Married", "Married"], index=0)
    par = st.radio("Parental Status", ["No Children", "Parent"], index=0)
    number = st.number_input("Please enter estimated yearly income", value=None, placeholder="Type a number...")
    st.write('Estimated yearly income is $', number)
    educ = st.selectbox("Please select an education level", 
             options = ["Less than high school (Grades 1-8 or no formal schooling)",
                        "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                        "High school graduate (Grade 12 with diploma or GED certificate)",
                        "Some college, no degree (includes some community college)",
                        "Two-year associate degree from a college or university"
                         "Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)",
                         "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                         "Postgraduate or professional degree, including master's, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"])
    age = st.slider("What is this person's age?", 0, 98, 25)
    st.write("They are", age, "years old")

# Education Selectbox 
if educ == "Less than high school (Grades 1-8 or no formal schooling)":
     educ = 1
elif educ == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
     educ = 2
elif educ == "High school graduate (Grade 12 with diploma or GED certificate)":
     educ = 3
elif educ == "Some college, no degree (includes some community college)":
     educ = 4
elif educ == "Two-year associate degree from a college or university":
     educ = 5
elif educ == "Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)":
     educ = 6
elif educ == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
     educ = 7
else:
     educ = 8

# Income Input
if number is None:
    number = 0
if number < 10000:
    inc = 1
elif number >= 10000 and number < 20000:
    inc = 2
elif number >= 20000 and number < 30000:
    inc = 3
elif number >= 30000 and number < 40000:
    inc = 4
elif number >= 40000 and number < 50000:
    inc = 5
elif number >= 50000 and number < 75000:
    inc = 6
elif number >= 75000 and number < 100000:
    inc = 7
elif number >= 100000 and number < 150000:
    inc = 8
else:
    inc = 9

# Create labels from numeric inputs
# Income
if number <= 30000:
   inc_label = "low income"
elif number > 30000 and number < 100000:
   inc_label = "middle income"
else:
   inc_label = "high income"

# Gender   
if gen == "Male":
    gen = 0
    gen_label = "man"
else:
    gen = 1
    gen_label = "woman"
    
# Marital
if mar == "Not Married":
    mar = 0
    mar_label = "non-married"
else:
    mar = 1
    mar_label = "married"

# Parent
if par == "No Children":
    par =0
    par_label = "no children"
else:
    par = 1
    par_label = "children"

st.write(f"**The applicant is a {age} year old {gen_label}, {mar_label}, with {par_label}, and is in a {inc_label} bracket**")

# New data for features: Income, Education, Parent, Martial Status, Gender, Age
person = [inc, educ, par, mar, gen, age]
# Predict class, given input features
predicted_class = lr.predict([person])

probs = lr.predict_proba([person])

probs = np.round(probs, 2)

result = st.button(":red[Calculate The Probability They Are On LinkedIn!]")
st.write(result)
if result: st.write(f"There is a **:blue[{probs[0][1]}] probability** that this {gen_label} is on :blue[LinkedIn]")
