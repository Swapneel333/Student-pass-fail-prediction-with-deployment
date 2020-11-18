#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


# In[2]:


model.predict([[117,2,33.0,2,3.0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0]])[0]


# In[3]:


@app.route("/")

def home():
    return render_template("index.html")


# In[4]:


@app.route("/predict", methods = ["GET", "POST"])

def predict():
    if request.method == "POST":
        
        program_duration = int(request.form['program_duration'])
        city_tier = int(request.form['city_tier'])
        age = float(request.form['age'])
        total_programs_enrolled = int(request.form['total_programs_enrolled'])
        trainee_engagement_rating = float(request.form['trainee_engagement_rating'])
        
        program_type = request.form['program_type']
        if(program_type =='T'):
            program_type_T = 1
            program_type_U = 0
            program_type_V = 0
            program_type_X = 0
            program_type_Y = 0
            program_type_Z = 0
            
        elif(program_type =='U'):
            program_type_T = 0
            program_type_U = 1
            program_type_V = 0
            program_type_X = 0
            program_type_Y = 0
            program_type_Z = 0

        elif(program_type =='V'):
            program_type_T = 0
            program_type_U = 0
            program_type_V = 1
            program_type_X = 0
            program_type_Y = 0
            program_type_Z = 0
            
        elif(program_type =='X'):
            program_type_T = 0
            program_type_U = 0
            program_type_V = 0
            program_type_X = 1
            program_type_Y = 0
            program_type_Z = 0
            
        elif(program_type =='Y'):
            program_type_T = 0
            program_type_U = 0
            program_type_V = 0
            program_type_X = 0
            program_type_Y = 1
            program_type_Z = 0
            
        elif(program_type =='Z'):
            program_type_T = 0
            program_type_U = 0
            program_type_V = 0
            program_type_X = 0
            program_type_Y = 0
            program_type_Z = 1
            
        else:
            program_type_T = 0
            program_type_U = 0
            program_type_V = 0
            program_type_X = 0
            program_type_Y = 0
            program_type_Z = 0
            
        test_type = request.form['test_type']
        if(test_type == 'online'):
            test_type_online = 1
        else:
            test_type_online = 0
            
        difficulty_level = request.form['difficulty_level']
        if(difficulty_level == 'hard'):
            difficulty_level_hard = 1
            difficulty_level_intermediate = 0
            difficulty_level_vary_hard = 0
            
        elif(difficulty_level == 'intermediate'):
            difficulty_level_hard = 0
            difficulty_level_intermediate = 1
            difficulty_level_vary_hard = 0
            
        elif(difficulty_level == 'vary hard'):
            difficulty_level_hard = 0
            difficulty_level_intermediate = 0
            difficulty_level_vary_hard = 1
            
        else:
            difficulty_level_hard = 0
            difficulty_level_intermediate = 0
            difficulty_level_vary_hard = 0
            
        gender = request.form['gender']
        if(gender == 'M'):
            gender_M = 1
        else:
            gender_M = 0
            
        education = request.form['education']
        if(education == 'Matriculation'):
            education_Matriculation = 1
            education_High_School_Diploma = 0
            education_Masters = 0
            education_No_Qualification = 0
            
        elif(education == 'High School Diploma'):
            education_Matriculation = 0
            education_High_School_Diploma = 1
            education_Masters = 0
            education_No_Qualification = 0
            
        elif(education == 'Masters'):
            education_Matriculation = 0
            education_High_School_Diploma = 0
            education_Masters = 1
            education_No_Qualification = 0
            
        elif(education == 'No Qualification'):
            education_Matriculation = 0
            education_High_School_Diploma = 0
            education_Masters = 0
            education_No_Qualification = 1
            
        else:
            education_Matriculation = 0
            education_High_School_Diploma = 0
            education_Masters = 0
            education_No_Qualification = 0
            
        is_handicapped = request.form['is_handicapped']
        if(is_handicapped == 'Y'):
            is_handicapped_Y = 1
        else:
            is_handicapped_Y = 0
            
        prediction = model.predict([[program_duration,city_tier,age,total_programs_enrolled,trainee_engagement_rating,
                                     program_type_T,program_type_U,program_type_X,program_type_Y,program_type_Z,
                                     difficulty_level_hard,difficulty_level_intermediate,difficulty_level_vary_hard,
                                     test_type_online,gender_M,education_Matriculation,education_High_School_Diploma,
                                     education_Masters,education_No_Qualification,is_handicapped_Y]])
        
        
        
        if prediction[0]==1:
            return render_template('index.html',prediction_text="Hurray!!, There is a high chance of pass")
        else:
            return render_template('index.html',prediction_text="OOps!!, There is a high chance of fail")
    
    else:
        return render_template('index.html')


# In[5]:


if __name__ == '__main__':
    app.run(debug=True)

