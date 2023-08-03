import pandas as pd
import streamlit as st 

from sqlalchemy import create_engine
import joblib, pickle

model1 = pickle.load(open('logic.pkl', 'rb'))
pipe = joblib.load('pipeline.joblib')


def predict_Y(data, user, pw, db):

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    
    
    clean = pd.DataFrame(pipe.transform(data), columns=pipe.named_steps['preprocess'].get_feature_names_out())
    
    
    prediction = pd.DataFrame(model1.predict(clean), columns = ['credit_pred'])
    
    final = pd.concat([prediction, data], axis = 1)
        
    final.to_sql('pla', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final


def main():

    st.title("PLA prdiction")
    st.sidebar.title("Classification ML")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Credit prediction </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Upload a file" , type = ['csv','xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
        
        
    else:
        st.sidebar.warning("Upload a CSV or Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_Y(data, user, pw, db)
                           
        import seaborn as sns
        cm = sns.light_palette("yellow", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm).set_precision(2))

                           
if __name__=='__main__':
    main()

