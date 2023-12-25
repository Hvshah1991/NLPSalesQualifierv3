#Core Packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import river
matplotlib.use('Agg')
import seaborn as sns
import altair as alt
import ast
import openai
from datetime import datetime
from utils.get_text import get_text
from PIL import Image
#from transformers import pipeline

# Set the model engine and your OpenAI API key
model_engine = "text-davinci-003"
import os
openai.api_key = os.environ['OPENAI_API_KEY']

def ChatGPT(user_query):
    '''
    This function uses the OpenAI API to generate a response to the given
    user_query using the ChatGPT model
    '''
    # Use the OpenAI API to generate a response
    completion = openai.Completion.create(
                                  engine = model_engine,
                                  prompt = user_query,
                                  max_tokens = 1024,
                                  n = 1,
                                  temperature = 0.5,
                                      )
    response = completion.choices[0].text
    return response

#Welcome Banner
display = image = Image.open("img/drishlabs.png")
display = np.array(display)
st.image(display, width=250)
st.title(":teal[NLP Sales Qualifier v3]")

#Create Subheader
st.subheader('''App created by Raj Shah''')
st.caption('''Predicts with English, Spanish, Polish and German languages''')
st.caption('''Utilizes ChatGPT API''')

#Online ML Pkgs
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords,TFIDF
from river.compose import Pipeline

#Training Data

data = [("business valuations","Business Valuation Expert"),("business valuation","Business Valuation Expert"),("company valuation","Business Valuation Expert"),("valuations","Business Valuation Expert"),("valuation","Business Valuation Expert"),("Business Valuators","Business Valuation Expert"),("valoraciones de empresas","Business Valuation Expert"),("valoración de la empresa","Business Valuation Expert"),("Unternehmensbewertungen","Business Valuation Expert"),("wycena przedsiębiorstw","Business Valuation Expert"),("wyceny firm","Business Valuation Expert"),("wyceny","Business Valuation Expert"),("business acquisitions","M&A"),("business mergers","M&A"),("buy-side","M&A"),("mergers and acquisitions","M&A"),("acquiring","M&A"),("M&A services","M&A"),("M&A due diligence","M&A"),("transaction execution","M&A"),("acquisition","M&A"),("transition","M&A"),("buyer","M&A"),("buying business","M&A"),("buyouts","M&A"),("post-deal","M&A"),("Corporate Finance Advisory","M&A"),("Debt Advisory","M&A"),("Divestures","M&A"),("M&A","M&A"),("M&A advice","M&A"),("M&A integration","M&A"),("M&A transactions","M&A"),("merger acquisition","M&A"),("mergers acquisitions","M&A"),("mergers & acquisitions","M&A"),("Restructurings","M&A"),("sell-side","M&A"),("M&A advisory","M&A"),("selling business","M&A"),("transaction","M&A"),("transactions","M&A"),("transaction advisory","M&A"),("Value Enhancement","M&A"),("adquisiciones de empresas","M&A"),("fusiones de negocios","M&A"),("transacciones m&a","M&A"),("fusiones adquisiciones","M&A"),("transacción","M&A"),("asesoramiento en transacciones","M&A"),("Unternehmensakquisitionen","M&A"),("Unternehmensfusionen","M&A"),("M&A-Transaktionen","M&A"),("Fusionen","M&A"),("Übernahmen","M&A"),("Transaktion","M&A"),("Transaktionsberatung","M&A"),("przejęcia spółek","M&A"),("fuzje spółek","M&A"),("strona kupującego","M&A"),("kupiec","M&A"),("zakup biznesu","M&A"),("wykupy","M&A"),("Doradztwo finansów przedsiębiorstwa","M&A"),("Doradztwo dłużne","M&A"),("Zbycia","M&A"),("transakcje M&A","M&A"),("połączenie przejęcie","M&A"),("fuzje przejęcia","M&A"),("fuzje i przejęcia","M&A"),("Restrukturyzacje","M&A"),("strona sprzedającego","M&A"),("sprzedaż firmy","M&A"),("transakcja","M&A"),("doradztwo transakcyjne","M&A"),("Zwiększanie wyceny","M&A"),("Broker","Business Broker"),("Business Brokerage","Business Broker"),("Buy a Company","Business Broker"),("Sell a company","Business Broker"),("transactions","Business Broker"),("Corredor","Business Broker"),("Corredora","Business Broker"),("transacciones","Business Broker"),("Makler","Business Broker"),("Maklerin","Business Broker"),("Transaktionen","Business Broker"),("pośrednik","Business Broker"),("pośredniczka","Business Broker"),("pośrednictwo biznesowe","Business Broker"),("kup firmę","Business Broker"),("business brokerage","Business Broker"),("buyers sellers","Business Broker"),("Sprzedaj firmę","Business Broker"),("transakcje","Business Broker"),("Capital Raising","Investment Banking"),("Equity Capital Markets","Investment Banking"),("Fundraising","Investment Banking"),("Investment bank","Investment Banking"),("Investment Banking","Investment Banking"),("IPO","Investment Banking"),("Raise capital","Investment Banking"),("Banco de inversiones","Investment Banking"),("Aumentar el capital","Investment Banking"),("Investmentbank","Investment Banking"),("Kapital beschaffen","Investment Banking"),("Pozyskiwanie kapitału","Investment Banking"),("Rynki kapitałowe","Investment Banking"),("Pozyskiwanie funduszy","Investment Banking"),("Bank Inwestycyjny","Investment Banking"),("Bankowość Inwestycyjna","Investment Banking"),("Wyjście na giełdę","Investment Banking"),("Pozyskanie kapitału","Investment Banking"),("appraisal","Business Appraiser"),("Appraiser","Business Appraiser"),("Intangible Valuation","Business Appraiser"),("tasación de negocios","Business Appraiser"),("Bewertung","Business Appraiser"),("wycena biznesowa","Business Appraiser"),("rzeczoznawca","Business Appraiser"),("Wycena wartości niematerialnych i prawnych","Business Appraiser"),("AIFMD","Private Equity"),("Buyouts","Private Equity"),("Investment Management","Private Equity"),("Portfolio Management","Private Equity"),("private equity","Private Equity"),("Investment Management","Private Equity"),("Investment Management","Private Equity"),("Investment Management","Private Equity"),("Capital privado","Private Equity"),("private investment","Private Equity"),("Privates Eigenkapital","Private Equity"),("wykupy","Private Equity"),("Zarządzanie inwestycjami","Private Equity"),("Zarządzanie portfolio","Private Equity"),("accounting","Tax Advisors"),("auditing","Tax Advisors"),("financial statement preparation","Tax Advisors"),("Accounting and tax","Tax Advisors"),("Audit","Tax Advisors"),("Auditor","Tax Advisors"),("Bookkeeping","Tax Advisors"),("certified public accountants","Tax Advisors"),("organization tax","Tax Advisors"),("Payroll","Tax Advisors"),("Tax","Tax Advisors"),("CPA","Tax Advisors"),("tax planning","Tax Advisors"),("assurance","Tax Advisors"),("compliance","Tax Advisors"),("tax return preparation","Tax Advisors"),("contador","Tax Advisors"),("contadora","Tax Advisors"),("impuesto","Tax Advisors"),("Buchhaltung","Tax Advisors"),("buchalter","Tax Advisors"),("buchalterin","Tax Advisors"),("Steuer","Tax Advisors"),("Rachunkowość","Tax Advisors"),("Rachunkowość i podatki","Tax Advisors"),("Doradztwo","Tax Advisors"),("audyt","Tax Advisors"),("biegły rewident","Tax Advisors"),("księgowość","Tax Advisors"),("biegły księgowy","Tax Advisors"),("biegła księgowa","Tax Advisors"),("Podatek organizacyjny","Tax Advisors"),("Lista płac","Tax Advisors"),("podatki","Tax Advisors"),("planowanie podatków","Tax Advisors"),("przygotowanie deklaracji podatkowych","Tax Advisors"),("family office","Family Office Investor"),("high net worth individuals","Family Office Investor"),("HNWI","Family Office Investor"),("wealth management","Family Office Investor"),("wealth planning","Family Office Investor"),("zarządzanie majątkiem","Family Office Investor"),("planowanie majątku","Family Office Investor"),("Oficina familiar","Family Office Investor"),("Familienbüro","Family Office Investor"),("Asset Management","Public Equity"),("fund","Public Equity"),("Fund Management","Public Equity"),("Mutual Fund","Public Equity"),("public equity","Public Equity"),("UCITS","Public Equity"),("world equity","Public Equity"),("patrimonio publico","Public Equity"),("öffentliches","Public Equity"),("zarządzanie aktywami","Public Equity"),("fundusz","Public Equity"),("zarządzanie funduszem","Public Equity"),("fundusz powierniczy","Public Equity"),("Kapitał publiczny","Public Equity"),("kapitał światowy","Public Equity"),("AIFMD","Venture Capital"),("founders","Venture Capital"),("growth capital","Venture Capital"),("growth equity","Venture Capital"),("investments","Venture Capital"),("late stage VC","Venture Capital"),("late stage venture","Venture Capital"),("portfolio","Venture Capital"),("venture capital","Venture Capital"),("scale","Venture Capital"),("venture","Venture Capital"),("entrepreneurs","Venture Capital"),("mentorship","Venture Capital"),("early stage","Venture Capital"),("guidance","Venture Capital"),("start-ups","Venture Capital"),("capital de riesgo","Venture Capital"),("Risikokapital","Venture Capital"),("założyciele","Venture Capital"),("Kapitał wzrostowy","Venture Capital"),("Kapitał własny wzrostowy","Venture Capital"),("Inwestycje","Venture Capital"),("VC wczesnej ekspansji","Venture Capital"),("firmy wczesnej ekspansji","Venture Capital"),("commercial real estate","Real Estate Valuation Expert"),("property","Real Estate Valuation Expert"),("architecture","Real Estate Valuation Expert"),("construction","Real Estate Valuation Expert"),("interior design","Real Estate Valuation Expert"),("real estate","Real Estate Valuation Expert"),("residential condominiums","Real Estate Valuation Expert"),("real estate development","Real Estate Valuation Expert"),("real estate appraisal","Real Estate Valuation Expert"),("bienes raíces comerciales","Real Estate Valuation Expert"),("propiedad comercial","Real Estate Valuation Expert"),("bienes raíces","Real Estate Valuation Expert"),("Gewerbeimmobilien","Real Estate Valuation Expert"),("Eigentum","Real Estate Valuation Expert"),("Immobilie","Real Estate Valuation Expert"),("nieruchomości gospodarcze","Real Estate Valuation Expert"),("Wycena nieruchomości","Real Estate Valuation Expert"),("management plan","Management Consultant"),("Forecasting","Management Consultant"),("Doradztwo","Management Consultant"),("biznesplan","Management Consultant"),("konsulting","Management Consultant"),("Prognozy","Management Consultant"),("Zarządzanie","Management Consultant"),("strategia","Management Consultant"),("Corporate","Corporates"),("Investor Relations","Corporates"),("Products","Corporates"),("Korporacja","Corporates"),("Relacje inwestorskie","Corporates"),("Produkty","Corporates"),("Valuation Expert","In-house Valuation Expert"),("Value Expert","In-house Valuation Expert"),("usługi finansowe","In-house Valuation Expert"),("Międzynarodowa sieć","In-house Valuation Expert"),("usługi","In-house Valuation Expert")]

#Model Building
model = Pipeline(('vectorizer',BagOfWords(lowercase=True)),('nv',MultinomialNB()))
for x,y in data:
    model = model.learn_one(x,y)
    
#Storage in a database
import sqlite3
conn = sqlite3.connect('data1.db')
c = conn.cursor()

#Create Fxn from SQL
def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS predictionTable(URL TEXT, message TEXT, rep_name TEXT, prediction TEXT,new_prediction TEXT, probability NUMBER, businessvaluationexpert_proba NUMBER, mna_proba NUMBER, businessbroker_proba NUMBER, investmentbanking_proba NUMBER, businessappraiser_proba NUMBER, privateequity_proba NUMBER, taxadvisors_proba NUMBER, familyofficeinvestor_proba NUMBER, publicequity_proba NUMBER, venturecapital_proba NUMBER, realestate_proba NUMBER, corporates_proba NUMBER, mgmtconsultant_proba NUMBER, inhousevaluexpert_proba NUMBER, postdate DATE)')
    
def add_data(URL,message,rep_name,prediction,new_prediction, probability,businessvaluationexpert_proba,mna_proba,businessbroker_proba,investmentbanking_proba,businessappraiser_proba,privateequity_proba,taxadvisors_proba,familyofficeinvestor_proba,publicequity_proba,venturecapital_proba,realestate_proba,corporates_proba,mgmtconsultant_proba,inhousevaluexpert_proba,postdate):
    c.execute('INSERT INTO predictionTable(URL,message,rep_name,prediction,new_prediction,probability,businessvaluationexpert_proba,mna_proba,businessbroker_proba,investmentbanking_proba,businessappraiser_proba,privateequity_proba,taxadvisors_proba,familyofficeinvestor_proba,publicequity_proba,venturecapital_proba,realestate_proba,corporates_proba,mgmtconsultant_proba,inhousevaluexpert_proba,postdate) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(URL,message,rep_name,prediction,new_prediction,probability,businessvaluationexpert_proba,mna_proba,businessbroker_proba,investmentbanking_proba,businessappraiser_proba,privateequity_proba,taxadvisors_proba,familyofficeinvestor_proba,publicequity_proba,venturecapital_proba,realestate_proba,corporates_proba,mgmtconsultant_proba,inhousevaluexpert_proba,postdate))
    conn.commit()
    
def view_all_data():
    c.execute("SELECT * FROM predictionTable")
    data = c.fetchall()
    return data


def main():
    menu = ["Home","Manage","People Search","About"]
    create_table()
    
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home")
        with st.form(key='gptform'):
            st.markdown("ChatGPT Summarizer")
            #Get user input
            user_query = st.text_input("Enter URL here, and ChatGPT will summarize it :q", "Provide a 100 word Summary of this company")
            gpt_summarize = st.form_submit_button(label='Summarize')
            if gpt_summarize:
                #Pass the query to the ChatGPT function
                response = ChatGPT(user_query)
                st.write(f"{user_query} {response}")
                
        with st.form(key='webform'):
            st.markdown("Web Scraper")
            text = ""
            submit_scrape = ""
            scrape = ""
            message = ""
            df = ""
            URL = st.text_input("Enter the URL of the webpage you want to scrape")
            do_scrape = st.form_submit_button(label='Scrape')
            #submit_scrape = st.form_submit_button(label='Transfer')
            if URL is not None:
                if do_scrape:
                    text = get_text(URL)
                    st.success("Scrape Successfully Done")
                    df = pd.DataFrame(text.splitlines(),columns=["Webpage_text"],index=None)
                    st.write("You can check the data you scraped from the above URL:")
                    scrape = st.text_area(label="",value=text,height=200)
            else:
                st.warning("Please enter a valid URL")


            
        with st.form(key='mlform'):
            col1,col2 = st.columns([2,1])
            with col1:
                st.write("Content Analyzer")
                message = st.text_area("Content")
                sales_rep = st.text_input("Sales Rep")
                submit_message = st.form_submit_button(label='Predict')
                st.write("Pre-Qualification Override")
                override_input = st.text_input("New Persona")
                do_override = st.form_submit_button(label='Override')
                    
            with col2:
                st.write("Web-Based Machine Learning Qualifier")
                st.write("Predict Text as per defined Finance Industry Personas to Qualify Leads")
                

            
    
        if submit_message or do_override:
            prediction = model.predict_one(message)
            prediction_proba = model.predict_proba_one(message)
            probability = max(prediction_proba.values())
            postdate = datetime.now()
            new_prediction = " "
            new_prediction = override_input
            rep_name = " "
            rep_name = sales_rep
            #Add data to database
            add_data(URL,message,rep_name,prediction,new_prediction,probability,prediction_proba['Business Valuation Expert'],prediction_proba['M&A'],prediction_proba['Business Broker'],prediction_proba['Investment Banking'],prediction_proba['Business Appraiser'],prediction_proba['Private Equity'],prediction_proba['Tax Advisors'],prediction_proba['Family Office Investor'],prediction_proba['Public Equity'],prediction_proba['Venture Capital'],prediction_proba['Real Estate Valuation Expert'],prediction_proba['Corporates'],prediction_proba['Management Consultant'],prediction_proba['In-house Valuation Expert'],postdate)
            st.success("Data Submitted")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info("Original Text")
                st.text_area(label="",value=message,height=200)
                
                st.success("Prediction")
                st.write(prediction)
                
                st.info("Persona")
                if prediction == "In-house Valuation Expert":
                    st.markdown(":green[Green]")
                elif prediction == "M&A":
                    st.markdown(":green[Green]")
                elif prediction == "Tax Advisors":
                    st.markdown(":green[Green]")
                elif prediction == "Business Broker":
                    st.markdown(":orange[Orange]")
                elif prediction == "Business Valuation Expert":
                    st.markdown(":orange[Orange]")
                elif prediction == "Corporates":
                    st.markdown(":orange[Orange]")
                elif prediction == "Business Appraiser":
                    st.markdown(":orange[Orange]")
                elif prediction == "Investment Banking":
                    st.markdown(":orange[Orange]")
                elif prediction == "Private Equity":
                    st.markdown(":orange[Orange]")
                elif prediction == "Family Office Investor":
                    st.markdown(":red[Red]")
                elif prediction == "Management Consultant":
                    st.markdown(":red[Red]")
                elif prediction == "Public Equity":
                    st.markdown(":red[Red]")
                elif prediction == "Venture Capital":
                    st.markdown(":red[Orange]")
                elif prediction == "Real Estate Valuation Expert":
                    st.markdown(":red[Out of Scope]")
                
                st.success("Verdict")
                if prediction == "In-house Valuation Expert":
                    st.markdown(":green[Qualified]")
                elif prediction == "M&A":
                    st.markdown(":green[Qualified]")
                elif prediction == "Tax Advisors":
                    st.markdown(":green[Qualified]")
                elif prediction == "Business Broker":
                    st.markdown(":orange[Qualified]")
                elif prediction == "Business Valuation Expert":
                    st.markdown(":orange[Qualified]")
                elif prediction == "Corporates":
                    st.markdown(":orange[Qualified]")
                elif prediction == "Business Appraiser":
                    st.markdown(":orange[Qualified]")
                elif prediction == "Investment Banking":
                    st.markdown(":orange[Qualified]")
                elif prediction == "Private Equity":
                    st.markdown(":orange[Qualified]")
                elif prediction == "Family Office Investor":
                    st.markdown(":red[Difficult Lead]")
                elif prediction == "Public Equity":
                    st.markdown(":red[Difficult Lead]")
                elif prediction == "Management Consultant":
                    st.markdown(":red[Difficult Lead]")
                elif prediction == "Venture Capital":
                    st.markdown(":orange[Difficult Lead]")
                elif prediction == "Real Estate Valuation Expert":
                    st.markdown(":red[Disqualified]")
                    
                #st.write("Pre-Qualification Override")
                #override_input = st.text_input("")
                #do_override = st.button(label='Override')
                #if do_override:
                    #new_prediction = override_input
                    #st.write(new_prediction)
                    #st.success("Override Successful")
                    
            with res_col2:
                st.info("Probability")
                st.write(prediction_proba)
                
                #Plot of Probability
                df_proba = pd.DataFrame({'label':prediction_proba.keys(),'probability':prediction_proba.values()})
                st.dataframe(df_proba)
                #visualization
                fig = alt.Chart(df_proba).mark_bar().encode(x='label',y='probability')
                st.altair_chart(fig,use_container_width=True)
                    
        #with st.form(key='overrideform'):
            #st.write("Pre-Qualification Override")
            #override_input = st.text_input("")
            #do_override = st.form_submit_button(label='Override')
            #if do_override:
                #new_prediction = override_input
                #st.write(new_prediction)
                #st.success("Override Successful")

                
    
    elif choice == "Manage":
        st.subheader("Manage & Monitor Results")
        stored_data = view_all_data()
        new_df = pd.DataFrame(stored_data,columns=['URL','message','rep_name','prediction','new_prediction','probability','businessvaluationexpert_proba','mna_proba','businessbroker_proba','investmentbanking_proba','businessappraiser_proba','privateequity_proba','taxadvisors_proba','familyofficeinvestor_proba','publicequity_proba','venturecapital_proba','realestate_proba','corporates_proba','mgmtconsultant_proba','inhousevaluexpert_proba','postdate'])
        st.dataframe(new_df)
        new_df['postdate'] = pd.to_datetime(new_df['postdate'])
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        csv = convert_df(new_df)
        
        st.download_button(label="Download data as CSV",data=csv,file_name='pre_qualified_data.csv',mime='text/csv',)
        
        #c = alt.Chart(new_df).mark_line().encode(x='minutes(postdate)',y='probability') #For Minutes
        c = alt.Chart(new_df).mark_line().encode(x='postdate',y='probability')
        st.altair_chart(c)
        
        c_businessvaluationexpert_proba = alt.Chart(new_df['businessvaluationexpert_proba'].reset_index()).mark_line().encode(x='businessvaluationexpert_proba',y='index')
        c_mna_proba = alt.Chart(new_df['mna_proba'].reset_index()).mark_line().encode(x='mna_proba',y='index')
        c_businessbroker_proba = alt.Chart(new_df['businessbroker_proba'].reset_index()).mark_line().encode(x='businessbroker_proba',y='index')
        c_investmentbanking_proba = alt.Chart(new_df['investmentbanking_proba'].reset_index()).mark_line().encode(x='investmentbanking_proba',y='index')
        c_businessappraiser_proba = alt.Chart(new_df['businessappraiser_proba'].reset_index()).mark_line().encode(x='businessappraiser_proba',y='index')
        c_privateequity_proba = alt.Chart(new_df['privateequity_proba'].reset_index()).mark_line().encode(x='privateequity_proba',y='index')
        c_taxadvisors_proba = alt.Chart(new_df['taxadvisors_proba'].reset_index()).mark_line().encode(x='taxadvisors_proba',y='index')
        c_familyofficeinvestor_proba = alt.Chart(new_df['familyofficeinvestor_proba'].reset_index()).mark_line().encode(x='familyofficeinvestor_proba',y='index')
        c_publicequity_proba = alt.Chart(new_df['publicequity_proba'].reset_index()).mark_line().encode(x='publicequity_proba',y='index')
        c_venturecapital_proba = alt.Chart(new_df['venturecapital_proba'].reset_index()).mark_line().encode(x='venturecapital_proba',y='index')
        c_realestate_proba = alt.Chart(new_df['realestate_proba'].reset_index()).mark_line().encode(x='realestate_proba',y='index')
        c_corporates_proba = alt.Chart(new_df['corporates_proba'].reset_index()).mark_line().encode(x='corporates_proba',y='index')
        c_mgmtconsultant_proba = alt.Chart(new_df['mgmtconsultant_proba'].reset_index()).mark_line().encode(x='mgmtconsultant_proba',y='index')
        c_inhousevaluexpert_proba = alt.Chart(new_df['inhousevaluexpert_proba'].reset_index()).mark_line().encode(x='inhousevaluexpert_proba',y='index')
        
        c1,c2 = st.columns(2)
        with c1:
            with st.expander("Business Valuation Expert Probability"):
                st.altair_chart(c_businessvaluationexpert_proba,use_container_width=True)
                
        with c2:
            with st.expander("M&A Probability"):
                st.altair_chart(c_mna_proba,use_container_width=True)
        
        c3,c4 = st.columns(2)
        with c3:
            with st.expander("Business Broker Probability"):
                st.altair_chart(c_businessbroker_proba,use_container_width=True)
                
        with c4:
            with st.expander("Investment Banking Probability"):
                st.altair_chart(c_investmentbanking_proba,use_container_width=True)
        
        c5,c6 = st.columns(2)
        with c5:
            with st.expander("Business Appraiser Probability"):
                st.altair_chart(c_businessappraiser_proba,use_container_width=True)
                
        with c6:
            with st.expander("Private Equity Probability"):
                st.altair_chart(c_privateequity_proba,use_container_width=True)
                
        c7,c8 = st.columns(2)
        with c7:
            with st.expander("Tax Advisors Probability"):
                st.altair_chart(c_taxadvisors_proba,use_container_width=True)
                
        with c8:
            with st.expander("Family Office Investor Probability"):
                st.altair_chart(c_familyofficeinvestor_proba,use_container_width=True)
                
        c9,c10 = st. columns(2)
        with c9:
            with st.expander("Public Equity Probability"):
                st.altair_chart(c_publicequity_proba,use_container_width=True)
        with c10:
            with st.expander("Venture Capital Probability"):
                st.altair_chart(c_venturecapital_proba,use_container_width=True)
                
        c11,c12 = st.columns(2)
        with c11:
            with st.expander("Real Estate Valuation Expert Probability"):
                st.altair_chart(c_realestate_proba,use_container_width=True)
        with c12:
            with st.expander("Corporates Probability"):
                st.altair_chart(c_corporates_proba,use_container_width=True)
                
        c13,c14 = st.columns(2)
        with c13:
            with st.expander("Management Consultant Probability"):
                st.altair_chart(c_mgmtconsultant_proba,use_container_width=True)
        with c14:
            with st.expander("In-house Valuation Expert Probability"):
                st.altair_chart(c_inhousevaluexpert_proba,use_container_width=True)
                
                
                
        #with st.expander("Prediction Distribution"):
            #fig2 = plt.figure()
            #ax = sns.countplot(y='probability',data=new_df)
            #ax.bar_label(ax.containers[0],label_type='edge')
            #st.pyplot(fig2)
    #Web Scraper
    elif choice == "People Search":
        st.subheader("Find People from Web Scraper")
        st.markdown('##### This option helps you to find employees, and extract their contact info')
        st.markdown('Under Development')
        URL = st.text_input("Enter the URL of the webpage you want to scrape")
        if URL is not None:
            if st.button("Scrape"):
                text = get_text(URL)
                st.success("Srape Successfully Done")
                df = pd.DataFrame(text.splitlines(),columns=["Webpage_text"],index=None)
                #st.markdown('## Showing the first ten lines of the text')
                #st.dataframe(df.head(10))
                st.text_area(label="",value=text,height=200)
                #st.info('''Download the text as a csv file if you like.''')
                #st.download_button(label="Download the text as a csv file", data=df.to_csv(index=False, encoding='utf-8'),file_name='webpage_text.csv',mime='text/csv')
                #summarizer = pipeline("summarization",model="philschmid/bart-large-cnn-samsum")
                #sum_text = summarizer(text, max_length=130,min_length=30,do_sample=False)
                #st.text_Area(label="",value=sum_text,height=1000)
        else:
            st.warning("Please enter a valid URL")
        
    else:
        st.subheader("About")
        st.caption("NLP Sales Qualifier can be used for Pre-Qualifying Prospects for Lead Sourcing. When you have identified potential leads, you can access their company website (URL) and summarize their website using this tool or web scrape their website using this program. The program can instantaneously give results when you copy-paste the details into the content analyzer section in Home screen - so it can start an analysis if this client Pre-Qualifies for sales outreach or sequencing. This program uses Naive Bayes to predict and classify the lead. This program also stores the data which was produced in the form of output and utilizes it as training data for new queries. For further information on this program, contact: harshvshah.22@gmail.com")
        
        display = image = Image.open("img/Artwork.png")
        display = np.array(display)
        st.image(display, width=600)
    
if __name__ == '__main__':
    main()
    
