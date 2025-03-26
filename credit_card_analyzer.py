import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import glob
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI with correct parameters
try:
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        request_timeout=30
    )
except Exception as e:
    st.error(f"Error initializing ChatOpenAI: {str(e)}")
    st.stop()

embeddings = OpenAIEmbeddings()

# Set page config
st.set_page_config(
    page_title="Bank and Credit Card Statement Analyzer",
    page_icon="💳",
    layout="wide"
)

# Title and description
st.title("💳 Bank and Credit Card Statement Analyzer")
st.markdown("""
This app helps you analyze your credit card statements and answer questions about your spending patterns.
Upload your credit card statements in CSV format, and ask questions about your spending!
""")

# File uploader
uploaded_files = st.file_uploader(
    "Upload your bank/credit card statements (CSV files)",
    type=['csv'],
    accept_multiple_files=True
)

# Initialize session state for storing processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None

def process_statement1(file, filename):
    #Process a credit card statement and return a DataFrame"""
    try:
        df = pd.read_csv(file,parse_dates=True, infer_datetime_format=True)
        # Convert Date column to datetime
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
        # Add filename as card identifier
        df['Card_Name'] = filename
        return df
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        return None


def process_statement(file, filename):
    """Process the bank statement with flexible handling for different formats."""
    try:
        # Load the CSV
        df = pd.read_csv(file, delimiter=",", quotechar='"', skipinitialspace=True)

        # Check if the file is incorrectly parsed into a single column or properly formatted
        if len(df.columns) == 1:  # Single-column issue
            print("⚠️ Detected single-column format. Splitting columns...")
            
            # Split the single-column data into multiple columns
            df_split = df[df.columns[0]].str.split(',', expand=True)

            # Merge date columns into one
            df_split['Transaction Date'] = df_split[0] + " " + df_split[1]
            df_split['Transaction Date'] = pd.to_datetime(df_split['Transaction Date'], errors='coerce')

            # Clean and convert amount fields
            df_split['Debit'] = df_split[3].str.replace('"', '').str.strip().replace('', '0').astype(float)
            df_split['Credit'] = df_split[4].str.replace('"', '').str.strip().replace('', '0').astype(float)

            # Calculate the final Amount column
            df_split['Amount'] = df_split['Debit'] - df_split['Credit']

            # Rename and structure the final DataFrame
            df_cleaned = df_split.rename(columns={
                2: 'Description',
                5: 'Category'
            })[['Transaction Date', 'Description', 'Amount', 'Category']]

        else:  # Properly formatted CSV
            print("✅ Detected properly formatted CSV.")
            
            # Rename columns
            df.rename(columns={
               # 'Date': 'Transaction Date',
                'Description': 'Description',
               # 'Debit': 'Amount',
                'Category': 'Category'
            }, inplace=True)

            # Convert date to datetime format
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')

            # Calculate the final Amount
           # df['Amount'] = pd.to_numeric(df['Debit'].fillna(0)) - pd.to_numeric(df['Credit'].fillna(0))

            # Select relevant columns
            df_cleaned = df[['Transaction Date', 'Description', 'Amount', 'Category']]

        # Add filename as card identifier
        df_cleaned['Card_Name'] = filename

        return df_cleaned

    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")
        return None

def process_statement2(file, filename):
    """Process bank statements dynamically and detect HDFC format."""
    try:
        # Load the CSV file
        df = pd.read_csv(file, delimiter=",", quotechar='"', skipinitialspace=True)

        # ✅ Detect HDFC format by checking for metadata or empty header
        if len(df.columns) == 1 or "********" in str(df.columns[0]):
            print(f"🔍 Detected HDFC format in {filename}. Processing as HDFC...")

            # Reload the HDFC statement, skipping the metadata rows
            df = pd.read_csv(file, skiprows=19)

            # Use positional indexing for HDFC columns
            df_cleaned = pd.DataFrame({
                'Transaction Date': pd.to_datetime(df.iloc[:, 0], errors='coerce'),
                'Description': df.iloc[:, 1],
                'Debit': pd.to_numeric(df.iloc[:, 4].str.replace(',', ''), errors='coerce').fillna(0),
                'Credit': pd.to_numeric(df.iloc[:, 5].str.replace(',', ''), errors='coerce').fillna(0)
            })

            # Calculate the Amount column (Credit - Debit)
            df_cleaned['Amount'] = df_cleaned['Credit'] - df_cleaned['Debit']

        else:
            print(f"✅ Processing standard CSV format for {filename}...")

            # Handle properly formatted CSVs or single-column issues
            if len(df.columns) == 1:
                print(f"⚠️ Detected single-column format in {filename}. Splitting columns...")

                # Split into multiple columns
                df_split = df[df.columns[0]].str.split(',', expand=True)

                # Merge date columns
                df_split['Transaction Date'] = df_split[0] + " " + df_split[1]
                df_split['Transaction Date'] = pd.to_datetime(df_split['Transaction Date'], errors='coerce')

                # Clean debit/credit columns
                df_split['Debit'] = pd.to_numeric(df_split[3].str.replace('"', '').str.strip().replace('', '0'), errors='coerce').fillna(0)
                df_split['Credit'] = pd.to_numeric(df_split[4].str.replace('"', '').str.strip().replace('', '0'), errors='coerce').fillna(0)

                # Calculate Amount
                df_split['Amount'] = df_split['Debit'] - df_split['Credit']

                # Rename and structure
                df_cleaned = df_split.rename(columns={2: 'Description', 5: 'Category'})[['Transaction Date', 'Description', 'Amount']]

            else:
                # Properly formatted CSV
                df.rename(columns={
                   # 'Date': 'Transaction Date',
                    'Description': 'Description',
                    #'Debit': 'Debit',
                    #'Credit': 'Credit',
                    'Category': 'Category'
                }, inplace=True)

                # Convert dates
                df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')

                # Calculate Amount
               # df['Amount'] = pd.to_numeric(df['Debit'].fillna(0)) - pd.to_numeric(df['Credit'].fillna(0))

                # Select relevant columns
                df_cleaned = df[['Transaction Date', 'Description', 'Amount']]

        # Add filename for identification
        df_cleaned['Card_Name'] = filename

        return df_cleaned

    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")
        return None



def create_text_summary(df):
    """Create a text summary of the statement"""
    total_spent = df['Amount'].sum()
    num_transactions = len(df)
    date_range = f"{df['Transaction Date'].min().strftime('%Y-%m-%d')} to {df['Transaction Date'].max().strftime('%Y-%m-%d')}"
    
    # Calculate category totals if Category column exists
    category_totals = ""
    if 'Category' in df.columns:
        category_totals = "\nCategory Totals:\n" + df.groupby('Category')['Amount'].sum().to_string()
    
    summary = f"""
    Statement Summary:
    - Card: {df['Card_Name'].iloc[0]}
    - Date Range: {date_range}
    - Total Transactions: {num_transactions}
    - Total Amount: ${total_spent:.2f}
    {category_totals}
    
    Top 5 Transactions:
    {df.nlargest(5, 'Amount')[['Transaction Date', 'Description', 'Amount']].to_string()}
    """
    return summary

def setup_vector_store(summaries):
    """Set up FAISS vector store with the summaries"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    texts = text_splitter.split_text("\n\n".join(summaries))
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

def create_visualizations(df):
    """Create visualizations for the data"""
    # Daily spending trend
    #daily_spending = df.groupby('Transaction Date','Card_Name')['Amount'].sum().reset_index()
    daily_spending = df.groupby(['Transaction Date', 'Card_Name'], as_index=False).agg({'Amount': 'sum'})
    fig1 = px.bar(daily_spending, x='Transaction Date', y='Amount', color ='Card_Name',
                   title='Daily Spending Trend')
    
    # Category spending (if Category column exists)
    if 'Category' in df.columns:
        category_spending = df.groupby('Category')['Amount'].sum().reset_index()
        fig2 = px.pie(category_spending, values='Amount', names='Category',
                     title='Spending by Category')
    else:
        fig2 = None
    
    return fig1, fig2

# Process uploaded files
if uploaded_files:
    st.subheader("📊 Processing Statements")
    
    for file in uploaded_files:
        filename = file.name
        if filename not in st.session_state.processed_data:
            with st.spinner(f"Processing {filename}..."):
                df = process_statement(file, filename)
                if df is not None:
                    st.session_state.processed_data[filename] = df
                    st.success(f"Successfully processed {filename}")
    
    # Create summaries and vector store
    if st.session_state.processed_data:
        summaries = []
        all_dfs = []
        
        for filename, df in st.session_state.processed_data.items():
            summary = create_text_summary(df)
            summaries.append(summary)
            all_dfs.append(df)
        
        # Combine all dataframes
        st.session_state.combined_df = pd.concat(all_dfs, ignore_index=True)
        
        st.session_state.vector_store = setup_vector_store(summaries)
        st.success("✅ All statements processed and indexed!")
        
        # Display visualizations
        st.subheader("📈 Spending Visualizations")
        
        # Overall spending trend
        fig1, fig2 = create_visualizations(st.session_state.combined_df)
        st.plotly_chart(fig1, use_container_width=True)
        
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Spending", f"${st.session_state.combined_df['Amount'].sum():.2f}")
        with col2:
            st.metric("Average Transaction", f"${st.session_state.combined_df['Amount'].mean():.2f}")
        with col3:
            st.metric("Number of Transactions", len(st.session_state.combined_df))

# Question answering interface
if st.session_state.vector_store:
    st.subheader("❓ Ask Questions About Your Spending")
    
    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Based on the following credit card statement summaries:
        
        {context}
        
        Please answer this question: {question}
        
        Answer:
        """
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Question input
    question = st.text_input("Enter your question about your spending:")
    
    if question:
        with st.spinner("Thinking..."):
            # Get relevant context
            docs = st.session_state.vector_store.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Get answer
            response = chain.run(context=context, question=question)
            
            # Display answer
            st.markdown("### Answer:")
            st.write(response)

# Display raw data if requested
if st.session_state.processed_data:
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Statement Data")
        for filename, df in st.session_state.processed_data.items():
            st.write(f"### {filename}")
            st.dataframe(df) 