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
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
    page_title="Credit Card Statement Analyzer",
    page_icon="💳",
    layout="wide"
)

# Title and description
st.title("💳 Credit Card Statement Analyzer")
st.markdown("""
This app helps you analyze your credit card statements and answer questions about your spending patterns.
Upload your credit card statements in CSV format, and ask questions about your spending!
""")

# File uploader
uploaded_files = st.file_uploader(
    "Upload your credit card statements (CSV files)",
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

def process_statement(file, filename):
    """Process a credit card statement and return a DataFrame"""
    try:
        df = pd.read_csv(file)
        # Convert Date column to datetime
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
        # Add filename as card identifier
        df['Card_Name'] = filename
        return df
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
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
    daily_spending = df.groupby('Transaction Date')['Amount'].sum().reset_index()
    fig1 = px.line(daily_spending, x='Transaction Date', y='Amount', 
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