Hi all,

I created this credit card statement analyzer as a beginner step to learn AI and its frameworks. I have used Langchain framewor, openAI model and pydantic for text categorization. 

It can take ingest credit card statements in PDFs as the input and return an overview of your spending by month and category. 

Here is the data pipeline architecture: 

	PDF → raw_transactions
         ↓
   stg_transactional        ← standardize
         ↓
int_transactions_cleaned    ← clean, dedupe, categorize
         ↓
┌─────────────────────────────────┐
│ mart_spending_by_category       │
│ mart_monthly_summary            │
│ mart_anomalies  			      │
└─────────────────────────────────┘

It also the following tabs:

    - Deep Dive category
	
    - Deep Dive merchant
	
    - Recommendations
	
    - Transactions by date, category, amount and merchant with csv export file 

Here is the link to the app. https://creditcardanalyzer-zftp.vercel.app/ 

Stack Used: 
- SQL
- dbt
- Python <img width="184" height="148" alt="image" src="https://github.com/user-attachments/assets/e928264b-9115-4487-b483-8bf201f9d7e8" />

- 
Please let me know any feedback. I'm constantly updating my app to make it more efficient. Thank you! 


  
