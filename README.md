# RetailMind-Analysis

RetailMind 🔮: Advanced Retail Analytics Dashboard
RetailMind is a powerful, interactive web application built with Streamlit that provides a comprehensive suite of tools for analyzing retail sales data. This dashboard empowers users to move beyond simple sales tracking and uncover deep insights into customer behavior, predict future trends, and make data-driven decisions to boost profitability and customer retention.

From high-level sales overviews to granular, predictive customer analytics, RetailMind transforms raw transactional data into actionable intelligence.

✨ Features
The dashboard is organized into several analytical tabs, each providing a unique perspective on your data:

📊 Sales Overview: Get a bird's-eye view of your business with key performance indicators (KPIs) like Total Sales, Total Orders, and Unique Customers. Visualize sales trends over time and identify top-performing products and countries.

🎯 RFM Segmentation: Automatically segment your customers using the powerful RFM (Recency, Frequency, Monetary) model. Identify your "Champions," "Loyal Customers," and customers who are "At Risk" to tailor your marketing strategies effectively.

🔄 Cohort Retention Analysis: Visualize customer loyalty with a retention heatmap. Understand how well you're retaining customers over time, cohort by cohort, to assess the health of your business.

🔮 Predictive CLV (Customer Lifetime Value): Forecast the future! This tab uses the BG/NBD and Gamma-Gamma models to predict the 12-month lifetime value of each customer, helping you identify and nurture your most valuable clients.

🛒 Product Affinity Analysis: Uncover hidden buying patterns with Market Basket Analysis. Discover which products are frequently purchased together to optimize product placement, promotions, and cross-selling opportunities.

📈 12-Month Sales Forecast: Using a SARIMA time-series model, the dashboard projects your sales for the next 12 months, complete with confidence intervals, to aid in inventory and financial planning.

💔 Churn Prediction: Proactively identify customers who are likely to churn. The dashboard uses a logistic regression model to calculate a churn probability for each customer, allowing you to intervene with targeted retention campaigns.

📋 Data Requirements
To use this dashboard, you must upload a CSV file containing transactional data. The file must include the following columns for all features to function correctly:

InvoiceNo: Unique identifier for each transaction.

StockCode: Unique identifier for each product.

Description: The name of the product.

Quantity: The number of items sold per transaction.

InvoiceDate: The date and time of the transaction (e.g., MM/DD/YYYY HH:MM).

UnitPrice: The price of a single unit of the product.

CustomerID: Unique identifier for each customer.

Country: The country where the purchase was made.

🚀 How to Run Locally
To run the RetailMind dashboard on your own machine, follow these steps:

Clone the Repository:

git clone [https://github.com/your-username/RetailMind-Analysis.git](https://github.com/your-username/RetailMind-Analysis.git)
cd RetailMind-Analysis

Create a Virtual Environment (Recommended):

python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

Install Dependencies:
The project's dependencies are listed in requirements.txt. Install them using pip:

pip install -r requirements.txt

Run the Streamlit App:
Execute the following command in your terminal:

streamlit run app.py

View the App:
Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

🛠️ Key Technologies & Libraries
Streamlit: For building the interactive web application.

Pandas: For data manipulation and analysis.

Plotly Express: For creating interactive data visualizations.

Lifetimes: For Customer Lifetime Value (CLV) modeling.

MLxtend: For Market Basket Analysis (Association Rules).

Statsmodels: For time-series forecasting (SARIMA).

Scikit-learn: For churn prediction modeling (Logistic Regression).
