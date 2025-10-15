import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data, ConvergenceError
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


st.set_page_config(
    page_title="RetailMind Analysis",
    page_icon="🔮",
    layout="wide"
)

st.markdown("""
<style>
    /* Main titles */
    .title-text {
        font-weight: bold;
        padding: 1rem 0;
    }
    /* KPI Card styling */
    .kpi-card {
        background-color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #2e2e2e;
        text-align: center;
    }
    .kpi-card h3 {
        margin: 0;
        font-size: 1.2rem;
        color: #a0a0a0;
    }
    .kpi-card p {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    /* Center align tabs */
    .stTabs [role="tablist"] {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string."""
    return df.to_csv(index=False).encode('utf-8')


@st.cache_data
def load_and_clean_data(uploaded_file):
    """
    Loads data from an uploaded file, cleans it, and returns a DataFrame.
    """
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

def get_rfm_segment(r, f, m):
    """Assigns a segment name based on R, F, M scores."""
    if r >= 4 and f >= 4: return 'Champions'
    if r >= 2 and f >= 3: return 'Loyal Customers'
    if r >= 3 and f >= 1: return 'Potential Loyalists'
    if r >= 4 and f >= 1: return 'Recent Customers'
    if r >= 3 and f >= 3: return 'Promising'
    if r >= 2 and f >= 2: return 'Customers Needing Attention'
    if r >= 2 and f >= 1: return 'About to Sleep'
    if r < 2 and f >= 2: return 'At Risk'
    if r < 2 and f < 2: return 'Hibernating'
    if r < 2 and f >= 1: return "Can't Lose Them"
    return 'Lost'



@st.cache_data
def calculate_metrics(df):
    """
    Calculates RFM metrics, segments, and cohort analysis data.
    """
    
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

    r_labels, f_labels, m_labels = range(4, 0, -1), range(1, 5), range(1, 5)
    rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), q=4, labels=r_labels)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=f_labels)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=4, labels=m_labels)
    
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['Segment'] = rfm.apply(lambda x: get_rfm_segment(x['R_Score'], x['F_Score'], x['M_Score']), axis=1)

    
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    df['CohortMonth'] = df.groupby('CustomerID')['InvoiceMonth'].transform('min')
    
    def get_cohort_index(df, cohort_month_col='CohortMonth', invoice_month_col='InvoiceMonth'):
        year_diff = df[invoice_month_col].dt.year - df[cohort_month_col].dt.year
        month_diff = df[invoice_month_col].dt.month - df[cohort_month_col].dt.month
        return year_diff * 12 + month_diff + 1

    df['CohortIndex'] = get_cohort_index(df)
    
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    cohort_count = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='CustomerID')
    
    cohort_size = cohort_count.iloc[:, 0]
    retention_matrix = cohort_count.divide(cohort_size, axis=0)
    retention_matrix = retention_matrix.round(3) * 100
    retention_matrix.index = retention_matrix.index.strftime('%Y-%m')

    return rfm, retention_matrix


@st.cache_data
def calculate_clv(df):
    """
    Fits BG/NBD and Gamma-Gamma models to predict CLV.
    Handles ConvergenceError and empty data gracefully.
    """
    clv_df = summary_data_from_transaction_data(
        df,
        customer_id_col='CustomerID',
        datetime_col='InvoiceDate',
        monetary_value_col='TotalPrice'
    )
    clv_df = clv_df[clv_df['monetary_value'] > 0]

    if clv_df.empty:
        return None

    try:
        bgf = BetaGeoFitter(penalizer_coef=0.1)
        bgf.fit(clv_df['frequency'], clv_df['recency'], clv_df['T'])

        ggf = GammaGammaFitter(penalizer_coef=0.1)
        ggf.fit(clv_df['frequency'], clv_df['monetary_value'])

        clv_df['predicted_clv'] = ggf.customer_lifetime_value(
            bgf,
            clv_df['frequency'],
            clv_df['recency'],
            clv_df['T'],
            clv_df['monetary_value'],
            time=12,
            discount_rate=0.01
        )
        return clv_df.sort_values(by="predicted_clv", ascending=False)
    
    except ConvergenceError:
        return None


@st.cache_data
def market_basket_analysis(df, selected_product):
    """
    Performs market basket analysis to find frequently co-purchased items.
    Optimized to prevent memory errors.
    """
    
    analysis_df = df.tail(20000)
    
    try:
        basket = analysis_df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
        
        def encode_units(x):
            return x > 0

        basket_sets = basket.applymap(encode_units)
        frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        
        product_rules = rules[rules['antecedents'].apply(lambda x: selected_product in str(x))].sort_values('lift', ascending=False)
        return product_rules
    except MemoryError:
        return "MemoryError"
    except Exception:
        return None


@st.cache_data
def calculate_forecast(df):
    sales_data = df.set_index('InvoiceDate')['TotalPrice'].resample('MS').sum()
    
    if len(sales_data) < 24: # Need enough data for seasonal model
        return None, None

    try:
        model = sm.tsa.statespace.SARIMAX(sales_data,
                                          order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12),
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=12)
        forecast_df = forecast.summary_frame()
        return sales_data, forecast_df
    except:
        return None, None


@st.cache_data
def calculate_churn_prediction(rfm_df):
    """
    Trains a logistic regression model to predict churn probability.
    """
    churn_df = rfm_df.copy()
    churn_df['Churn'] = churn_df['Segment'].apply(lambda x: 1 if x in ['Hibernating', 'Lost', 'At Risk'] else 0)
    
    X = churn_df[['Recency', 'Frequency', 'Monetary']]
    y = churn_df['Churn']
    
    if len(y.unique()) < 2:
        return None

    try:
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X, y)
        churn_df['ChurnProbability'] = model.predict_proba(X)[:, 1]
        return churn_df.sort_values('ChurnProbability', ascending=False)
    except Exception:
        return None



st.title("RetailMind Analysis")
st.markdown("---")


st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is None:
    st.header("Welcome to the RetailMind Analysis🔮")
    st.info("To get started, please upload your sales data in CSV format using the uploader in the sidebar.")
    
    with st.expander("Click here to see the required data format"):
        st.write("""
        Your CSV file should contain the following columns for the dashboard to function correctly:
        - **InvoiceNo**: A unique identifier for each transaction.
        - **StockCode**: A unique identifier for each product.
        - **Description**: The name of the product.
        - **Quantity**: The number of items sold in the transaction.
        - **InvoiceDate**: The date and time of the transaction (e.g., `MM/DD/YYYY HH:MM`).
        - **UnitPrice**: The price of a single item.
        - **CustomerID**: A unique identifier for each customer.
        - **Country**: The country where the transaction occurred.
        """)
        
    st.stop()


try:
    df = load_and_clean_data(uploaded_file)
except Exception as e:
    st.error(f"Error processing the file: {e}")
    st.stop()


st.sidebar.header("Filters")
country_options = ['All'] + list(df['Country'].unique())
selected_country = st.sidebar.multiselect(
    "Select Country",
    options=country_options,
    default='All'
)

if 'All' in selected_country:
    country_filter = df['Country'].unique()
else:
    country_filter = selected_country

min_date = df['InvoiceDate'].min().date()
max_date = df['InvoiceDate'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

filtered_df = df[
    (df['Country'].isin(country_filter)) &
    (df['InvoiceDate'].dt.date >= start_date) &
    (df['InvoiceDate'].dt.date <= end_date)
]


if not filtered_df.empty:
    rfm_df, retention_matrix = calculate_metrics(filtered_df)
    
    tab_list = ["📊 Sales Overview", "🎯 RFM Segmentation", "🔄 Cohort Retention", "🔮 Predictive (CLV)", "🛒 Product Affinity", "📈 Sales Forecast", "💔 Churn Prediction"]
    tabs = st.tabs(tab_list)

    
    with tabs[0]:
        st.header("Key Performance Indicators")
        total_sales = int(filtered_df['TotalPrice'].sum())
        total_orders = filtered_df['InvoiceNo'].nunique()
        unique_customers = filtered_df['CustomerID'].nunique()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="kpi-card"><h3>Total Sales</h3><p>${total_sales:,}</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi-card"><h3>Total Orders</h3><p>{total_orders:,}</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="kpi-card"><h3>Unique Customers</h3><p>{unique_customers:,}</p></div>', unsafe_allow_html=True)
            
        st.divider()

        
        st.subheader("Geospatial Sales Distribution")
        map_data = filtered_df.groupby('Country')['TotalPrice'].sum().reset_index()
        fig_map = px.choropleth(
            map_data,
            locations="Country",
            locationmode="country names",
            color="TotalPrice",
            hover_name="Country",
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Global Sales by Country"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("---")
        st.divider()


        st.subheader("Sales Trends and Top Products/Countries")
        monthly_sales = filtered_df.copy()
        monthly_sales['InvoiceMonth'] = monthly_sales['InvoiceDate'].dt.to_period('M').astype(str)
        monthly_sales = monthly_sales.groupby('InvoiceMonth')['TotalPrice'].sum().reset_index()
        fig_monthly = px.line(monthly_sales, x='InvoiceMonth', y='TotalPrice', title="Total Sales Over Time")
        st.plotly_chart(fig_monthly, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            top_products = filtered_df.groupby('Description')['Quantity'].sum().nlargest(10).reset_index()
            fig_prod = px.bar(top_products.sort_values('Quantity', ascending=True), 
                              x='Quantity', y='Description', orientation='h', title="Top 10 Products")
            st.plotly_chart(fig_prod, use_container_width=True)
        with c2:
            country_sales = filtered_df.groupby('Country')['TotalPrice'].sum().nlargest(10).reset_index()
            fig_country = px.bar(country_sales.sort_values('TotalPrice', ascending=True), 
                                 x='TotalPrice', y='Country', orientation='h', title="Top 10 Countries by Sales")
            st.plotly_chart(fig_country, use_container_width=True)

    
    with tabs[1]:
        st.header("Customer Segmentation (RFM Analysis)")
        with st.expander("What is RFM Analysis?"):
            st.write("""
                RFM analysis is a marketing technique used to quantitatively rank and group customers based on their transaction history.
                - **Recency:** How recently a customer has made a purchase.
                - **Frequency:** How often they purchase.
                - **Monetary Value:** How much money they spend.
            """)
        
        st.subheader("Explore Customer Segments")
        segment_options = ['All Segments'] + list(rfm_df['Segment'].unique())
        chosen_segment = st.selectbox("Select a Segment to Drill Down", segment_options)

        if chosen_segment == 'All Segments':
            rfm_display_df = rfm_df
        else:
            rfm_display_df = rfm_df[rfm_df['Segment'] == chosen_segment]

        c1, c2 = st.columns((2,3))
        with c1:
            segment_counts = rfm_df['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            fig_segments = px.bar(segment_counts.sort_values('Count', ascending=False), 
                                  x='Count', y='Segment', orientation='h', title="Customer Segment Distribution")
            st.plotly_chart(fig_segments, use_container_width=True)

        with c2:
            fig_rfm_scatter = px.scatter(rfm_display_df, x='Recency', y='Frequency', color='Segment',
                                         size='Monetary', hover_name=rfm_display_df.index,
                                         title=f"Displaying: {chosen_segment}")
            st.plotly_chart(fig_rfm_scatter, use_container_width=True)

        st.subheader("Customer Data for Selected Segment")
        st.dataframe(rfm_display_df)

        csv = convert_df_to_csv(rfm_display_df)
        st.download_button(
            label="Download Data for Selected Segment",
            data=csv,
            file_name=f"{chosen_segment}_customers.csv",
            mime='text/csv',
        )


    
    with tabs[2]:
        st.header("Customer Retention Cohort Analysis")
        with st.expander("What is Cohort Analysis?"):
            st.write("""
                Cohort analysis is a behavioral analytics tool that breaks down data into groups of people with common characteristics over time. 
                This heatmap shows the percentage of customers from an acquisition cohort (row) who made another purchase in the months following (column).
            """)
        
        fig_cohort = go.Figure(data=go.Heatmap(
            z=retention_matrix.values,
            x=[f"Month {i}" for i in retention_matrix.columns],
            y=retention_matrix.index,
            colorscale='Viridis'
        ))
        fig_cohort.update_layout(title='Monthly Customer Retention Rate (%)',
                                 xaxis_title='Months After First Purchase',
                                 yaxis_title='Acquisition Month')
        st.plotly_chart(fig_cohort, use_container_width=True)

    
    with tabs[3]:
        st.header("Customer Lifetime Value (CLV) Prediction")
        
        clv_df = calculate_clv(filtered_df)

        if clv_df is not None:
            with st.expander("What is CLV?"):
                st.write("""
                    Customer Lifetime Value (CLV) is a prediction of the net profit attributed to the entire future relationship with a customer. 
                    This model predicts the total purchase value of a customer over the next 12 months.
                """)
            
            st.subheader("Top 10 Customers by Predicted CLV")
            top_clv = clv_df.head(10).reset_index()
            fig_clv = px.bar(top_clv.sort_values('predicted_clv', ascending=True), 
                             x='predicted_clv', y='CustomerID', orientation='h',
                             title="Top 10 Customers by Predicted CLV (Next 12 Months)")
            fig_clv.update_layout(xaxis_title="Predicted CLV ($)", yaxis_title="Customer ID")
            st.plotly_chart(fig_clv, use_container_width=True)

            st.subheader("Full Customer CLV Data")
            st.dataframe(clv_df.reset_index())

            csv_clv = convert_df_to_csv(clv_df.reset_index())
            st.download_button(
                label="Download CLV Data",
                data=csv_clv,
                file_name="customer_clv.csv",
                mime='text/csv',
            )
        else:
            st.warning("The CLV prediction model could not converge for the selected data filters. Please try a different date range or country selection.")

    
    with tabs[4]:
        st.header("Product Affinity (Market Basket Analysis)")
        with st.expander("What is Product Affinity?"):
            st.write("""
                Market Basket Analysis is a technique used to uncover associations between items. It works by looking for combinations of items that occur together frequently in transactions.
                This model helps answer the question: "If a customer buys Product A, what are they likely to buy next?"
            """)
        st.info("Analysis is run on the last 20,000 transactions for performance.")

        product_list = filtered_df['Description'].unique()
        selected_product = st.selectbox("Select a Product", product_list)

        if selected_product:
            rules_df = market_basket_analysis(filtered_df, selected_product)

            if isinstance(rules_df, str) and rules_df == "MemoryError":
                 st.error("The analysis could not be completed due to a memory error. Please select a smaller date range or fewer countries.")
            elif rules_df is not None and not rules_df.empty:
                st.subheader(f"Products frequently bought with '{selected_product}'")
                
                
                rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
                
                st.dataframe(rules_df[['consequents', 'lift', 'confidence']].head(10))
            else:
                st.warning("No significant product associations found for the selected product and filters. Try a more common product or wider filters.")
    
    
    with tabs[5]:
        st.header("12-Month Sales Forecast")
        with st.expander("What is Sales Forecasting?"):
            st.write("""
                Time series forecasting is a method for predicting future values based on previously observed values. 
                This model uses the historical sales data (from the selected filters) to forecast sales for the next 12 months.
            """)
        
        historical_data, forecast_data = calculate_forecast(filtered_df)

        if historical_data is not None and forecast_data is not None:
            fig_forecast = go.Figure()
            
            
            fig_forecast.add_trace(go.Scatter(x=historical_data.index.to_timestamp(), y=historical_data, mode='lines', name='Historical Sales'))
            
            
            fig_forecast.add_trace(go.Scatter(x=forecast_data.index.to_timestamp(), y=forecast_data['mean'], mode='lines', name='Forecasted Sales', line=dict(dash='dash')))
            
            
            fig_forecast.add_trace(go.Scatter(x=forecast_data.index.to_timestamp(), y=forecast_data['mean_ci_upper'], fill='tonexty', mode='none', name='Upper Confidence Interval', line=dict(color='rgba(0,0,0,0)')))
            fig_forecast.add_trace(go.Scatter(x=forecast_data.index.to_timestamp(), y=forecast_data['mean_ci_lower'], fill='tonexty', mode='none', name='Lower Confidence Interval', line=dict(color='rgba(0,0,0,0)')))

            fig_forecast.update_layout(title="Sales Forecast with Confidence Interval",
                                       xaxis_title="Date",
                                       yaxis_title="Total Sales")
            
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.subheader("Forecast Data")
            st.dataframe(forecast_data)
        else:
            st.warning("Could not generate a forecast. The model requires at least 24 months of historical data with consistent sales. Please select a wider date range.")
            
    
    with tabs[6]:
        st.header("Customer Churn Prediction")
        with st.expander("What is Churn Prediction?"):
            st.write("""
                Churn prediction is the process of identifying customers who are likely to stop using a service or product. This model uses customer transaction history (Recency, Frequency, Monetary value) to predict the probability of a customer churning.
            """)

        churn_data = calculate_churn_prediction(rfm_df)

        if churn_data is not None:
            st.subheader("Top 10 Customers at Risk of Churning")
            top_churn = churn_data.head(10).reset_index()
            
            fig_churn = px.bar(top_churn.sort_values('ChurnProbability', ascending=True),
                               x='ChurnProbability',
                               y='CustomerID',
                               orientation='h',
                               title="Top 10 Customers by Churn Probability")
            fig_churn.update_layout(xaxis_title="Churn Probability", yaxis_title="Customer ID")
            st.plotly_chart(fig_churn, use_container_width=True)

            st.subheader("Full Customer Churn Data")
            st.dataframe(churn_data.reset_index())
            
            csv_churn = convert_df_to_csv(churn_data.reset_index())
            st.download_button(
                label="Download Churn Data",
                data=csv_churn,
                file_name="customer_churn_data.csv",
                mime='text/csv',
            )

        else:
            st.warning("Could not train the churn prediction model. This can happen if all customers in the filtered data belong to the same category (e.g., all are active).")


else:
    st.warning("No data available for the selected filters. Please adjust your selection.")

