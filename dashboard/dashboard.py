import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# GATHERING DATA
@st.cache_data
def load_data():
    customers = pd.read_csv('data/customers_dataset.csv')
    geolocation = pd.read_csv('data/geolocation_dataset.csv')
    order_items = pd.read_csv('data/order_items_dataset.csv')
    order_payments = pd.read_csv('data/order_payments_dataset.csv')
    order_reviews = pd.read_csv('data/order_reviews_dataset.csv')
    orders = pd.read_csv('data/orders_dataset.csv')
    product_category_translation = pd.read_csv('data/product_category_name_translation.csv')
    products = pd.read_csv('data/products_dataset.csv')
    sellers = pd.read_csv('data/sellers_dataset.csv')
    
    return customers, geolocation, order_items, order_payments, order_reviews, orders, product_category_translation, products, sellers

customers, geolocation, order_items, order_payments, order_reviews, orders, product_category_translation, products, sellers = load_data()

# ASSESSING DATA
st.title("E-Commerce Dataset Analysis")

st.write("## Defining Business Question")
st.write("""
- Question 1
1. What is the relationship between customer location and order delivery time?
- Question 2
2. Which product categories generate the highest revenue, and how does customer satisfaction vary across those categories?
- Question 3
3. How does seller location influence sales performance, and which states contribute the most to total sales?
""")

st.write("### Data Information:")
st.write("**Customers Dataset:**")
st.write(customers.info())
st.dataframe(customers)
st.write("**geolocation Dataset:**")
st.write(geolocation.info())
st.dataframe(geolocation)
st.write("**Order Item Dataset:**")
st.write(order_items.info())
st.dataframe(order_items)
st.write("**Order Payments Dataset:**")
st.write(order_payments.info())
st.dataframe(order_payments)
st.write("**Order Reviews Dataset:**")
st.write(order_reviews.info())
st.dataframe(order_reviews)
st.write("**Orders Dataset:**")
st.write(orders.info())
st.dataframe(orders)
st.write("**Product Category Translation Dataset:**")
st.write(product_category_translation.info())
st.dataframe(product_category_translation)
st.write("**Products Dataset:**")
st.write(products.info())
st.dataframe(products)
st.write("**Sellers Dataset:**")
st.write(sellers.info())
st.dataframe(sellers)

# CLEANING DATA
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders_clean = orders.dropna(subset=['order_delivered_customer_date'])

# Merge datasets for analysis
customer_orders = pd.merge(customers, orders_clean, on='customer_id', how='inner')
order_product_reviews = pd.merge(order_items, products[['product_id', 'product_category_name']], on='product_id', how='inner')
order_product_reviews = pd.merge(order_product_reviews, product_category_translation, on='product_category_name', how='inner')
order_product_reviews = pd.merge(order_product_reviews, order_reviews[['order_id', 'review_score']], on='order_id', how='left')

# EXPLORATION AND VISUALIZATION
st.write("### Question 1: Relationship between Customer Location and Delivery Time")

# Step 1: Calculate delivery time in days
customer_orders['delivery_time'] = (customer_orders['order_delivered_customer_date'] - customer_orders['order_purchase_timestamp']).dt.days
location_delivery = customer_orders.groupby(['customer_state'])['delivery_time'].mean().reset_index()

# Step 2: Visualize average delivery time by customer state
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='customer_state', y='delivery_time', data=location_delivery, ax=ax)
ax.set_title('Average Delivery Time by Customer State')
ax.set_xlabel('Customer State')
ax.set_ylabel('Average Delivery Time (days)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# Step 3: Visualize geolocation distribution
st.write("### Geolocation Distribution by State")
fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot(x='geolocation_lng', y='geolocation_lat', data=geolocation, hue='geolocation_state', ax=ax)
ax.set_title('Geolocation Distribution by State')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
st.pyplot(fig)

# Question 2: Revenue and Satisfaction by Product Category
st.write("### Question 2: Revenue and Satisfaction by Product Category")

# Step 1: Calculate revenue by product category
category_revenue = order_product_reviews.groupby('product_category_name_english')['price'].sum().reset_index()

# Step 2: Visualize total revenue by product category
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x='product_category_name_english', y='price', data=category_revenue.sort_values(by='price', ascending=False), ax=ax)
ax.set_title('Revenue by Product Category')
ax.set_xlabel('Product Category')
ax.set_ylabel('Total Revenue')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Step 3: Customer Satisfaction by Product Category
category_satisfaction = order_product_reviews.groupby('product_category_name_english')['review_score'].mean().reset_index()

# Step 4: Visualize average review score by product category
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x='product_category_name_english', y='review_score', data=category_satisfaction.sort_values(by='review_score', ascending=False), ax=ax)
ax.set_title('Customer Satisfaction by Product Category')
ax.set_xlabel('Product Category')
ax.set_ylabel('Average Review Score')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Question 3: Seller Performance by State
st.write("### Question 3: Seller Performance by State")

# Step 1: Merge seller information with order items
order_seller_info = pd.merge(order_items, sellers, on='seller_id', how='inner')

# Step 2: Calculate seller performance by state (total sales)
seller_performance = order_seller_info.groupby('seller_state').agg({
    'price': 'sum',
    'seller_id': 'count'
}).reset_index()

# Step 3: Visualize seller performance by state
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x='seller_state', y='price', data=seller_performance.sort_values(by='price', ascending=False), ax=ax)
ax.set_title('Total Sales by Seller State')
ax.set_xlabel('Seller State')
ax.set_ylabel('Total Sales')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

st.write("""
**Insight:**
- A bar plot visualizing average delivery time by state shows significant variations, likely influenced by geographical or logistical factors.
- Geolocation distribution visualization helps illustrate the geographical spread of customers.
- A bar plot of revenue by product category highlights which categories have the highest sales.
- A separate visualization of customer satisfaction by product category provides insight into areas where product quality might need improvement.
- Seller performance by state is visualized, showing which states contribute most to total sales, giving insights into regional seller activity.""")

# Conclusion
st.write("### Conclusion:")
st.write("""
Conclusion Question 1:
1. Customers in certain states experience significantly longer delivery times compared to others. Geographic factors, such as distance from distribution centers or the availability of logistics infrastructure in those regions, are likely the main contributors to this variation. By understanding these patterns, businesses can optimize their delivery strategies, such as improving logistics routes or opening new distribution centers, to reduce delivery times in regions with higher averages.
Conclusion Question 2:
2. Certain product categories, while generating the highest revenue, have lower customer satisfaction compared to others. This indicates that despite high sales, there may be quality or service issues affecting customer satisfaction in those categories. Businesses can use this insight to focus on improving product or service quality in these high-revenue categories to boost customer satisfaction while maintaining strong sales.
Conclusion Question 3:
3. Sellers from certain states contribute more significantly to total sales compared to others. This suggests that geographic factors, such as proximity to distribution hubs or better logistics infrastructure, may influence sales performance. This information can be used to optimize partnerships with sellers in high-performing states, while also exploring opportunities to improve sales in states with lower contributions.
""")