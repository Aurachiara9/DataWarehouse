import os
import pandas as pd
import numpy as np

BASE_PATH = r"C:\Users\acer\Desktop\Magistrale 1-02-2026\I anno\Data Ware House\Olist\Cleaning\Cleaning2"
OUTPUT_PATH = r"C:\Users\acer\Desktop\Magistrale 1-02-2026\I anno\Data Ware House\Olist\DataWarehouse"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print(f"Folder created successfully: {OUTPUT_PATH}")
else:
    print(f"The destination folder already exists: {OUTPUT_PATH}")

np.random.seed(42)

print("Loading source files...")

items = pd.read_csv(os.path.join(BASE_PATH, 'cleaned_olist_order_items.csv'))
orders = pd.read_csv(os.path.join(BASE_PATH, 'cleaned_olist_orders.csv'))
products = pd.read_csv(os.path.join(BASE_PATH, 'cleaned_olist_products.csv'))
categories = pd.read_csv(os.path.join(BASE_PATH, 'cleaned_olist_category.csv'))
customers = pd.read_csv(os.path.join(BASE_PATH, 'cleaned_olist_customers.csv'))
sellers = pd.read_csv(os.path.join(BASE_PATH, 'cleaned_olist_sellers.csv'))


df_base = items.merge(orders, on='order_id', how='inner')
df_base = df_base.merge(products, on='product_id', how='left')
df_base = df_base.merge(customers, on='customer_id', how='left')
df_base = df_base.merge(sellers, on='seller_id', how='left')


date_cols = [
    'order_purchase_timestamp', 
    'order_estimated_delivery_date', 
    'order_approved_at', 
    'order_delivered_carrier_date', 
    'order_delivered_customer_date'
]
for col in date_cols:
    df_base[col] = pd.to_datetime(df_base[col])

print("Generating dimensional tables (with Primary Key in the first column)...")

# PurchaseTimestamps.csv
unique_purchase = pd.DataFrame({'order_purchase_timestamp': df_base['order_purchase_timestamp'].unique()})
unique_purchase['ID_order_purchase_timestamp'] = np.random.choice(np.arange(100000, 100000 + len(unique_purchase)), len(unique_purchase), replace=False)
unique_purchase['order_purchase_year'] = unique_purchase['order_purchase_timestamp'].dt.year
unique_purchase['order_purchase_month'] = unique_purchase['order_purchase_timestamp'].dt.month

unique_purchase = unique_purchase[['ID_order_purchase_timestamp', 'order_purchase_timestamp', 'order_purchase_month', 'order_purchase_year']]
unique_purchase.to_csv(os.path.join(OUTPUT_PATH, 'PurchaseTimestamps.csv'), index=False)

# EstimatedDeliveryDates.csv
unique_estimated = pd.DataFrame({'order_estimated_delivery_date': df_base['order_estimated_delivery_date'].unique()})
unique_estimated['ID_order_estimated_delivery_date'] = np.random.choice(np.arange(100000, 100000 + len(unique_estimated)), len(unique_estimated), replace=False)
unique_estimated['order_estimated_delivery_year'] = unique_estimated['order_estimated_delivery_date'].dt.year
unique_estimated['order_estimated_delivery_month'] = unique_estimated['order_estimated_delivery_date'].dt.month
unique_estimated = unique_estimated[['ID_order_estimated_delivery_date', 'order_estimated_delivery_date', 'order_estimated_delivery_month', 'order_estimated_delivery_year']]
unique_estimated.to_csv(os.path.join(OUTPUT_PATH, 'EstimatedDeliveryDates.csv'), index=False)

# ApprovalDates.csv
unique_approval = pd.DataFrame({'order_approved_at': df_base['order_approved_at'].unique()})
unique_approval['ID_order_approved_at'] = np.random.choice(np.arange(100000, 100000 + len(unique_approval)), len(unique_approval), replace=False)
unique_approval['order_approved_at_year'] = unique_approval['order_approved_at'].dt.year
unique_approval['order_approved_at_month'] = unique_approval['order_approved_at'].dt.month
unique_approval = unique_approval[['ID_order_approved_at', 'order_approved_at', 'order_approved_at_month', 'order_approved_at_year']]
unique_approval.to_csv(os.path.join(OUTPUT_PATH, 'ApprovalDates.csv'), index=False)

# DeliveredCarrierDates.csv 
unique_carrier = pd.DataFrame({'order_delivered_carrier_date': df_base['order_delivered_carrier_date'].unique()})
unique_carrier['ID_order_delivered_carrier_date'] = np.random.choice(np.arange(100000, 100000 + len(unique_carrier)), len(unique_carrier), replace=False)
unique_carrier['order_delivered_carrier_year'] = unique_carrier['order_delivered_carrier_date'].dt.year
unique_carrier['order_delivered_carrier_month'] = unique_carrier['order_delivered_carrier_date'].dt.month
unique_carrier = unique_carrier[['ID_order_delivered_carrier_date', 'order_delivered_carrier_date', 'order_delivered_carrier_month', 'order_delivered_carrier_year']]
unique_carrier.to_csv(os.path.join(OUTPUT_PATH, 'DeliveredCarrierDates.csv'), index=False)

# ActualDeliveryDates.csv 
unique_actual = pd.DataFrame({'order_delivered_customer_date': df_base['order_delivered_customer_date'].unique()})
unique_actual['ID_order_delivered_customer_date'] = np.random.choice(np.arange(100000, 100000 + len(unique_actual)), len(unique_actual), replace=False)
unique_actual['order_delivered_customer_year'] = unique_actual['order_delivered_customer_date'].dt.year
unique_actual['order_delivered_customer_month'] = unique_actual['order_delivered_customer_date'].dt.month
unique_actual = unique_actual[['ID_order_delivered_customer_date', 'order_delivered_customer_date', 'order_delivered_customer_month', 'order_delivered_customer_year']]
unique_actual.to_csv(os.path.join(OUTPUT_PATH, 'ActualDeliveryDates.csv'), index=False)

# ProductCategories.csv 
product_categories_df = categories.copy()
product_categories_df['ID_product_category_name'] = np.random.choice(np.arange(1000, 1000 + len(product_categories_df)), len(product_categories_df), replace=False)
product_categories_df = product_categories_df[['ID_product_category_name', 'product_category_name', 'product_category_name_english']]
product_categories_df.to_csv(os.path.join(OUTPUT_PATH, 'ProductCategories.csv'), index=False)


# CustomerInformation.csv
unique_customers = df_base[['customer_city', 'customer_state']].drop_duplicates().reset_index(drop=True)
unique_customers['ID_customer_city'] = np.random.choice(np.arange(200000, 200000 + len(unique_customers)), len(unique_customers), replace=False)
unique_customers = unique_customers[['ID_customer_city', 'customer_city', 'customer_state']]
unique_customers.to_csv(os.path.join(OUTPUT_PATH, 'CustomerInformation.csv'), index=False)

# SellerInformation.csv 
unique_sellers = df_base[['seller_city', 'seller_state']].drop_duplicates().reset_index(drop=True)
unique_sellers['ID_seller_city'] = np.random.choice(np.arange(300000, 300000 + len(unique_sellers)), len(unique_sellers), replace=False)
unique_sellers = unique_sellers[['ID_seller_city', 'seller_city', 'seller_state']]
unique_sellers.to_csv(os.path.join(OUTPUT_PATH, 'SellerInformation.csv'), index=False)


print("Construction of the Fact Table...")

fact_table = df_base.copy()

fact_table = fact_table.merge(product_categories_df, on='product_category_name', how='left')
fact_table = fact_table.merge(unique_actual, on='order_delivered_customer_date', how='left')
fact_table = fact_table.merge(unique_customers, on=['customer_city', 'customer_state'], how='left')
fact_table = fact_table.merge(unique_sellers, on=['seller_city', 'seller_state'], how='left')
fact_table = fact_table.merge(unique_purchase, on='order_purchase_timestamp', how='left')
fact_table = fact_table.merge(unique_estimated, on='order_estimated_delivery_date', how='left')
fact_table = fact_table.merge(unique_approval, on='order_approved_at', how='left')
fact_table = fact_table.merge(unique_carrier, on='order_delivered_carrier_date', how='left')



pk_columns = [
    'ID_product_category_name', 'ID_order_delivered_customer_date', 'ID_customer_city', 
    'ID_seller_city', 'ID_order_purchase_timestamp', 'ID_order_estimated_delivery_date',  
    'ID_order_approved_at', 'ID_order_delivered_carrier_date', 'order_item_id'
]
fact_columns = pk_columns + ['price', 'freight_value']
fact_table = fact_table[fact_columns]

fact_table = fact_table.drop_duplicates(subset=pk_columns)

for col in pk_columns:
    fact_table[col] = fact_table[col].astype(int)


fact_table.to_csv(os.path.join(OUTPUT_PATH, 'FactTable.csv'), index=False)

print(f"Process completed! All files are saved in: {OUTPUT_PATH}")