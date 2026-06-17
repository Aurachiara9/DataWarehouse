Here a short description of the contents of the files in this repository

--------------------------------------------------

The original dataset consists of 9 files in CSV format which are located in the folder called "csv"

In the folder called "Code" there are the following files: Olist_reconciled_database_schema.sql, DataQualityAssessment1.py, DataQualityAssessment2.py, MissingValues_Outliers.py, DataCleaning.py, creazioneDataWareHouse.py, Olist_data_warehouse_schema.sql, progetto.twb

-------------------------------

Olist_reconciled_database_schema.sql

The file Olist_reconciled_database_schema.sql contains the code to build the schema for the reconciled database.

--------------------------------------------------

DataQualityAssessment1.py 

This file performs an initial data quality audit on the raw Olist csv files before any cleaning operations are applied. The output files of the execution of DataQualityAssessment1.py are in the folder called "InitialDQ" .

The DQAReport class is implemented to evaluate 5 core metrics (Completeness, Uniqueness, Validity, Consistency, Timeliness) scoring tables from 0.0 to 1.0 for each of the metrics. The scores of the table are saved in the following CSV files: dq_scorecard_category.csv, dq_scorecard_seller.csv, dq_scorecard_customer.csv, dq_scorecard_product.csv, dq_scoecard_order_item.csv, dq_scorecard_order.csv

In addition, quality report charts are generated.

For each table, a two-panel image is generated that contains: a Missing Values Matrix and a horizontal bar chart showing the percentage scores for each metric. The images that are generated are called: dq_report_ProductCategoryName_table.png, dq_report_Seller_table.png, dq_report_Customer_table.png, dq_report_Product_table.png, dq_report_OrderItem_table.png, dq_report_Order_table.png

For each table, the following have also been created:
-  a chart showing the exact percentage of null values in attributes with missing data;
- for each attribute for which the table contains at least one invalid value, a bar chart is generated showing the exact number of rows that violate the validity rules and the exact number of rows containing only valid values. 
The charts that are created are named: dq_plot_ProductCategoryName_table.png, dq_plot_Seller_table.png, dq_plot_Customer_table.png, dq_plot_Product_table.png, dq_plot_OrderItem_table.png, dq_plot_Order_table.png

A file named dq_comparison_pivot.csv has also been created; this is a summary report that compares the scores of all the tables to immediately identify the most “problematic” ones.

Finally, the file dq_comparison_plot.png is created, containing a bar chart that shows the final comparison of data quality across all the analyzed tables.

---------------------------------

DataQualityAssessment2.py 

The files dq_scorecard_category.csv, dq_scorecard_seller.csv, dq_scorecard_customer.csv, dq_scorecard_product.csv, dq_scorecard_order_item.csv, and dq_scorecard_order.csv located in the “InitialDQ” folder 
are used by the local Llama 3.2:3b model, trained to act as an expert Data Quality Analyst, to generate a formatted text file for each table that includes:
a 2-3 sentence summary for business managers, a clear list of the issues found and some recommended cleaning steps.
The generated files are placed in the folder called "AuditReport" in the folder called “LLMResults” and are called: dq_audit_customer.md, dq_audit_order.md, dq_audit_orderitem.md, ddq_audit_product.md,
dq_audit_productcategoryname.md, dq_audit_seller.md

Note: All generated files have been subjected to human checks.

--------------------------------

MissingValues_Outliers.py    

The mechanisms of missing data are analyzed using the chi-squared test of independence, logistic regression, and the proxy test.
Outliers were identified using the modified Z-score method, the QR Fence method, and the Isolation Forest method.
Combining multiple methods and using a consensus threshold (i.e. a row is flagged as a ’consensus outlier’ only if it is identified as anomalous by at least 2 methods) provide reliable results.

The file `orders_missing_by_status.png` is generated, containing three bar charts. These charts show how the percentage of missing values for the attributes `order_approved_at`, `order_delivered_carrier_date`, and `order_delivered_customer_date` (all three of which are attributes of the `Order` table) varies depending on the value of the `order_status` attribute.

The tables for which missing data was detected are Product and Order. For these tables, the files Order_missing_summary.csv and Product_missing_summary.csv were generated, containing a summary that maps the following fields for each column with null values:
- column: The name of the attribute containing the missing data.
- missing_count: The exact number of rows with null values in that column.
missing_pct: The percentage of missing data relative to the total number of rows in the table.

The files generated are saved in the folder called "Cleaning1" in the folder called "Cleaning".

--------------------------------

DataCleaning.py

This file contains all the dataset cleanup operations (i.e. handling missing values, handling invalid values, handling anomalies and handling inconsistencies).
The CSV files that make up the cleaned dataset are saved in the folder called "Cleaning2" within the folder called "Cleaning".

-------------------------------

creazioneDataWareHouse.py

Based on the data contained in the CSV files located in the folder "Cleaning2" within the folder "Cleaning", the creazioneDataWareHouse.py file contains the code to create CSV files, 
each of which contains the data to populate a table of the star schema of the "data warehouse" DBMS.
The CSV files created are saved in the folder called "DataWarehouse".

-------------------------------

Olist_data_warehouse_schema.sql 

The file Olist_data_warehouse_schema.sql contains the code to build the schema of the star schema for the data warehouse.

-------------------------------

progetto.twb --> It is in the folder called "TableauFiles" and it is the Tableau Workbook file containing the sheets, the interactive dashboards and the story designed for the data visualization phase.




















