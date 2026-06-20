-- 1. CREAZIONE DELLE TABELLE DI DIMENSIONE 

CREATE TABLE "ProductCategories" (
    "ID_product_category_name" INT PRIMARY KEY,
    product_category_name VARCHAR(100) NOT NULL UNIQUE,
    product_category_name_english VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE "ActualDeliveryDates" (
    "ID_order_delivered_customer_date" INT PRIMARY KEY,
    order_delivered_customer_date TIMESTAMP NOT NULL UNIQUE,
    order_delivered_customer_month INT NOT NULL,
    order_delivered_customer_year INT NOT NULL
);

CREATE TABLE "CustomerInformation" (
    "ID_customer_city" INT PRIMARY KEY,
    customer_city VARCHAR(100) NOT NULL,
    customer_state CHAR(2) NOT NULL,
    CONSTRAINT "UQ_Customer_City_State" UNIQUE (customer_city, customer_state)
);

CREATE TABLE "SellerInformation" (
    "ID_seller_city" INT PRIMARY KEY,
    seller_city VARCHAR(100) NOT NULL,
    seller_state CHAR(2) NOT NULL,
    CONSTRAINT "UQ_Seller_City_State" UNIQUE (seller_city, seller_state)
);

CREATE TABLE "PurchaseTimestamps" (
    "ID_order_purchase_timestamp" INT PRIMARY KEY,
    order_purchase_timestamp TIMESTAMP NOT NULL UNIQUE,
    order_purchase_month INT NOT NULL,
    order_purchase_year INT NOT NULL
);

CREATE TABLE "EstimatedDeliveryDates" (
    "ID_order_estimated_delivery_date" INT PRIMARY KEY,
    order_estimated_delivery_date TIMESTAMP NOT NULL UNIQUE,
    order_estimated_delivery_month INT NOT NULL,
    order_estimated_delivery_year INT NOT NULL
);

CREATE TABLE "ApprovalDates" (
    "ID_order_approved_at" INT PRIMARY KEY,
    order_approved_at TIMESTAMP NOT NULL UNIQUE,
    order_approved_at_month INT NOT NULL,
    order_approved_at_year INT NOT NULL
);

CREATE TABLE "DeliveredCarrierDates" (
    "ID_order_delivered_carrier_date" INT PRIMARY KEY,
    order_delivered_carrier_date TIMESTAMP NOT NULL UNIQUE,
    order_delivered_carrier_month INT NOT NULL,
    order_delivered_carrier_year INT NOT NULL
);


CREATE TABLE "OrderItemInformation" (
    "ID_order_item" INT PRIMARY KEY,
    order_item_id INT NOT NULL UNIQUE
);

-- 2. CREAZIONE DELLA TABELLA DEI FATTI 

CREATE TABLE "FactTable" (
    -- Foreign Keys
    "ID_product_category_name" INT NOT NULL REFERENCES "ProductCategories"("ID_product_category_name"),
    "ID_order_delivered_customer_date" INT NOT NULL REFERENCES "ActualDeliveryDates"("ID_order_delivered_customer_date"),
    "ID_customer_city" INT NOT NULL REFERENCES "CustomerInformation"("ID_customer_city"),
    "ID_seller_city" INT NOT NULL REFERENCES "SellerInformation"("ID_seller_city"),
    "ID_order_purchase_timestamp" INT NOT NULL REFERENCES "PurchaseTimestamps"("ID_order_purchase_timestamp"),
    "ID_order_estimated_delivery_date" INT NOT NULL REFERENCES "EstimatedDeliveryDates"("ID_order_estimated_delivery_date"),
    "ID_order_approved_at" INT NOT NULL REFERENCES "ApprovalDates"("ID_order_approved_at"),
    "ID_order_delivered_carrier_date" INT NOT NULL REFERENCES "DeliveredCarrierDates"("ID_order_delivered_carrier_date"),
    "ID_order_item" INT NOT NULL REFERENCES "OrderItemInformation"("ID_order_item"),
    
    -- Measures
    price NUMERIC NOT NULL,
    freight_value NUMERIC NOT NULL,
    
    PRIMARY KEY (
        "ID_product_category_name", "ID_order_delivered_customer_date", 
        "ID_customer_city", "ID_seller_city", "ID_order_purchase_timestamp", 
        "ID_order_estimated_delivery_date", "ID_order_approved_at", "ID_order_delivered_carrier_date",
        "ID_order_item"
    )
);