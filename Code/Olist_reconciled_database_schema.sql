-- Tabella Customer
CREATE TABLE "Customer" (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_city VARCHAR(100) NOT NULL,
    customer_state CHAR(2) NOT NULL
);

-- Tabella Seller
CREATE TABLE "Seller" (
    seller_id VARCHAR(50) PRIMARY KEY,
    seller_city VARCHAR(100) NOT NULL,
    seller_state CHAR(2) NOT NULL
);

-- Tabella ProductCategoryName
CREATE TABLE "ProductCategoryName" (
    product_category_name VARCHAR(100) PRIMARY KEY,
    product_category_name_english VARCHAR(100) NOT NULL
);

-- Tabella Product
CREATE TABLE "Product" (
    product_id VARCHAR(50) PRIMARY KEY,
    product_category_name VARCHAR(100) NOT NULL REFERENCES "ProductCategoryName"(product_category_name)
);

-- Tabella Order
CREATE TABLE "Order" (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES "Customer"(customer_id),
    order_status VARCHAR(20),
    order_purchase_timestamp TIMESTAMP NOT NULL,
    order_approved_at TIMESTAMP NOT NULL,
    order_delivered_carrier_date TIMESTAMP NOT NULL,
    order_delivered_customer_date TIMESTAMP NOT NULL,
    order_estimated_delivery_date TIMESTAMP NOT NULL
);

-- Tabella OrderItem
CREATE TABLE "OrderItem" (
    order_id VARCHAR(50) NOT NULL REFERENCES "Order"(order_id),
    order_item_id INT,
    product_id VARCHAR(50) NOT NULL REFERENCES "Product"(product_id),
    seller_id VARCHAR(50) NOT NULL REFERENCES "Seller"(seller_id),
    shipping_limit_date TIMESTAMP NOT NULL,
    price NUMERIC NOT NULL,
    freight_value NUMERIC NOT NULL,
    PRIMARY KEY (order_id, order_item_id)
);


