import pandas as pd
import numpy as np
import missingno as msno
import matplotlib
matplotlib.use('Agg') # Forza il backend non interattivo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from datetime import datetime, timedelta
from scipy import stats
import os
import pkgutil

if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = lambda x: None
from ydata_profiling import ProfileReport


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.6f}')
print('✅ Libraries loaded successfully')
print(f' pandas : {pd.__version__}')
print(f' numpy : {np.__version__}')
np.random.seed(42)



class DQAReport:
    # Structured, repeatable Data Quality Assessment Report.
    # Evaluates all ISO 25012 dimensions, produces a scoreboard and returns a summary report.
    def __init__(self, df: pd.DataFrame, table_name: str, primary_key: str = None):
        self.df = df.copy()
        self.table_name = table_name
        self.pk = primary_key
        self.results = {}
        # dimension -> {'score': float, 'issues': int, 'details': str}
        # Initialize a dictionary to store the final results for each quality dimension.
        self.flags = {}
        self.validity_details = {}
    
    # dimension -> boolean Series (True = issue)
    # ── DIMENSION 1: Completeness ──────────────────────────────────────────
    def check_completeness(self, required_cols: list = None):
        cols = required_cols or self.df.columns.tolist()
        missing_counts = self.df[cols].isnull().sum()
        total_values = len(self.df) * len(cols)
        total_missing = missing_counts.sum()
        score = 1 - (total_missing / total_values)
        flag_rows = self.df[cols].isnull().any(axis=1)
        self.flags['completeness'] = flag_rows
        self.results['completeness'] = {
            'score' :  round(score, 6),
            'issues' : int(total_missing),
            'details' : f"Missing per column: {missing_counts[missing_counts > 0].to_dict()}"
        }
        return self
        
    # ── DIMENSION 2: Uniqueness ────────────────────────────────────────────
    def check_uniqueness(self, key_cols: list = None):
        cols = key_cols or ([self.pk] if self.pk else self.df.columns.tolist())
        dup = self.df.duplicated(subset=cols, keep=False)
        score = 1 - (dup.sum() / len(self.df))
        self.flags['uniqueness'] = dup
        self.results['uniqueness'] = {
            'score' :  round(score, 6),
            'issues' : int(dup.sum()),
            'details' : f"Duplicate rows on {cols}:{int(dup.sum())}"
        }
        return self
    
    # ── DIMENSION 3: Validity ──────────────────────────────────────────────
    def check_validity(self, rules: dict):
        all_invalid = pd.Series(False, index=self.df.index)
        rule_details = []
        self.validity_details = {}
        for col, rule_fn in rules.items():
            if col not in self.df.columns:
                continue
            valid_mask = rule_fn(self.df[col])
            invalid_mask = ~valid_mask & self.df[col].notna()
            self.validity_details[col] = int(invalid_mask.sum())
            all_invalid |= invalid_mask
            rule_details.append(f"{col}: {int(invalid_mask.sum())} invalid")
        score = 1 - (all_invalid.sum() / len(self.df))
        self.flags['validity'] = all_invalid
        self.results['validity'] = {
            'score' :  round(score, 6),
            'issues' : int(all_invalid.sum()),
            'details' : " | ".join(rule_details)
        }

        return self
    
    # ── DIMENSION 4: Consistency ───────────────────────────────────────────
    def check_consistency(self, rules : list):
        all_inconsistent = pd.Series(False, index = self.df.index)
        for rule_fn in rules:
            inconsistent = ~rule_fn(self.df)
            all_inconsistent |= inconsistent

        score = 1 - (all_inconsistent.sum() /len(self.df))
        self.flags['consistency'] = all_inconsistent
        self.results['consistency'] = {
            'score' :  round(score, 6),
            'issues' : int(all_inconsistent.sum()),
            'details' : f"{int(all_inconsistent.sum())} rows violate at least one consistency rule"
        }
        return self
    
    # ── DIMENSION 5: Timeliness ────────────────────────────────────────────
    def check_timeliness(self, date_col: str, min_year: int = 2016, max_year: int = 2018, allow_future: bool = False):
        if date_col not in self.df.columns:
            return self
        
        col = pd.to_datetime(self.df[date_col], errors='coerce')
        # Identificazione dei valori "Stale" (precedenti al 2016)
        stale = col.dt.year < min_year
        # Identificazione dei valori "Future" (successivi al 2018)
        # Se allow_future è False, tutti i record > 2018 sono considerati problemi di tempestività 
        future = col.dt.year > max_year if not allow_future else pd.Series(False, index=col.index)
       
        flag = (stale | future) & col.notna()
        score = 1 - (flag.sum() / len(self.df))
        self.flags['timeliness'] = flag
        
        self.results['timeliness'] = {
            'score' : round(score, 6),
            'issues' : int(flag.sum()),
            'details' : f"Future Dates > {max_year}: {int(future.sum())} | Stale < {min_year}: {int(stale.sum())}"
        }

        return self
    
    # ── SCORECARD ─────────────────────────────────────────────────────────
    def scorecard(self) -> pd.DataFrame:
        rows = []
        for dim, res in self.results.items():
            emoji = '🟢' if res['score'] >= 0.95 else ('🟡' if res['score'] >= 0.80 else '🔴')
            rows.append({
                'Table' : self.table_name,
                'Dimension' : dim.capitalize(),
                'Score' : res['score'],
                'Issues' : res['issues'],
                'Status' : emoji,
                'Details' : res['details']
            })
        return pd.DataFrame(rows)
    
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return round(np.mean([v['score'] for v in self.results.values()]), 4)

print('✅ DQAReport class defined')




# — Visual: Missing Values Pattern

def plot_dqa_report(df, dqa, table_name, output_path=None):
    """
    Generate a complete visual report for a table:
    1. Missingness matrix
    2. DQ Dimension Scorecard
    """ 
    fig, axes = plt.subplots(1, 2, figsize=(14, 4)) 
    
    # 1. Missing Values ​​Matrix
    msno.matrix(df, ax=axes[0], color=(0.25, 0.45, 0.75), fontsize=10)
    axes[0].set_title(f'Missing Values Matrix — {table_name}', fontsize=12, fontweight='bold')
    
    # 2. DQ Bar Chart
    sc = dqa.scorecard()
    colors = ['#2ecc71' if s >= 0.95 else ('#f39c12' if s >= 0.80 else '#e74c3c') for s in sc['Score']] 
    
    bars = axes[1].barh(sc['Dimension'], sc['Score'], color=colors, edgecolor='white', height=0.5) 
    
    axes[1].set_xlim(0, 1.05) 
    axes[1].axvline(0.95, color='green', linestyle='--', alpha=0.5, label='Target ≥ 0.95')
    axes[1].axvline(0.80, color='orange', linestyle='--', alpha=0.5, label='Warning ≥ 0.80')

    axes[1].set_title(f'DQ Dimension Scores — {table_name}', fontsize=12, fontweight='bold') 

    for i, (score, dim) in enumerate(zip(sc['Score'], sc['Dimension'])): 
        axes[1].text(score +0.01, i,f'{score:.2%}', va='center', fontsize=9)

    axes[1].set_xlabel('Score (0-1)') 
    axes[1].legend(fontsize=8) 
    plt.tight_layout() 
    
    if output_path:
        file_name = f"dq_report_{table_name.replace(' ', '_')}.png"
        save_path = os.path.join(output_path, file_name)
        fig.savefig(save_path, dpi=300) 
        print(f"✅ Grafico salvato: {save_path}")
        
    return fig


def missing_bar_and_validity_plot(dqa, output_path=None):
    df = dqa.df
    table_name = dqa.table_name
    # Identify the columns that have at least one validity error
    cols_with_issues = [col for col, count in dqa.validity_details.items() if count > 0]
    # Let's determine the number of subplots needed: 1 for missing + N for valid ones
    num_plots = 1 + len(cols_with_issues)
    n_cols_final = max(2, num_plots)
    fig, axes = plt.subplots(1, n_cols_final, figsize=(6 * n_cols_final, 5))

    # Missing values bar: it shows the  missing data percentages
    miss = df.isnull().sum()
    miss_pct = (miss/len(df) *100).sort_values(ascending=True)
    miss_plot_data = miss_pct[miss_pct > 0]
    if not miss_plot_data.empty:
        miss_plot_data.plot(kind='barh', ax=axes[0], color='#e74c3c', alpha=0.8)
        axes[0].set_title(f'Missing Values % — {table_name}', fontweight='bold')
        axes[0].set_xlabel('% missing') 
    else:
        axes[0].axis('off') 
        axes[0].text(0.5, 0.5, f'No missing values in the {table_name}', ha='center', va='center')
        

    # GRAFICO VALIDITY ISSUES
    if len(cols_with_issues) > 0:
        for i, col in enumerate(cols_with_issues):
            ax = axes[i + 1]
            invalid_count = dqa.validity_details[col]
            valid_count = len(dqa.df) - invalid_count
            
            
            pd.Series({
                'Valid': valid_count,
                'Invalid': invalid_count
            }).plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], rot=0)
            
            ax.set_title(f'{col} Validity', fontweight='bold')
            ax.set_ylabel('Count')
            # Let's add numeric labels above the bars
            for p in ax.patches:
                ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, f'No invalid values in the {table_name}', ha='center', va='center')
    
    plt.tight_layout()
    
    if output_path:
        save_path = os.path.join(output_path, f"dq_plot_{table_name.replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Chart saved: {save_path}")
    
    return fig
    


def run_dqa_pipeline(df, table_name, completeness_cols,
    uniqueness_cols, validity_rules, consistency_rules, output_folder, pk=None, date_col=None, min_year=2016, max_year=2018):
    # Define a reusable function that executes the full Data Quality Assessment pipeline.
    report = DQAReport(df, table_name=table_name, primary_key=pk)
    report.check_completeness(completeness_cols)
    report.check_uniqueness(uniqueness_cols)
    report.check_validity(validity_rules)
    report.check_consistency(consistency_rules)
    if date_col:
        # Check if a date column is provided for timeliness evaluation.
        report.check_timeliness(date_col, min_year=min_year, max_year=max_year, allow_future=False)
    if table_name == 'OrderItem table':
        if 'order_delivered_carrier_date' in df.columns:
            df = df.drop(columns=['order_delivered_carrier_date'])
        if 'order_delivered_carrier_date' in report.df.columns:
            report.df = report.df.drop(columns=['order_delivered_carrier_date'])
    plot_dqa_report(df, report, table_name, output_folder)
    missing_bar_and_validity_plot(report, output_folder)

    return report


def print_validity_exceptions(report_list):
    
    print("\n" + "="*50)
    print("DETAIL OF VALIDITY EXCEPTIONS ")
    print("="*50)

    for report in report_list:
        table_name = report.table_name
        
        invalid_mask = report.flags.get('validity', pd.Series(False, index=report.df.index))
        
        if invalid_mask.any():
            print(f"\nTabella: {table_name}")
            print("-" * 30)
            
            
            df_invalid = report.df[invalid_mask]
            
            
            cols_to_check = [col for col, count in report.validity_details.items() if count > 0]
            
            if not cols_to_check:
                print("No specific value identified (possible global consistency error).")
                continue

            for idx, row in df_invalid.iterrows():
                issues = []
                for col in cols_to_check:
                    
                    val = row[col]
                    issues.append(f"{col}: '{val}'")
                
                
        else:
            print(f"\nTable: {table_name} ->  All values ​​are valid.")

def report_consistency_violations(order_df, order_item_df):
    print("\n" + "="*50)
    print("REPORT OF VIOLATIONS OF CONSISTENCY")
    print("="*50)

    # --- ANALISI ORDER TABLE ---
    print("\n>>> TABLE ANALYSIS: Order")
    
    # Duplicati PK
    dups_order = order_df.duplicated(subset=['order_id'], keep=False)
    if dups_order.any():
        print(f" PK Violated: {dups_order.sum()} rows with duplicate order_id")

    # Inconsistenze Temporali
    # Definiamo le maschere di errore basandoci sulle tue regole
    err_approved = (order_df['order_approved_at'].notna()) & (order_df['order_approved_at'] < order_df['order_purchase_timestamp'])
    err_carrier = (order_df['order_delivered_carrier_date'].notna()) & (order_df['order_delivered_carrier_date'] < order_df['order_purchase_timestamp'])
    err_customer = (order_df['order_delivered_customer_date'].notna()) & (order_df['order_delivered_customer_date'] < order_df['order_purchase_timestamp'])
    
    if err_approved.any():
        print(f" Chronology: {err_approved.sum()} orders approved BEFORE purchase")
    if err_carrier.any():
        print(f" Chronology: {err_carrier.sum()} orders delivered to the courier BEFORE purchase")
    if err_customer.any():
        print(f" Chronology: {err_customer.sum()} orders delivered to the customer BEFORE purchase")

    # --- ANALISI ORDERITEM TABLE ---
    print("\n>>> TABLE ANALYSIS: OrderItem")

    # Duplicati PK composta
    dups_item = order_item_df.duplicated(subset=['order_id', 'order_item_id'], keep=False)
    if dups_item.any():
        print(f"❌ PK Violated: {dups_item.sum()} rows with duplicate (order_id, order_item_id) pair")
    else:
        print("no consistency rules were violated")


    print("\n" + "="*50)

def report_promises_violations(order_df, order_item_df):
    # Shipping Limit vs Carrier Date
    err_shipping = (order_item_df['shipping_limit_date'].notna()) & \
                   (order_item_df['order_delivered_carrier_date'].notna()) & \
                   (order_item_df['shipping_limit_date'] < order_item_df['order_delivered_carrier_date'])
    
    if err_shipping.any():
        print(f"❌ Logic: {err_shipping.sum()} articoli consegnati al corriere OLTRE il limite di spedizione")




def generate_comparison_from_csv(folder_path, reports_map):
    data_list = []
    
    for table_name, file_name in reports_map.items():
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            df_temp = pd.read_csv(file_path)
            
            if all(col in df_temp.columns for col in ['Dimension', 'Score']):
                df_temp['Table'] = table_name
                data_list.append(df_temp[['Table', 'Dimension', 'Score']])
                print(f"✅ {file_name} loaded successfully.")
            else:
                print(f"⚠️ {file_name} does not contain the columns 'Dimension' or 'Score'.")
        else:
            print(f"❌ Error: The file {file_name} does not exist in the specified path.")

    if not data_list:
        print("🛑 No data found. Unable to generate comparison..")
        return

    combined_df = pd.concat(data_list, ignore_index=True)
    
    pivot = combined_df.pivot(index='Dimension', columns='Table', values='Score')
    
    pivot = pivot.round(6)
    
    print('\n📊 DQ SCORE COMPARISON \n')
    print(pivot.to_string())

    output_path = os.path.join(folder_path, 'dq_comparison_pivot.csv')
    pivot.to_csv(output_path)
    print(f"\n💾 Comparison saved in: {output_path}")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind='bar', ax=ax, width=0.5, rot=20)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.95, color='green', linestyle='--', linewidth=1, alpha=0.6, label='Target 0.95')
    ax.set_ylabel('Score')
    ax.set_title('DQ Dimension Scores — Multi-Table Comparison', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_folder +'dq_comparison_plot.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    
    output_folder = r'C:\Users\acer\Desktop\Magistrale 1-02-2026\I anno\Data Ware House\Olist\InitialDQ\\'
    # Define the folder where the CSVs are located
    path = r'C:\Users\acer\Desktop\Magistrale 1-02-2026\I anno\Data Ware House\Olist\csv'

    order_item_df = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
    order_df = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
    product_df = pd.read_csv(os.path.join(path, "olist_products_dataset.csv"))
    product_df = product_df.drop(columns=['product_name_lenght', 'product_description_lenght', 
    'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'])
    customer_df = pd.read_csv(os.path.join(path, "olist_customers_dataset.csv"))
    customer_df = customer_df.drop(columns=['customer_unique_id','customer_zip_code_prefix'])
    seller_df = pd.read_csv(os.path.join(path, "olist_sellers_dataset.csv")) 
    print("Available columns in Seller:", seller_df.columns.tolist())
    seller_df = seller_df.drop(columns=['seller_zip_code_prefix'])
    category_df = pd.read_csv(os.path.join(path, "product_category_name_translation.csv"))
    

    
    # We want to enable cross-table quality checks, 
    # specifically between the Order table and the OrderItem table.
    # We populate order_item_df with the necessary data from order_df
    # Let's use `order_id` as the common field.
    # Since the DQAReport class is designed to analyze only one DataFrame at a time, 
    # this operation “consolidates” the necessary information in a single location.
    order_item_enriched_df = order_item_df.merge(
        order_df[['order_id', 'order_delivered_carrier_date']], 
        on='order_id', 
        how='left'
    )
    
   
    print(f'order_item_df: {order_item_df.shape[0]:>4} rows × {order_item_df.shape[1]} cols')
    print(order_item_df.head())
    print(f'order_df: {order_df.shape[0]:>4} rows × {order_df.shape[1]} cols')
    print(order_df.head())
    print(f'product_df: {product_df.shape[0]:>4} rows × {product_df.shape[1]} cols')
    print(product_df.head())
    print(f'customer_df: {customer_df.shape[0]:>4} rows × {customer_df.shape[1]} cols')
    print(customer_df.head())
    print(f'seller_df: {seller_df.shape[0]:>4} rows × {seller_df.shape[1]} cols')
    print(seller_df.head())
    print(f'category_df: {category_df.shape[0]:>4} rows × {category_df.shape[1]} cols')
    print(category_df.head())
    
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Validity rules for the Order table 
    import re
    customers_list = customer_df['customer_id'].unique().tolist()
    order_validity_rules = {
        'customer_id' : lambda s: s.isin(customers_list) & s.notna() & (s != ""),
        'order_purchase_timestamp' : lambda s: pd.to_datetime(s, errors='coerce').notna(),
        'order_approved_at' : lambda s: pd.to_datetime(s, errors='coerce').notna(),
        'order_delivered_carrier_date' : lambda s: pd.to_datetime(s, errors='coerce').notna(),
        'order_delivered_customer_date' : lambda s: pd.to_datetime(s, errors='coerce').notna(),
        'order_estimated_delivery_date' : lambda s: pd.to_datetime(s, errors='coerce').notna(),
    }
    # Validity rules for the OrderItem table 
    order_id_list = order_df['order_id'].unique().tolist()
    products_list = product_df['product_id'].unique().tolist()
    sellers_list = seller_df['seller_id'].unique().tolist()
    order_item_validity_rules = {
        'order_id' : lambda s: s.isin(order_id_list) & s.notna() & (s != ""),
        'product_id' : lambda s: s.isin(products_list) & s.notna() & (s != ""),
        'seller_id' : lambda s: s.isin(sellers_list) & s.notna() & (s != ""),
        'shipping_limit_date' : lambda s: pd.to_datetime(s, errors='coerce').notna(),
        'price' : lambda s: s > 0,
        'freight_value' : lambda s: s >= 0,
    }
    # Validity rules for the Customer table 
    customer_validity_rules = {
        'customer_city': lambda s: s.str.match(r'^[A-Za-zÀ-ÿã\s\'-]+$') == True,
        'customer_state' : lambda s: s.isin(['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'ES', 'GO', 'PE', 'CE', 'PA', 'MT', 'MA', 'MS', 'PB', 'RN', 'PI', 'AL', 'SE', 'TO', 'RO', 'AM', 'AC', 'AP', 'RR']),
    }

    # Validity rules for the Seller table 
    seller_validity_rules = {
        'seller_city': lambda s: s.str.match(r'^[A-Za-zÀ-ÿ\s\'-]+$') == True,
        'seller_state' : lambda s: s.str.upper().isin(['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'ES', 'GO', 'PE', 'CE', 'PA', 'MT', 'MA', 'MS', 'PB', 'RN', 'PI', 'AL', 'SE', 'TO', 'RO', 'AM', 'AC', 'AP', 'RR']),
    }

    # Validity rules for the Product table 
    categories_list = [c.upper() for c in category_df['product_category_name'].unique()]
    product_validity_rules = {
        'product_category_name': lambda s: s.str.upper().isin(categories_list) & ~s.str.contains('_', na=False) & 
                                           (s.str.match(r'^[A-Za-zÀ-ÿãç\s\'-]+$') == True),
    }

    # Validity rules for the ProductCategoryName table 
    product_category_name_validity_rules = {
        'product_category_name':lambda s:~s.str.contains('_', na=False) & 
                                           (s.str.match(r'^[A-Za-zÀ-ÿãç\s\'-]+$') == True),
        'product_category_name_english':lambda s:~s.str.contains('_', na=False) & 
                                           (s.str.match(r'^[A-Za-zÀ-ÿãç\s\'-]+$') == True),}

    # Consistency rules 
    # N.B.: All consistency checks regarding the foreign keys of the tables have also been performed
    order_consistency_rules = [
        lambda df: ~df.duplicated(subset=['order_id'], keep=False),
        lambda df: (df['order_approved_at'].isna()) | \
                (df['order_purchase_timestamp'].isna()) | \
                (df['order_approved_at'] >= df['order_purchase_timestamp']),
        lambda df: (df['order_delivered_carrier_date'].isna()) | \
                (df['order_purchase_timestamp'].isna()) | \
                (df['order_delivered_carrier_date'] > df['order_purchase_timestamp']),
        lambda df: (df['order_delivered_customer_date'].isna()) | \
                (df['order_purchase_timestamp'].isna()) | \
                (df['order_delivered_customer_date'] > df['order_purchase_timestamp']),
        lambda df: (df['order_estimated_delivery_date'].isna()) | \
                (df['order_purchase_timestamp'].isna()) | \
                (df['order_estimated_delivery_date'] > df['order_purchase_timestamp']),
        lambda df: (df['order_approved_at'].isna()) | \
                (df['order_delivered_carrier_date'].isna()) | \
                (df['order_approved_at'] < df['order_delivered_carrier_date']),
        lambda df: (df['order_delivered_carrier_date'].isna()) | \
                (df['order_delivered_customer_date'].isna()) | \
                (df['order_delivered_carrier_date'] < df['order_delivered_customer_date']),
        lambda df: (df['order_delivered_carrier_date'].isna()) | \
                (df['order_estimated_delivery_date'].isna()) | \
                (df['order_delivered_carrier_date'] < df['order_estimated_delivery_date']),
    ]

    order_item_consistency_rules = [
        lambda df: ~df.duplicated(subset=['order_id', 'order_item_id'], keep=False),
    ]


    customer_consistency_rules = [
        lambda df: ~df.duplicated(subset=['customer_id'], keep=False),
        lambda df: df['customer_state'].fillna('') == df['customer_state'].fillna('').str.upper(),
        lambda df: df['customer_city'].fillna('') == df['customer_city'].fillna('').str.strip(),
        lambda df: df['customer_state'].fillna('') == df['customer_state'].fillna('').str.strip(),
    ]

    seller_consistency_rules = [
        lambda df: ~df.duplicated(subset=['seller_id'], keep=False),
        lambda df: df['seller_state'].fillna('') == df['seller_state'].fillna('').str.upper(),
        lambda df: df['seller_city'].fillna('') == df['seller_city'].fillna('').str.strip(),
        lambda df: df['seller_state'].fillna('') == df['seller_state'].fillna('').str.strip(),
    ]

    product_consistency_rules = [
        lambda df: ~df.duplicated(subset=['product_id'], keep=False),
        lambda df: df['product_category_name'].fillna('') == df['product_category_name'].fillna('').str.lower(),
        lambda df: df['product_category_name'].fillna('') == df['product_category_name'].fillna('').str.strip(),
    ]

    product_category_name_consistency_rules = [
        lambda df: ~df.duplicated(subset=['product_category_name'], keep=False),
        lambda df: df['product_category_name'].fillna('') == df['product_category_name'].fillna('').str.lower(),
        lambda df: df['product_category_name'].fillna('') == df['product_category_name'].fillna('').str.strip(),
        lambda df: df['product_category_name_english'].fillna('') == df['product_category_name_english'].fillna('').str.lower(),
        lambda df: df['product_category_name_english'].fillna('') == df['product_category_name_english'].fillna('').str.strip(),
    ]


    report_order = run_dqa_pipeline( 
        df = order_df, table_name='Order table',
        completeness_cols=['order_id','customer_id','order_status', 'order_purchase_timestamp','order_approved_at','order_delivered_carrier_date', 
        'order_delivered_customer_date', 'order_estimated_delivery_date'],
        uniqueness_cols=['order_id'],
        validity_rules=order_validity_rules,
        consistency_rules=order_consistency_rules,
        output_folder=output_folder,
        pk='order_id',
        date_col='order_purchase_timestamp',
        min_year=2016, max_year=2018)


    report_order_item = run_dqa_pipeline( 
        df=order_item_df, table_name='OrderItem table', 
        completeness_cols=['order_id','order_item_id','product_id','seller_id','price', 'freight_value'],
        uniqueness_cols=['order_id', 'order_item_id'],
        validity_rules=order_item_validity_rules,
        consistency_rules=order_item_consistency_rules,
        output_folder=output_folder
    )


    report_product = run_dqa_pipeline(
        df=product_df, table_name='Product table',
        completeness_cols=['product_id','product_category_name'],
        uniqueness_cols=['product_id'],
        validity_rules=product_validity_rules,
        consistency_rules=product_consistency_rules,
        output_folder=output_folder,
        pk='product_id'
    )



    report_customer = run_dqa_pipeline(
        df=customer_df, table_name='Customer table',
        completeness_cols=['customer_id', 'customer_city', 'customer_state'],
        uniqueness_cols=['customer_id'],
        validity_rules=customer_validity_rules,
        consistency_rules=customer_consistency_rules,
        output_folder=output_folder,
        pk='customer_id'
    )


    report_seller = run_dqa_pipeline(
        df=seller_df, table_name='Seller table',
        completeness_cols=['seller_id', 'seller_city', 'seller_state'],
        uniqueness_cols=['seller_id'],
        validity_rules=seller_validity_rules,
        consistency_rules=seller_consistency_rules,
        output_folder=output_folder,
        pk='seller_id'
    )


    report_category = run_dqa_pipeline(
        df=category_df, table_name='ProductCategoryName table',
        completeness_cols=['product_category_name', 'product_category_name_english'],
        uniqueness_cols=['product_category_name', 'product_category_name_english'],
        validity_rules=product_category_name_validity_rules,
        consistency_rules=product_category_name_consistency_rules,
        output_folder=output_folder,
        pk='product_category_name'
    )


    print('\n📦 Order — DQ Report\n')
    scorecard_order = report_order.scorecard()
    scorecard_order.to_csv(output_folder + 'dq_scorecard_order.csv', index=False)
    print(report_order.scorecard()[['Dimension','Score','Issues','Status','Details']].to_string(index=False))
    print(f'\nOverall DQ Score (Order table):{report_order.overall_score():.2%}')


    print('\n📦 OrderItem — DQ Report\n')
    scorecard_order_item = report_order_item.scorecard()
    scorecard_order_item.to_csv(output_folder + 'dq_scorecard_order_item.csv', index=False)
    print(report_order_item.scorecard()[['Dimension','Score','Issues','Status','Details']].to_string(index=False))
    print(f'\nOverall DQ Score (OrderItem table):{report_order_item.overall_score():.2%}')


    print('\n📦 Product — DQ Report\n')
    scorecard_product = report_product.scorecard()
    scorecard_product.to_csv(output_folder + 'dq_scorecard_product.csv', index=False)
    print(report_product.scorecard()[['Dimension','Score','Issues','Status','Details']].to_string(index=False))
    print(f'\nOverall DQ Score (Product table):{report_product.overall_score():.2%}')


    print('\n📦 Customer — DQ Report\n')
    scorecard_customer = report_customer.scorecard()
    scorecard_customer.to_csv(output_folder + 'dq_scorecard_customer.csv', index=False)
    print(report_customer.scorecard()[['Dimension','Score','Issues','Status','Details']].to_string(index=False))
    print(f'\nOverall DQ Score (Customer table):{report_customer.overall_score():.2%}')


    print('\n📦 Seller — DQ Report\n')
    scorecard_seller = report_seller.scorecard()
    scorecard_seller.to_csv(output_folder + 'dq_scorecard_seller.csv', index=False)
    print(report_seller.scorecard()[['Dimension','Score','Issues','Status','Details']].to_string(index=False))
    print(f'\nOverall DQ Score (Seller table):{report_seller.overall_score():.2%}')


    print('\n📦 ProductCategoryName — DQ Report\n')
    scorecard_category = report_category.scorecard()
    scorecard_category.to_csv(output_folder + 'dq_scorecard_category.csv', index=False)
    print(report_category.scorecard()[['Dimension','Score','Issues','Status','Details']].to_string(index=False))
    print(f'\nOverall DQ Score (ProductCategoryName table):{report_category.overall_score():.2%}')


    reports = [report_order, report_order_item, report_product, report_customer, report_seller, report_category]

    print_validity_exceptions(reports)


    report_consistency_violations(order_df, order_item_df)
    report_promises_violations(order_df, order_item_enriched_df)
    
    

    CSV_REPORTS_MAP = {
        'Order': 'dq_scorecard_order.csv',
        'OrderItem': 'dq_scorecard_order_item.csv',
        'Product': 'dq_scorecard_product.csv',
        'Customer': 'dq_scorecard_customer.csv',
        'Seller': 'dq_scorecard_seller.csv',
        'ProductCategoryName': 'dq_scorecard_category.csv'
    }
    generate_comparison_from_csv(output_folder, CSV_REPORTS_MAP)
   
    

    