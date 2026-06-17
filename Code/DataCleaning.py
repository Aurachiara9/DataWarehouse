import pandas as pd
import numpy as np
import re
import hashlib
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from copy import deepcopy
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import recordlinkage
from difflib import get_close_matches
import os
import unicodedata
from DataCleaning1 import OutlierDetector


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 50)
np.random.seed(42)
print('✅ Libraries loaded')

path = r'C:\Users\acer\Desktop\Magistrale 1-02-2026\I anno\Data Ware House\Olist\csv'

order_item_df = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
order_df = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
product_df = pd.read_csv(os.path.join(path, "olist_products_dataset.csv"))
product_df = product_df.drop(columns=['product_name_lenght', 'product_description_lenght', 
'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'])

customer_df = pd.read_csv(os.path.join(path, "olist_customers_dataset.csv"))
customer_df = customer_df.drop(columns=['customer_unique_id','customer_zip_code_prefix'])
seller_df = pd.read_csv(os.path.join(path, "olist_sellers_dataset.csv")) 
print("Colonne disponibili in Seller:", seller_df.columns.tolist())
seller_df = seller_df.drop(columns=['seller_zip_code_prefix'])
category_df = pd.read_csv(os.path.join(path, "product_category_name_translation.csv"))

BASE_PATH = r'C:\Users\acer\Desktop\Magistrale 1-02-2026\I anno\Data Ware House\Olist\Cleaning\Cleaning2'
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
    print(f"Directory creata: {BASE_PATH}")


class AuditLog:
    """
    Records every cleaning transformation applied to the data.
    Each entry captures: step name, column, row index, before, after, timestamp.
    """
    def __init__(self):
        self._entries = []
    
    def log(self, step: str, col: str, idx, before, after, reason: str = ''):
        # Log a single transformation event

        self._entries.append({
            'step' : step,
            # Name of the cleaning step

            'column' : col,
            # Column affected by the transformation

            'row_index' : idx,
            # Row index where the change occurred

            'before' : before,
            # Original value before cleaning

            'after' : after,
            # New value after cleaning

            'reason' : reason,
            # Optional explanation for why the change was made

            'timestamp' : datetime.now().isoformat()
            # Timestamp of the transformation for traceability
        })

    def log_batch(self, step: str, col: str, mask: pd.Series,
                  before_series: pd.Series, after_series: pd.Series, reason: str = ''):
        # Log multiple changes at once (vectorized step).
        # Log multiple row-level changes produced by a vectorized operation
        changed_idx = mask[mask].index
        # Extract indexes of rows that were modified

        for idx in changed_idx:
            self.log(step, col, idx,
                     str(before_series.get(idx, 'N/A')),
                     str(after_series.get(idx, 'N/A')),
                     reason)
    
    def to_df(self) -> pd.DataFrame:
        # Convert the audit log into a pandas DataFrame
        return pd.DataFrame(self._entries)

    def summary(self) -> pd.DataFrame:
        # Return an aggregated summary of the audit log by cleaning step
        if not self._entries:
            return pd.DataFrame()
        # If no entries exist, return an empty DataFrame

        df = self.to_df()
        # Convert raw entries to DataFrame
        return (df.groupby('step')
                  .agg(changes=('row_index','count'),
                       cols_affected=('column', lambda x: ', '.join(x.unique())))
                  .reset_index()
                  .sort_values('changes', ascending=False))
        # Group entries by step, count changes, list affected columns, and sort by frequency
        
    def __len__(self):
        return len(self._entries)

#----------------------------------------------------------------------------------------------------------
#Handling invalid values
#----------------------------------------------------------------------------------------------------------
def fix_and_clean_product_categories(product_df, category_df, order_item_df, order_df, audit_log_object):
    """
    Esegue la pulizia delle categorie e l'eliminazione a cascata su 3 livelli:
    1. Product -> 2. OrderItems -> 3. Orders (se rimasti senza articoli)
    """
    valid_categories = category_df['product_category_name'].unique().tolist()
    
    
    invalid_mask = (~product_df['product_category_name'].isin(valid_categories)) & \
                   (product_df['product_category_name'].notna())
    
    invalid_rows = product_df[invalid_mask]
    totally_different_indices = {}

    # Fase 1: Fuzzy Matching
    for idx, row in invalid_rows.iterrows():
        original_val = row['product_category_name']
        matches = get_close_matches(original_val, valid_categories, n=1, cutoff=0.8)
        
        if matches:
            suggested_val = matches[0]
            product_df.at[idx, 'product_category_name'] = suggested_val
            audit_log_object.log('clean_categories', 'product_category_name', idx, 
                                 original_val, suggested_val, 'Fuzzy match correction')
        else:
            if original_val not in totally_different_indices:
                totally_different_indices[original_val] = []
            totally_different_indices[original_val].append(idx)

    # Fase 2: Gestione Valori Rari ed Eliminazione a Cascata
    indices_to_drop_product = []
    product_ids_to_remove = []
    
    for val, indices in totally_different_indices.items():
        if len(indices) > 1:
            englishName = ""
            if(val=="portateis_cozinha_e_preparadores_de_alimentos"):
                englishName = "portable kitchen and food preparators"
            else:
                englishName = val
            new_row = pd.DataFrame({'product_category_name': [val], 'product_category_name_english': [englishName]})
            category_df = pd.concat([category_df, new_row], ignore_index=True)
            audit_log_object.log('add_category', 'product_category_name', 'GLOBAL', 
                                 'raw', val, f'Promoted category: {len(indices)} occurrences')
        else:
            idx_prod = indices[0]
            pid = product_df.at[idx_prod, 'product_id']
            indices_to_drop_product.append(idx_prod)
            product_ids_to_remove.append(pid)
            
            audit_log_object.log('remove_invalid_product', 'product_id', idx_prod, 
                                 pid, 'REMOVED', f'Rare invalid category: {val}')

    
    # Rimozione dalla tabella Product
    product_df = product_df.drop(index=indices_to_drop_product).reset_index(drop=True)
    
    # Rimozione dalla tabella OrderItems
    if not order_item_df is None and len(product_ids_to_remove) > 0:
        # Identifichiamo gli order_id potenzialmente impattati prima di cancellare
        orders_to_check = order_item_df[order_item_df['product_id'].isin(product_ids_to_remove)]['order_id'].unique()
        
        initial_items = len(order_item_df)
        order_item_df = order_item_df[~order_item_df['product_id'].isin(product_ids_to_remove)].reset_index(drop=True)
        
        removed_items = initial_items - len(order_item_df)
        if removed_items > 0:
            audit_log_object.log('cascade_delete_items', 'product_id', 'ALL', 
                                 'existing_refs', 'REMOVED', f'Removed {removed_items} items')

        # Rimozione dalla tabella Order (Ordini rimasti orfani di articoli)
        if not order_df is None and len(orders_to_check) > 0:
            # Un ordine va rimosso se il suo order_id non esiste più nella tabella order_items
            valid_order_ids = order_item_df['order_id'].unique()
            
            initial_orders = len(order_df)
            # Filtriamo: teniamo solo gli ordini che hanno ancora almeno un articolo corrispondente
            order_df = order_df[order_df['order_id'].isin(valid_order_ids)].reset_index(drop=True)
            
            removed_orders = initial_orders - len(order_df)
            if removed_orders > 0:
                audit_log_object.log('cascade_delete_orders', 'order_id', 'ALL', 
                                     'existing_orders', 'REMOVED', 
                                     f'Removed {removed_orders} orders that became empty')

    # Rimozione numeri in product_df['product_category_name']
    if 'product_category_name' in product_df.columns:
        mask_digits_prod = product_df['product_category_name'].str.contains(r'\d', na=False)
        for idx, row in product_df[mask_digits_prod].iterrows():
            original_val = row['product_category_name']
            cleaned_val = re.sub(r'\d+', '', original_val)
            product_df.at[idx, 'product_category_name'] = cleaned_val
            audit_log_object.log('remove_digits', 'product_category_name', idx, 
                                 original_val, cleaned_val, 'Removed numbers from category name')

    # Rimozione numeri in category_df['product_category_name']
    if 'product_category_name' in category_df.columns:
        mask_digits_cat = category_df['product_category_name'].str.contains(r'\d', na=False)
        for idx, row in category_df[mask_digits_cat].iterrows():
            original_val = row['product_category_name']
            cleaned_val = re.sub(r'\d+', '', original_val)
            category_df.at[idx, 'product_category_name'] = cleaned_val
            audit_log_object.log('remove_digits', 'product_category_name', idx, 
                                 original_val, cleaned_val, 'Removed numbers from translation source')

    # Rimozione numeri in category_df['product_category_name_english']
    if 'product_category_name_english' in category_df.columns:
        mask_digits_eng = category_df['product_category_name_english'].str.contains(r'\d', na=False)
        for idx, row in category_df[mask_digits_eng].iterrows():
            original_val = row['product_category_name_english']
            cleaned_val = re.sub(r'\d+', '', original_val)
            category_df.at[idx, 'product_category_name_english'] = cleaned_val
            audit_log_object.log('remove_digits', 'product_category_name_english', idx, 
                                 original_val, cleaned_val, 'Removed numbers from english translation source')
    

    # Sostituzione dei trattini bassi (_) con spazi 
    if 'product_category_name' in product_df.columns:
        mask_prod = product_df['product_category_name'].str.contains('_', na=False)
        for idx, row in product_df[mask_prod].iterrows():
            original_val = row['product_category_name']
            cleaned_val = original_val.replace('_', ' ')
            product_df.at[idx, 'product_category_name'] = cleaned_val
            audit_log_object.log('clean_categories', 'product_category_name', idx, 
                                 original_val, cleaned_val, 'Replacing underscore with space')

    if 'product_category_name' in category_df.columns:
        mask_cat = category_df['product_category_name'].str.contains('_', na=False)
        for idx, row in category_df[mask_cat].iterrows():
            original_val = row['product_category_name']
            cleaned_val = original_val.replace('_', ' ')
            category_df.at[idx, 'product_category_name'] = cleaned_val
            audit_log_object.log('clean_categories', 'product_category_name', idx, 
                                 original_val, cleaned_val, 'Replacing underscore with space')

    if 'product_category_name_english' in category_df.columns:
        mask_cat = category_df['product_category_name_english'].str.contains('_', na=False)
        for idx, row in category_df[mask_cat].iterrows():
            original_val = row['product_category_name_english']
            cleaned_val = original_val.replace('_', ' ')
            category_df.at[idx, 'product_category_name_english'] = cleaned_val
            audit_log_object.log('clean_categories', 'product_category_name_english', idx, 
                                 original_val, cleaned_val, 'Replacing underscore with space')

    return product_df, category_df, order_item_df, order_df


class CleaningPipeline:
    """
    Modular, auditable data cleaning pipeline for DW tables.
    Each step is registered as a named transformation.
    All changes are captured in AuditLog.
    """
    def __init__(self, df: pd.DataFrame, table_name: str, pk_col: str = None):
        self.original   = df.copy()
        self.df         = df.copy()
        self.table_name = table_name
        self.pk_col     = pk_col
        self.audit      = AuditLog()
        self._steps_run = []

    # ── Standardize strings ────────────────────────────────────────
    def standardize_strings(self, cols: list,
                              strip: bool = True,
                              lower: bool = False,
                              title_case: bool = False):
        """Strip whitespace, normalize casing."""
        # Standardize string columns by trimming whitespace and/or normalizing case

        for col in cols:
            # Iterate over the selected columns

            if col not in self.df.columns: continue
            # Skip missing columns

            before = self.df[col].copy()
            # Save original values for audit comparison

            s = self.df[col].astype(str)
            # Convert values to string for text operations

            if strip:      s = s.str.strip()
            # Remove leading and trailing whitespace if requested

            if lower:      s = s.str.lower()
            # Convert text to lowercase if requested

            if title_case: s = s.str.title()
            # Convert text to title case if requested

            changed = (s != before.astype(str)) & before.notna()
            # Identify rows where the standardized value differs from the original

            self.df.loc[changed, col] = s[changed]
            # Apply the transformation only to changed rows

            self.audit.log_batch('standardize_strings', col, changed,
                                  before, self.df[col],
                                  'Strip whitespace + normalize case')
            # Log all row-level changes

        self._steps_run.append('standardize_strings')
        # Register the executed step

        return self
        # Return self to allow method chaining

    # ── Canonicalize enum values ───────────────────────────────────
    def canonicalize_enum(self, col: str, mapping: dict,
                            unknown_value: str = 'Unknown'):
        """ 
        Map all variants of a categorical value to a canonical form.
        mapping: {canonical: [variant1, variant2, ...]}
        """ 
        if col not in self.df.columns: return self
        reverse = {}
        # Build reverse lookup dictionary: dirty variant -> canonical value
        for canonical, variants in mapping.items():
            for v in variants:
                reverse[v.strip().lower()] = canonical

        before = self.df[col].copy()

        def _map(val):
            if pd.isna(val): return val
            return reverse.get(str(val).strip().lower(), val)
        
        self.df[col] = self.df[col].apply(_map)
        changed = (self.df[col] != before) & before.notna()
        self.audit.log_batch('canonicalize_enum', col, changed,
                              before, self.df[col],
                              f'Canonical mapping for {col}')
        self._steps_run.append(f'canonicalize_enum:{col}')
        return self

    # ── Parse and standardize dates ────────────────────────────────
    def parse_dates(self, col: str, output_format: str = '%Y-%m-%d %H:%M:%S'):
        """Parse mixed date formats into a single canonical format."""

        if col not in self.df.columns: return self
        # Skip if column is missing

        before = self.df[col].copy()
        # Save original values

        parsed = pd.to_datetime(self.df[col], infer_datetime_format=True,
                                 dayfirst=False, errors='coerce')
        # Attempt to parse dates; invalid values become NaT

        failed = parsed.isna() & before.notna()
        # Identify rows that could not be parsed

        if failed.sum() > 0:
            print(f'  ⚠️  {failed.sum()} dates could not be parsed in {col}')
            # Print warning for unparseable dates

        self.df[col]  = parsed.dt.strftime(output_format).where(parsed.notna(), other=np.nan)
        # Convert parsed dates to the desired string format; keep invalid ones as NaN

        changed = (self.df[col].astype(str) != before.astype(str)) & before.notna()
        # Detect changed rows

        self.audit.log_batch('parse_dates', col, changed,
                              before, self.df[col], f'Normalize to {output_format}')
        # Log date transformations

        self._steps_run.append(f'parse_dates:{col}')
        # Register step execution

        return self
        # Return self for chaining

    # ── Winsorize con Consensus ──────────────────────────────────────
    def winsorize_with_consensus(self, col: str, outlier_mask: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99):
        """
        Applica la Winsorizzazione solo ai record identificati come outlier 
        dal metodo del Consensus (almeno N metodi concordi).
        """
        if col not in self.df.columns: return self
        before = self.df[col].copy()
        # Calcoliamo le soglie 
        lo = self.df[col].quantile(lower_pct)
        hi = self.df[col].quantile(upper_pct)
        # Applichiamo il clipping SOLO dove outlier_mask è True
        # I valori che non sono outlier nel consensus rimangono invariati
        self.df.loc[outlier_mask, col] = self.df.loc[outlier_mask, col].clip(lower=lo, upper=hi)

        # Identifichiamo i cambiamenti effettivi per l'audit log
        changed = (self.df[col] != before) & before.notna()

        # Registriamo nell'Audit Log
        self.audit.log_batch('winsorize_consensus', col, changed,
            before, self.df[col],
            f'Consensus Winsorize [{lower_pct:.0%}, {upper_pct:.0%}] → soglie [{lo:.2f}, {hi:.2f}]'
        )

        self._steps_run.append(f'winsorize_consensus:{col}')
        
        return self

    # Impute missing values
    def impute(self, col: str, strategy: str = 'median',
                group_col: str = None, mnar_flag: bool = False):
        """
        strategy: 'mean' | 'median' | 'mode' | 'constant:VALUE' | 'mnar_flag'
        group_col: if set, compute statistic within each group (MAR strategy)
        mnar_flag: if True, add a binary indicator column instead of imputing
        """
        if col not in self.df.columns: return self
        miss_mask = self.df[col].isna()
        if not miss_mask.any(): return self
        if mnar_flag:
            flag_col = f'{col}_missing_flag'
            # Build name of indicator column

            self.df[flag_col] = miss_mask.astype(int)
            # Add binary flag instead of imputing values

            self.audit.log('impute', col, 'ALL', 'null', f'flag → {flag_col}', 'MNAR — do not impute')
            # Log the MNAR handling decision

            self._steps_run.append(f'mnar_flag:{col}')
            # Register special MNAR step

            return self
            # Stop here because no imputation is performed
        
        before = self.df[col].copy()

        if group_col and group_col in self.df.columns:
            if strategy == 'median':
                fill_vals = self.df.groupby(group_col)[col].transform('median')
            elif strategy == 'mean':
                fill_vals = self.df.groupby(group_col)[col].transform('mean')
            else:
                fill_vals = self.df.groupby(group_col)[col].transform(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
                # Compute group-specific mode
            
            self.df[col] = self.df[col].fillna(fill_vals)
            # Fill missing values with group-level statistics

        elif strategy.startswith('constant:'):
            val = strategy.split(':')[1]
            self.df[col] = self.df[col].fillna(val)
            # Fill missing values with constant

        elif strategy in ('mean', 'median'):
            stat_val = getattr(self.df[col], strategy)()
            # Compute global mean or median

            self.df[col] = self.df[col].fillna(stat_val)
            # Fill missing values with global statistic
        
        elif strategy == 'mode':
            self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])
            # Fill missing values with most frequent value

        changed = miss_mask & self.df[col].notna()

        self.audit.log_batch('impute', col, changed,
                              before, self.df[col],
                              f'strategy={strategy}, group={group_col}')
        self._steps_run.append(f'impute:{col}')
        
        return self

    

city_corrections = {
    'sbc': 'Sao Bernardo Do Campo',
    's.b.c.': 'Sao Bernardo Do Campo', 
    'sao paulo': 'Sao Paulo',
    'são paulo': 'Sao Paulo',
}


def clean_seller_city_with_null(seller_df, audit_log_object):
    #Pipeline di pulizia per seller_city
    
    # Regex standard di validità
    base_regex = r'^[A-Za-zÀ-ÿã\s\'-]+$'
    
    for idx, row in seller_df.iterrows():
        original_val = str(row['seller_city'])
        current_val = original_val
        current_val = unicodedata.normalize('NFC', original_val)
        # Gestione / o \
        current_val = current_val.replace('\\', '/').split('/')[0]
        
        # Gestione virgola ,
        current_val = current_val.split(',')[0]
        
        # Gestione parentesi () e contenuto
        current_val = re.sub(r'\(.*\)', '', current_val)
        
        # Sostituzione ´ con '
        current_val = current_val.replace('´', "'")
        
        # Trimming (rimozione spazi inizio/fine)
        current_val = current_val.strip()
        
        # VERIFICA FINALE E GESTIONE INVALIDI 
        # Controlliamo se è ancora invalida e contiene numeri o @
        is_still_invalid = not bool(re.match(base_regex, current_val))
       
        
        if  original_val != current_val and not is_still_invalid:
            # Salvataggio valore pulito (se diverso dall'originale)
            seller_df.at[idx, 'seller_city'] = current_val
            audit_log_object.log('clean_city_format', 'seller_city', idx, 
                                 original_val, current_val, "Cleaned format")
        elif is_still_invalid:
            print("Invalid seller_city: ", current_val)
        
            
    return seller_df


#"quilometro 14 do mutum" è l'unico valore di customer_city invalido
# Filtra i customer che appartengono alla città specifica
target_city = "quilometro 14 do mutum"
customers_in_city = customer_df[customer_df['customer_city'] == target_city]['customer_id']

# Conta quanti ordini in order_df sono associati a questi customer_id
num_orders = order_df[order_df['customer_id'].isin(customers_in_city)].shape[0]

print(f"Number of orders for the city'{target_city}': {num_orders}")

#Dal momento che gli ordini nella tabella Order sono moltissimi e solo un order ha il customer della città "quilometro 14 do mutum",
#non importa se eliminiamo quell'ordine, il customer e gli order item associati.



def clean_invalid_cities_cascade(customer_df, order_df, order_item_df, audit_log_object):
    city_regex = r'^[A-Za-zÀ-ÿã\s\'-]+$'
    
    invalid_customers_mask = ~customer_df['customer_city'].str.match(city_regex, na=False)
    invalid_customer_ids = customer_df[invalid_customers_mask]['customer_id'].unique()
    
    print(f"Customers with invalid city identified: {len(invalid_customer_ids)}")
    
    if len(invalid_customer_ids) == 0:
        return customer_df, order_df, order_item_df

    orders_to_remove_mask = order_df['customer_id'].isin(invalid_customer_ids)
    order_ids_to_remove = order_df[orders_to_remove_mask]['order_id'].unique()
    
    
    initial_items = len(order_item_df)
    order_item_df = order_item_df[~order_item_df['order_id'].isin(order_ids_to_remove)].reset_index(drop=True)
    items_removed = initial_items - len(order_item_df)
    if items_removed > 0:
        audit_log_object.log('cascade_city_cleanup', 'order_item_id', 'ALL', 
                             'EXISTING', 'REMOVED', f'Removed {items_removed} items due to invalid customer city')
    
    initial_orders = len(order_df)
    order_df = order_df[~order_df['customer_id'].isin(invalid_customer_ids)].reset_index(drop=True)
    orders_removed = initial_orders - len(order_df)
    if orders_removed > 0:
        audit_log_object.log('cascade_city_cleanup', 'order_id', 'ALL', 
                             'EXISTING', 'REMOVED', f'Removed {orders_removed} orders due to invalid customer city')
    
    initial_cust = len(customer_df)
    customer_df = customer_df[~invalid_customers_mask].reset_index(drop=True)
    cust_removed = initial_cust - len(customer_df)
    if cust_removed > 0:
        audit_log_object.log('cascade_city_cleanup', 'customer_city', 'MULTIPLE', 
                             'INVALID_FORMAT', 'REMOVED', f'Removed {cust_removed} customers with invalid city names')
    
    print("--- Deletion Report ---")
    print(f"Customers removed: {cust_removed}")
    print(f"Orders removed: {orders_removed}")
    print(f"Order Items (OrderItem) removed: {items_removed}")
    
    return customer_df, order_df, order_item_df

audit = AuditLog()

product_df, category_df, order_item_df, order_df = fix_and_clean_product_categories(
    product_df, category_df, order_item_df, order_df, audit)

seller_df = clean_seller_city_with_null(seller_df, audit)
customer_df, order_df, order_item_df = clean_invalid_cities_cascade(customer_df, order_df, order_item_df, audit)

print(audit.summary())


#GESTIONE DEGLI ULTIMI DUE VALORI INVALIDI DI seller_city RIMASTI (ovvero i valori "vendas@creditparts.com.br" e "04482255")

# Identifica i seller_id associati alla "città" (email) specifica
target_email_city = "vendas@creditparts.com.br"
target_seller_ids = seller_df[seller_df['seller_city'] == target_email_city]['seller_id']

# Conta gli OrderItem associati a questi seller_id
# Nota: Gli OrderItem hanno il riferimento diretto al seller_id
items_count = order_item_df[order_item_df['seller_id'].isin(target_seller_ids)].shape[0]

# Conta gli Ordini unici associati a questi articoli
# Dobbiamo passare per order_item_df perché order_df non contiene il seller_id
relevant_order_ids = order_item_df[order_item_df['seller_id'].isin(target_seller_ids)]['order_id'].unique()
orders_count = len(relevant_order_ids)

print(f" Results for seller_city= '{target_email_city}':")
print(f"  - Number of OrderItems: {items_count}")
print(f"  - Number of Unique Orders: {orders_count}")

# Identifica i seller_id associati alla "città" specifica
target_numbers_city = "04482255"
target_seller_ids = seller_df[seller_df['seller_city'] == target_numbers_city]['seller_id']

# Conta gli OrderItem associati a questi seller_id
# Nota: Gli OrderItem hanno il riferimento diretto al seller_id
items_count = order_item_df[order_item_df['seller_id'].isin(target_seller_ids)].shape[0]

# Conta gli Ordini unici associati a questi articoli
# Dobbiamo passare per order_item_df perché order_df non contiene il seller_id
relevant_order_ids = order_item_df[order_item_df['seller_id'].isin(target_seller_ids)]['order_id'].unique()
orders_count = len(relevant_order_ids)

print(f" Results for seller_city= '{target_numbers_city}':")
print(f"  - Number of OrderItems: {items_count}")
print(f"  - Number of Unique Orders: {orders_count}")


def process_specific_sellers(seller_df, order_df, order_item_df, audit_log_object): 
    invalid_cities = ["04482255", "vendas@creditparts.com.br"]
    
    # Identificazione di TUTTI i seller_id associati a questi valori invalidi
    to_delete_ids = seller_df[seller_df['seller_city'].isin(invalid_cities)]['seller_id'].unique()
    
    if len(to_delete_ids) > 0:
        
        relevant_order_ids = order_item_df[order_item_df['seller_id'].isin(to_delete_ids)]['order_id'].unique()
        
        # Rimozione degli OrderItem associati a questi seller
        initial_items = len(order_item_df)
        order_item_df = order_item_df[~order_item_df['seller_id'].isin(to_delete_ids)].reset_index(drop=True)
        items_removed = initial_items - len(order_item_df)
        
        # Rimozione degli Order che sono rimasti orfani (senza più alcun articolo nel carrello)
        valid_order_ids = order_item_df['order_id'].unique()
        initial_orders = len(order_df)
        order_df = order_df[order_df['order_id'].isin(valid_order_ids)].reset_index(drop=True)
        orders_removed = initial_orders - len(order_df)
        
        # Rimozione dei Seller dalla tabella Seller
        initial_sellers = len(seller_df)
        seller_df = seller_df[~seller_df['seller_id'].isin(to_delete_ids)].reset_index(drop=True)
        sellers_removed = initial_sellers - len(seller_df)
        
        audit_log_object.log('cascade_delete_seller', 'seller_id', 'ALL', 
                             f"Cities: {invalid_cities}", 'REMOVED', 
                             f'Removed {sellers_removed} sellers, {orders_removed} orders, {items_removed} items')

        print(f"--- Deletion Report (Specific Sellers) ---")
        print(f"Sellers removed: {sellers_removed}")
        print(f"Orders removed: {orders_removed}")
        print(f"Order Items (OrderItem) removed: {items_removed}")
    else:
        print("No matching invalid sellers found for deletion.")

    print(f" Specific processing completed.")
    return seller_df, order_df, order_item_df



seller_df, order_df, order_item_df = process_specific_sellers(
    seller_df, 
    order_df, 
    order_item_df, 
    audit
)

def print_missing_values(df, table_name="DataFrame"):
    
    # Stampa il conteggio dei valori nulli per colonna e la relativa percentuale.
    
    print(f"--- Null Values ​​Summary: {table_name} ---")
    
    # Calcolo del numero di nulli per colonna
    missing_count = df.isnull().sum()
    
    # Calcolo della percentuale
    missing_pct = (missing_count / len(df)) * 100
    
    
    summary_df = pd.DataFrame({
        'Null Values': missing_count,
        'Percentage (%)': missing_pct.map('{:.2f}%'.format)
    })
    
    # Filtriamo per mostrare solo le colonne con almeno un valore nullo
    only_missing = summary_df[summary_df['Null Values'] > 0]
    
    if not only_missing.empty:
        print(only_missing)
    else:
        print(" No null values ​​found in the analyzed columns.")
    print("-" * 40)


#Inconsistency Handling 
inconsistent_order_ids = order_df[order_df['order_delivered_carrier_date'] < order_df['order_purchase_timestamp']]['order_id']
order_df = order_df[~order_df['order_id'].isin(inconsistent_order_ids)].reset_index(drop=True)
order_item_df = order_item_df[~order_item_df['order_id'].isin(inconsistent_order_ids)].reset_index(drop=True)



#Handling missing values
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
miss_before_order = (order_df.isnull().sum() / len(order_df) * 100)
miss_before_product = (product_df.isnull().sum() / len(product_df) * 100)

# Order Table
""" Missing Value Mechanism is MAR (Missing At Random) since the missingness of the
columns order_approved_at, order_delivered_carrier_date and
order_delivered_customer_date depends on the observed variable order_status """

# Volendo fare analisi sugli ordini realmente effettuati,
# eliminiamo le righe di order_df che rappresentano ordini aventi order_status = canceled
# ed eliminiamo le righe in order_item_df in cui sono presenti riferimenti a tali ordini.
# In tal modo i valori mancanti diminuiscono notevolmente.

print("Missing values in Order Table before cleaning")
print_missing_values(order_df, "order_df")

def clean_canceled_orders(order_df, order_item_df, audit_log_object):
    """
    Rimuove gli ordini con stato 'canceled' da order_df e propaga 
    la cancellazione su order_item_df per mantenere la coerenza.
    """
    initial_orders = len(order_df)
    initial_items = len(order_item_df)

    print(f" Initial rows - Orders: {len(order_df)}, Items: {len(order_item_df)}")
    
    # Identifichiamo gli ordini da mantenere (quelli NON 'canceled')
    order_df_cleaned = order_df[order_df['order_status'] != 'canceled'].copy()
    
    # Otteniamo la lista degli ID ordini validi rimasti
    valid_order_ids = order_df_cleaned['order_id']
    
    # Filtriamo order_item_df: teniamo solo gli item il cui order_id esiste ancora
    order_item_df_cleaned = order_item_df[order_item_df['order_id'].isin(valid_order_ids)].copy()
    orders_removed = initial_orders - len(order_df_cleaned)
    items_removed = initial_items - len(order_item_df_cleaned)

    
    if orders_removed > 0:
        audit_log_object.log(
            step='clean_canceled_orders', 
            col='order_status', 
            idx='ALL', 
            before='canceled', 
            after='REMOVED', 
            reason=f'Removed {orders_removed} canceled orders and {items_removed} related items'
        )

    print(f"Final rows   - Orders: {len(order_df_cleaned)}, Items: {len(order_item_df_cleaned)}")
    print(f"Removed {len(order_df) - len(order_df_cleaned)} caceled orders and their items.")
    
    return order_df_cleaned, order_item_df_cleaned

order_df, order_item_df = clean_canceled_orders(order_df, order_item_df, audit)

order_df['order_approved_at'] = pd.to_datetime(order_df['order_approved_at'])
order_df['order_delivered_carrier_date'] = pd.to_datetime(order_df['order_delivered_carrier_date'])
order_df['order_delivered_customer_date'] = pd.to_datetime(order_df['order_delivered_customer_date'])
# Creazione di colonne numeriche temporanee (ogni data è rappresentata in termini di secondi trascorsi dal 1° gennaio 1970)
order_df['order_approved_at_num'] = order_df['order_approved_at'].view(np.int64) // 10**9
order_df.loc[order_df['order_approved_at'].isna(), 'order_approved_at_num'] = np.nan
order_df['order_delivered_carrier_date_num'] = order_df['order_delivered_carrier_date'].view(np.int64) // 10**9
order_df.loc[order_df['order_delivered_carrier_date'].isna(), 'order_delivered_carrier_date_num'] = np.nan
order_df['order_delivered_customer_date_num'] = order_df['order_delivered_customer_date'].view(np.int64) // 10**9
order_df.loc[order_df['order_delivered_customer_date'].isna(), 'order_delivered_customer_date_num'] = np.nan


pipe_order = (CleaningPipeline(order_df, 'order_fact', pk_col='order_id')
            .parse_dates('order_purchase_timestamp', output_format='%Y-%m-%d %H:%M:%S')
            .parse_dates('order_approved_at', output_format='%Y-%m-%d %H:%M:%S')
            .parse_dates('order_delivered_carrier_date', output_format='%Y-%m-%d %H:%M:%S')
            .parse_dates('order_delivered_customer_date', output_format='%Y-%m-%d %H:%M:%S')
            .parse_dates('order_estimated_delivery_date', output_format='%Y-%m-%d %H:%M:%S')
            .impute(col='order_approved_at_num', strategy='mean', group_col='order_status')
            .impute(col='order_delivered_carrier_date_num', strategy='mean', group_col='order_status')
            .impute(col='order_delivered_customer_date_num', strategy='mean', group_col='order_status')
)
order_df = pipe_order.df
# Arrotondamento dei valori imputati all' intero più vicino e gestione dei nulli residui
for col in ['order_approved_at_num', 'order_delivered_carrier_date_num', 'order_delivered_customer_date_num']:
    order_df[col] = order_df[col].round().astype('Int64')

order_df['order_approved_at'] = pd.to_datetime(order_df['order_approved_at_num'], unit='s')
order_df['order_delivered_carrier_date'] = pd.to_datetime(order_df['order_delivered_carrier_date_num'], unit='s')
order_df['order_delivered_customer_date'] = pd.to_datetime(order_df['order_delivered_customer_date_num'], unit='s')
# Rimuoviamo le colonne numeriche di supporto
order_df = order_df.drop(columns=['order_approved_at_num'])
order_df = order_df.drop(columns=['order_delivered_carrier_date_num'])
order_df = order_df.drop(columns=['order_delivered_customer_date_num'])




rows_with_nulls = order_df.isnull().any(axis=1).sum()

print(f"Number of rows in order_df with at least one null value: {rows_with_nulls}")


total_rows = len(order_df)
percentage = (rows_with_nulls / total_rows) * 100
print(f"Percentage of “dirty” rows: {percentage:.2f}%")

def drop_null_orders_cascade(order_df, order_item_df, audit_log_object):
    """
    Rimuove gli ordini (pochissimi) con almeno un valore nullo e propaga 
    la cancellazione agli articoli corrispondenti.
    """
    initial_orders = len(order_df)
    order_df_cleaned = order_df.dropna(axis=0, how='any').reset_index(drop=True)
    
    orders_removed = initial_orders - len(order_df_cleaned)
    
    valid_order_ids = order_df_cleaned['order_id'].unique()
    
    initial_items = len(order_item_df)
    order_item_df_cleaned = order_item_df[order_item_df['order_id'].isin(valid_order_ids)].reset_index(drop=True)
    
    items_removed = initial_items - len(order_item_df_cleaned)
    
    print("--- Null Row Deletion Report ---")
    print(f"Orders deleted: {orders_removed}")
    print(f"Items (OrderItems) deleted: {items_removed}")
    print(f"Remaining rows in order_df: {len(order_df_cleaned)}")
    
    return order_df_cleaned, order_item_df_cleaned


order_df, order_item_df = drop_null_orders_cascade(order_df, order_item_df, audit)


print("Missing values in Order Table after cleaning")
print_missing_values(order_df, "order_df")

# Nella tabella Product l'attributo product_category_name ha dei valori nulli
# Identifichiamo i product_id che hanno la categoria nulla nella tabella Product
invalid_product_ids = product_df[product_df['product_category_name'].isna()]['product_id']
# Contiamo quante righe in order_item_df hanno un product_id presente in questo elenco
matching_order_items = order_item_df[order_item_df['product_id'].isin(invalid_product_ids)]
num_rows = len(matching_order_items)
print(f"Number of rows in OrderItem corresponding to products with null product_category_name: {num_rows}")
unique_orders_count = matching_order_items['order_id'].nunique()
print(f"Number of orders matching those items: {unique_orders_count}")

def remove_null_category_cascade(product_df, order_item_df, order_df, audit_log_object):
    """
    Rimuove i prodotti senza categoria e propaga la cancellazione su OrderItem e Order.
    """
    # Identifica i product_id da rimuovere (hanno categoria nulla)
    mask_null_prod = product_df['product_category_name'].isna()
    product_ids_to_remove = product_df.loc[mask_null_prod, 'product_id'].unique()
    
    print(f"Products with NULL category to be removed: {len(product_ids_to_remove)}")

    # Identificazione degli OrderItem collegati a questi prodotti
    mask_items_to_remove = order_item_df['product_id'].isin(product_ids_to_remove)
    order_ids_affected = order_item_df.loc[mask_items_to_remove, 'order_id'].unique()
    

    initial_products = len(product_df)
    product_df = product_df[~mask_null_prod].reset_index(drop=True)
    
    initial_items = len(order_item_df)
    order_item_df = order_item_df[~mask_items_to_remove].reset_index(drop=True)
    items_removed = initial_items - len(order_item_df)

    # Un ordine va rimosso solo se NON ha più alcun articolo rimasto in order_item_df
    initial_orders = len(order_df)
    remaining_order_ids = order_item_df['order_id'].unique()
    
    order_df = order_df[order_df['order_id'].isin(remaining_order_ids)].reset_index(drop=True)
    orders_removed = initial_orders - len(order_df)

    audit_log_object.log('cascade_delete_null_cat', 'multiple', 'ALL', 
                         'NULL_CATEGORY', 'REMOVED', 
                         f'Removed {len(product_ids_to_remove)} products, {items_removed} items, {orders_removed} orders')

    print("--- Deletion Report ---")
    print(f"Removed rows in product_df:    {initial_products - len(product_df)}")
    print(f"Removed rows in order_item_df: {items_removed}")
    print(f"Removed rows in order_df:      {orders_removed}")
    
    return product_df, order_item_df, order_df


product_df, order_item_df, order_df = remove_null_category_cascade(
    product_df, order_item_df, order_df, audit
)



pipe_customer = (CleaningPipeline(customer_df, 'customer_fact', "customer_id")
                 .standardize_strings(['customer_city'], title_case = True)
                 .canonicalize_enum('customer_city', city_corrections)
)

customer_df = pipe_customer.df



pipe_seller = (CleaningPipeline(seller_df, 'seller_fact', "seller_id")
                .standardize_strings(['seller_city'], title_case = True) 
                .canonicalize_enum('seller_city', city_corrections)
)

seller_df = pipe_seller.df

pipe_product = (CleaningPipeline(product_df, 'product_fact', "product_category_name")
                 .standardize_strings(['product_category_name'], lower = True)
)

product_df = pipe_product.df

#Handling outliers

cols_transazionali = ['price', 'freight_value']

od_transazioni = ( 
    OutlierDetector(order_item_df, 'Olist_Transactions')
    .detect_iqr(cols_transazionali, multiplier=3.0)
    .detect_modified_zscore(cols_transazionali, threshold=3.5)
    .detect_isolation_forest(cols_transazionali, contamination=0.02)
)

consensus_outliers = od_transazioni.consensus(min_methods=2)

pipe_order_item = (CleaningPipeline(order_item_df, 'order_item_fact')
                    .winsorize_with_consensus('price', consensus_outliers, lower_pct=0.01, upper_pct=0.99)
                    .winsorize_with_consensus('freight_value', consensus_outliers, lower_pct=0.01, upper_pct=0.99)
                    
)

order_item_df = pipe_order_item.df


date_cols_fix = ['order_approved_at', 'order_delivered_carrier_date', 
                 'order_delivered_customer_date', 'order_estimated_delivery_date', 
                 'order_purchase_timestamp']


mask_app_late = (order_df['order_approved_at'] > order_df['order_delivered_carrier_date']) & \
                order_df['order_approved_at'].notna() & order_df['order_delivered_carrier_date'].notna()

if mask_app_late.any():
    before_vals = order_df['order_approved_at'].copy()
    
    order_df.loc[mask_app_late, 'order_approved_at'] = order_df.loc[mask_app_late, 'order_delivered_carrier_date']
    
    audit.log_batch('fix_temporal_inconsistency', 'order_approved_at', mask_app_late, 
                     before_vals, order_df['order_approved_at'], 
                     'Approved date after carrier delivery; forced to carrier date')

mask_delivery_early = (order_df['order_delivered_customer_date'] < order_df['order_delivered_carrier_date']) & \
                      order_df['order_delivered_customer_date'].notna() & \
                      order_df['order_delivered_carrier_date'].notna()

if mask_delivery_early.any():
    before_vals = order_df['order_delivered_customer_date'].copy()
    
    order_df.loc[mask_delivery_early, 'order_delivered_customer_date'] = \
        order_df.loc[mask_delivery_early, 'order_delivered_carrier_date'] + pd.Timedelta(hours=1)
    
    audit.log_batch('fix_temporal_inconsistency', 'order_delivered_customer_date', mask_delivery_early, 
                     before_vals, order_df['order_delivered_customer_date'], 
                     'Delivery before shipping; added 1h buffer')


mask_est_early = (order_df['order_estimated_delivery_date'] < order_df['order_delivered_carrier_date']) & \
                 order_df['order_estimated_delivery_date'].notna() & \
                 order_df['order_delivered_carrier_date'].notna()

if mask_est_early.any():
    before_vals = order_df['order_estimated_delivery_date'].copy()
    
    order_df.loc[mask_est_early, 'order_estimated_delivery_date'] = \
        order_df.loc[mask_est_early, 'order_delivered_carrier_date'] + pd.Timedelta(days=1)
    
    audit.log_batch('fix_temporal_inconsistency', 'order_estimated_delivery_date', mask_est_early, 
                     before_vals, order_df['order_estimated_delivery_date'], 
                     'Estimated delivery before shipping; added 1d buffer')

    
mask_lower_cat_pt = category_df['product_category_name'].notna() & \
                    (category_df['product_category_name'].astype(str) != category_df['product_category_name'].astype(str).str.lower().str.strip())

if mask_lower_cat_pt.any():
    before_vals = category_df['product_category_name'].copy()
    category_df['product_category_name'] = category_df['product_category_name'].astype(str).str.lower().str.strip()
    audit.log_batch('final_forced_lowercase', 'product_category_name', mask_lower_cat_pt,
                    before_vals, category_df['product_category_name'],
                    'Forced lowercase coercion to resolve remaining consistency violations')

mask_lower_cat_en = category_df['product_category_name_english'].notna() & \
                    (category_df['product_category_name_english'].astype(str) != category_df['product_category_name_english'].astype(str).str.lower().str.strip())

if mask_lower_cat_en.any():
    before_vals = category_df['product_category_name_english'].copy()
    category_df['product_category_name_english'] = category_df['product_category_name_english'].astype(str).str.lower().str.strip()
    audit.log_batch('final_forced_lowercase', 'product_category_name_english', mask_lower_cat_en,
                    before_vals, category_df['product_category_name_english'],
                    'Forced lowercase coercion to resolve remaining consistency violations')


subset_cols1 = ['product_category_name']
final_duplicate_mask = category_df.duplicated(subset=subset_cols1, keep='first')
    
if final_duplicate_mask.any():
    print(f"\n Rilevati {final_duplicate_mask.sum()} duplicati finali nella colonna primaria product_category_name.")
    final_duplicated_rows = category_df[final_duplicate_mask]
    for idx, row in final_duplicated_rows.iterrows():
        val_pt = row['product_category_name']
        val_en = row['product_category_name_english']
        audit.log(
            step='final_pipeline_deduplication', 
            col='product_category_name', 
            idx=idx, 
            before=f"PT: {val_pt} | EN: {val_en}", 
            after='DELETED_POST_PIPELINE_DUPLICATE', 
            reason='Removed duplicate row to ensure Primary Key uniqueness'
        )

category_df = category_df.drop_duplicates(subset=subset_cols1, keep='first').reset_index(drop=True)
print(f"Removal complete. Unique records remaining in category_df: {len(category_df)}")


output_files = {
    "customer": "cleaned_olist_customers.csv",
    "seller": "cleaned_olist_sellers.csv",
    "product": "cleaned_olist_products.csv",
    "order": "cleaned_olist_orders.csv",
    "order_item": "cleaned_olist_order_items.csv",
    "category": "cleaned_olist_category.csv"
}




customer_df.to_csv(os.path.join(BASE_PATH, output_files["customer"]), index=False)
order_df.to_csv(os.path.join(BASE_PATH, output_files["order"]), index=False)
order_item_df.to_csv(os.path.join(BASE_PATH, output_files["order_item"]), index=False)
product_df.to_csv(os.path.join(BASE_PATH, output_files["product"]), index=False)
seller_df.to_csv(os.path.join(BASE_PATH, output_files["seller"]), index=False)
category_df.to_csv(os.path.join(BASE_PATH, output_files["category"]), index=False)
product_df.to_csv(os.path.join(BASE_PATH, output_files["product"]), index=False)


print(f"Files successfully saved to directory: {BASE_PATH}")
print(f"Files created: {list(output_files.values())}")

# Missing values before vs after
miss_after_order = (order_df.isnull().sum() / len(order_df) * 100)
cols_to_show_order = [c for c in ["order_approved_at", "order_delivered_carrier_date","order_delivered_customer_date"] if c in miss_before_order.index]
x = np.arange(len(cols_to_show_order))
rects1 = ax.bar(x - 0.2, miss_before_order[cols_to_show_order], 0.35, label='Before', color='#e74c3c', alpha=0.8)
rects2 = ax.bar(x + 0.2, miss_after_order[cols_to_show_order], 0.35, label='After', color='#2ecc71', alpha=0.8)
ax.bar_label(rects1, padding=3, fmt='%.2f%%', fontsize=8, fontweight='bold', rotation=45)
ax.bar_label(rects2, padding=3, fmt='%.2f%%', fontsize=8, fontweight='bold', rotation=45)
ax.set_xticks(x)
ax.set_xticklabels(cols_to_show_order, rotation=15)
ax.set_ylabel('% missing')
ax.set_title('Missing Values in the Order Table: Before vs After\n', fontweight='bold', fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()

fig_output_path = BASE_PATH + 'order_missingness_cleaning_comparisonFinal.png'
plt.savefig(fig_output_path, bbox_inches='tight', dpi=120)

plt.show()

print(order_df.groupby('order_status')['order_approved_at'].count())
print(order_df.groupby('order_status')['order_delivered_customer_date'].count())
print(order_df.groupby('order_status')['order_delivered_carrier_date'].count())



