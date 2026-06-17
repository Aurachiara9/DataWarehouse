import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
np.random.seed(42)


# MCAR / MAR / MNAR Classification
class MissingnessAnalyzer:
    """ 
    Structured analysis of missing values.
    Provides tests to classify MCAR / MAR / MNAR.
    """
    def __init__(self,df:pd.DataFrame, table_name:str):
        self.df = df.copy()
        self.table_name = table_name
        self.results = {}
    
    def summary(self) -> pd.DataFrame:
        miss = self.df.isnull().sum()
        total = len(self.df)
        pct = (miss / total * 100).round(2)
        result = pd.DataFrame({
            'missing_count' : miss,
            'missing_pct' : pct,
            'present_count' : total - miss,
            'dtype' : self.df.dtypes
        })

        result['status'] = result['missing_pct'].apply(
            lambda p: '🟢 OK' if p == 0 else ('🟡 Moderate' if p < 10 else '🔴 High')
        )

        return result[result['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    def test_mcar_chi2(self, col: str, group_col: str) -> dict:
        """
        Chi-squared test of independence:
        H0: missingness in `col` is independent of `group_col` (MCAR)
        Low p-value → reject H0 → likely MAR (depends on group_col)
        """
        temp = pd.DataFrame({
            'is_missing': self.df[col].isna().astype(int),
            'group' : self.df[group_col].fillna('__MISSING__')
        })
        
        contingency = pd.crosstab(temp['is_missing'], temp['group'])
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        return {
            'col' : col,
            'group_col' : group_col,
            'chi2' : round(chi2, 4),
            'p_value' : round(p_value, 6),
            'dof' : dof,
            'verdict' : ('MAR — missingness depends on '+ group_col)
                        if p_value < 0.05 else'Cannot reject MCAR',
            'significance' : '***' if p_value < 0.001 else ('**' if p_value < 0.01 else
                             ('*' if p_value < 0.05 else 'ns'))
        }

    def test_mar_logistic(self, col : str, candidate_predictors : list) -> dict:
        # Define a logistic regression test to evaluate whether missingness in one column
        # can be predicted from observed numeric variables
        # High pseudo-R² → strong MAR signal.
        
        y = self.df[col].isna().astype(int)
        predictors = [c for c in candidate_predictors 
                        if c in self.df.columns and c != col
                        and pd.api.types.is_numeric_dtype(self.df[c])]
        X_raw = self.df[predictors].fillna(self.df[predictors].median())
        X = StandardScaler().fit_transform(X_raw)
        X_sm = sm.add_constant(X)
        try:
            model = sm.Logit(y, X_sm).fit(disp=False)
            pseudo_r2 = model.prsquared
            verdict = ('Strong MAR signal' if pseudo_r2 > 0.1 else
                      ('Weak MAR signal' if pseudo_r2 > 0.02 else 'Consistent with MCAR'))
            # Interpret the pseudo-R²:
            # - above 0.1 → strong evidence that missingness depends on observed variables
            # - above 0.02 → weak evidence
            # - otherwise → weak or no MAR signal
            return {'col': col, 'predictors': predictors, 
                    'pseudo_r2': round(pseudo_r2,4), 'verdict': verdict}
        except Exception as e:
            return {'col': col, 'predictors': predictors, 'pseudo_r2': None, 'verdict': f'Test failed:{e}'}
        
    def test_mnar_correlation(self, col: str, numeric_cols: list) -> dict:
        """
        Proxy MNAR test: compare mean of other numeric columns
        between rows where `col` is missing vs present.
        Large differences suggest MNAR (the missing values are 'special').
        """
        present = self.df[self.df[col].notna()]
        missing = self.df[self.df[col].isna()]
        diffs = {} 
        # Initialize a dictionary to store comparison results for each numeric column.
        for nc in numeric_cols:
            if nc == col or nc not in self.df.columns: continue
            if not pd.api.types.is_numeric_dtype(self.df[nc]): continue
            mu_pres = present[nc].mean()
            mu_miss = missing[nc].mean()
            if pd.notna(mu_pres) and pd.notna(mu_miss):
                _, pval = stats.ttest_ind(present[nc].dropna(), missing[nc].dropna(),
                            equal_var=False)
                          # Run Welch’s t-test to compare the distributions between the two groups.
                diffs[nc] = {'mean_present': round(mu_pres, 2),
                             'mean_missing': round(mu_miss, 2),
                              'p_value': round(pval, 4),
                              'significant': pval < 0.05}
                               # Mark whether the difference is statistically significant.
        
        mnar_signal = any(v['significant'] for v in diffs.values())

        return {'col': col, 'comparisons': diffs,

                'mnar_signal': mnar_signal,
                # Return a boolean indicating whether any proxy MNAR signal was detected.
                'verdict': ('Possible MNAR — significant group differences' if mnar_signal
                           else 'No MNAR signal detected')}

print('✅ MissingnessAnalyzer class defined')

class OutlierDetector:
    def __init__(self, df: pd.DataFrame, table_name: str):
        self.df         = df.copy()
        self.table_name = table_name
        self.flags      = pd.DataFrame(False, index=df.index,
                                       columns=['iqr','zscore','mod_zscore',
                                                'isolation_forest','lof'])
        # Initialize a boolean DataFrame that will store row-level outlier flags
        # for each detection method. All rows start as False (not flagged).
        
        self.results    = {}

    # ── IQR fence ─────────────────────────────────────────────────────────
    def detect_iqr(self, cols: list, multiplier: float = 1.5):
        # Define a univariate outlier detector based on the IQR rule.
        flag = pd.Series(False, index=self.df.index)
        # Initialize a boolean Series to accumulate row-level outlier flags across columns.
       

        details = {}
        for col in cols:  
                         
            if col not in self.df.columns: continue
            s  = self.df[col].dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR    = Q3 - Q1
            # Compute the interquartile range.
            lo, hi = Q1 - multiplier * IQR, Q3 + multiplier * IQR
            # Compute lower and upper IQR fences.
            col_flag = ((self.df[col] < lo) | (self.df[col] > hi)) & self.df[col].notna()
            # Flag rows where the value falls outside the IQR fences
            # and is not missing.
            flag |= col_flag
            details[col] = {'Q1': round(Q1,2), 'Q3': round(Q3,2), 'IQR': round(IQR,2),
                            'lower_fence': round(lo,2), 'upper_fence': round(hi,2),
                            'outlier_count': int(col_flag.sum())}
            # Store per-column statistics and number of detected outliers.
        
        self.flags['iqr'] = flag
        # Save the final row-level IQR outlier flags.
        self.results['iqr'] = {'total_flagged': int(flag.sum()),
                               'pct': round(flag.mean()*100, 2), 'details': details}
        # Save the total number and percentage of flagged rows, plus detailed per-column info.
        return self
    
    # ── Z-score ────────────────────────────────────────────────────────────
    def detect_zscore(self, cols: list, threshold: float = 3.0):
        # Define a univariate outlier detector based on standard Z-scores.
        flag = pd.Series(False, index=self.df.index)
        # Initialize a boolean Series for accumulated row-level flags.
        details = {} # Initialize a dictionary to store per-column Z-score details.
        for col in cols:
            if col not in self.df.columns: continue
            s         = self.df[col].fillna(self.df[col].mean())
            # Fill missing values with the column mean so Z-scores can be computed.
            z         = np.abs(stats.zscore(s))
            # Compute absolute Z-scores for the column.
            col_flag  = pd.Series(z > threshold, index=self.df.index) & self.df[col].notna()
            # Flag rows whose absolute Z-score exceeds the threshold
            # and whose original value is not missing.
            flag |= col_flag
            # Accumulate row-level flags across columns.
            details[col] = {'mean': round(self.df[col].mean(), 2),
                            'std' : round(self.df[col].std(), 2),
                            'max_z': round(z.max(), 2),
                            'outlier_count': int(col_flag.sum())}
            # Store summary statistics and number of outliers for the column.
        
        self.flags['zscore'] = flag
        # Save final row-level Z-score flags.

        self.results['zscore'] = {'total_flagged': int(flag.sum()),
                                  'pct': round(flag.mean()*100, 2), 'details': details}
        # Save overall Z-score summary results.

        return self
    
    # ── Modified Z-score (robust — uses median) ───────────────────────────
    def detect_modified_zscore(self, cols: list, threshold: float = 3.5):
        # Define a robust univariate outlier detector based on median and MAD.

        flag = pd.Series(False, index=self.df.index)
        # Initialize a boolean Series for accumulated row-level flags.

        details = {}
        # Initialize a dictionary to store per-column Modified Z-score details.

        for col in cols:
            if col not in self.df.columns: continue
            s      = self.df[col].dropna()
            median = s.median()
            mad    = (s - median).abs().median()
            # Compute the Median Absolute Deviation (MAD).
            if mad == 0: continue
            mod_z  = 0.6745 * (self.df[col] - median) / mad
            # Compute the Modified Z-score using the standard scaling constant.
            col_flag = (mod_z.abs() > threshold) & self.df[col].notna()
            # Flag rows whose absolute Modified Z-score exceeds the threshold
            # and whose original value is not missing.
            flag |= col_flag
            # Accumulate row-level flags across columns.
            details[col] = {'median': round(median, 2), 'MAD': round(mad, 2),
                            'max_mod_z': round(mod_z.abs().max(), 2),
                            'outlier_count': int(col_flag.sum())}
            self.flags['mod_zscore'] = flag
            self.results['mod_zscore'] = {'total_flagged': int(flag.sum()),
                                      'pct': round(flag.mean()*100, 2), 'details': details}
            return self
        
    # ── Isolation Forest (multivariate) ───────────────────────────────────
    def detect_isolation_forest(self, cols: list,
                                 contamination: float = 0.05,
                                 n_estimators: int = 100):
        # Define a multivariate outlier detector based on Isolation Forest.
        valid_cols = [c for c in cols if c in self.df.columns
                    and pd.api.types.is_numeric_dtype(self.df[c])]

        X = self.df[valid_cols].fillna(self.df[valid_cols].median())
        # Build the feature matrix and fill missing values with median values.
        X_scaled = StandardScaler().fit_transform(X)
        # Standardize features before fitting the model.
        model  = IsolationForest(n_estimators=n_estimators,
                                 contamination=contamination,
                                 random_state=42)
        # Initialize the Isolation Forest model with chosen parameters.
        preds  = model.fit_predict(X_scaled)
        # Fit the model and predict labels:
        # -1 means outlier, 1 means inlier.

        scores = model.decision_function(X_scaled)
        # Compute anomaly scores: lower values indicate more anomalous rows.

        flag   = pd.Series(preds == -1, index=self.df.index)
        # Convert model predictions into a boolean outlier flag Series.

        self.flags['isolation_forest'] = flag
        # Save row-level Isolation Forest flags.

        self.results['isolation_forest'] = {
        'total_flagged': int(flag.sum()),
        # Store total number of outlier rows.

        'pct': round(flag.mean()*100, 2),
        # Store percentage of outlier rows.

        'cols_used': valid_cols,
        # Store which columns were used in the model.

        'min_score': round(scores.min(), 4),
        # Store the minimum anomaly score.

        'mean_score': round(scores.mean(), 4)
        # Store the average anomaly score.
        }

        self._if_scores = scores
        # Store raw Isolation Forest scores for possible later inspection.

        return self
    
    # ── Local Outlier Factor (density-based) ──────────────────────────────
    def detect_lof(self, cols: list, n_neighbors: int = 20,
                   contamination: float = 0.05):
        # Define a multivariate outlier detector based on Local Outlier Factor.
            
        valid_cols = [c for c in cols if c in self.df.columns
                    and pd.api.types.is_numeric_dtype(self.df[c])]
            
        X = self.df[valid_cols].fillna(self.df[valid_cols].median())
        # Build the feature matrix and fill missing values with median values.

        X_scaled = StandardScaler().fit_transform(X)
        # Standardize features before applying LOF.

        model  = LocalOutlierFactor(n_neighbors=n_neighbors,
                                    contamination=contamination)
            
        preds  = model.fit_predict(X_scaled)
        # Fit the LOF model and predict labels:
        # -1 means outlier, 1 means inlier.

        flag   = pd.Series(preds == -1, index=self.df.index)
        # Convert predictions into a boolean outlier flag Series.

        self.flags['lof'] = flag
        # Save row-level LOF flags.

        self.results['lof'] = {
        'total_flagged': int(flag.sum()),
        # Store total number of flagged rows.

        'pct': round(flag.mean()*100, 2),
        # Store percentage of flagged rows.

        'cols_used': valid_cols,
        # Store which columns were used for LOF.
        }
        return self

    # ── Consensus flag (flagged by ≥ k methods) ───────────────────────────
    def consensus(self, min_methods: int = 2) -> pd.Series:
        count = self.flags.sum(axis=1)
        # Count how many methods flagged each row.

        return count >= min_methods
        # Return a boolean Series marking rows flagged by at least min_methods methods.

    def scorecard(self) -> pd.DataFrame:
        # Define a method that summarizes outlier detection results across methods.
        rows = []
        for method, res in self.results.items():
            rows.append({
                'Method': method.replace('_',' ').title(),
                'Flagged': res['total_flagged'],
                'Pct (%)': res['pct'],
                'Type': 'Multivariate' if method in ('isolation_forest','lof') else 'Univariate'
                })

        df = pd.DataFrame(rows)
        consensus = self.consensus()
        df.loc[len(df)] = {'Method': 'CONSENSUS (≥2 methods)',
                           'Flagged': int(consensus.sum()),
                           'Pct (%)': round(consensus.mean()*100, 2),
                           'Type': 'Combined'}
        return df

print('✅ OutlierDetector class defined')
    



if __name__ == "__main__":
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
    print("Colonne disponibili in Seller:", seller_df.columns.tolist())
    seller_df = seller_df.drop(columns=['seller_zip_code_prefix'])
    category_df = pd.read_csv(os.path.join(path, "product_category_name_translation.csv"))
    category_df = category_df.drop(columns=['product_category_name_english'])

    BASE_PATH = r'C:\Users\acer\Desktop\Magistrale 1-02-2026\I anno\Data Ware House\Olist\Cleaning\Cleaning1\\'
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        print(f"Directory created: {BASE_PATH}")

    def save_missing_summary(df, table_name, base_path):
        missing_summary_df = df.isnull().sum()
        missing_summary_df = missing_summary_df[missing_summary_df > 0]
        if missing_summary_df.empty:
            print(f"No missing values were found for {table_name}.")
            return None
        missing_summary_df = missing_summary_df.reset_index()
        missing_summary_df.columns = ['column', 'missing_count']
        missing_summary_df['missing_pct'] = (missing_summary_df['missing_count'] / len(df) * 100).round(2)
        output_path = f"{base_path}{table_name}_missing_summary.csv"
        missing_summary_df.to_csv(output_path, index=False)
        print(f"✔ Missing summary saved: {output_path}")
        return missing_summary_df

    save_missing_summary(order_df, "Order", BASE_PATH)
    save_missing_summary(order_item_df, "OrderItem", BASE_PATH)
    save_missing_summary(product_df, "Product", BASE_PATH)
    save_missing_summary(customer_df, "Customer", BASE_PATH)
    save_missing_summary(seller_df, "Seller", BASE_PATH)
    save_missing_summary(category_df, "ProductCategoryName", BASE_PATH)

    # In a symmetric (normal) distribution, the mean and median are identical.
    # In a positively skewed distribution,
    # the mean is "pulled" upward by luxury products, resulting in it being much higher than the median.
    # Selecting columns of interest
    cols_to_verify = ['price', 'freight_value']

    print("📊 ASYMMETRY TEST (Skewness Analysis):")
    print("-" * 40)

    for col in cols_to_verify:
        
        stats_data = order_item_df[col].agg(['mean', 'median', 'skew'])
        
        mean_val = stats_data['mean']
        median_val = stats_data['median']
        skew_val = stats_data['skew']
        
        print(f"\nAnalisi per l'attributo: {col.upper()}")
        print(f"   • Media   : {mean_val:.2f}")
        print(f"   • Mediana : {median_val:.2f}")
        print(f"   • Skewness: {skew_val:.2f}")
        
        # Verifica del trascinamento della media (Right Skewness)
        if mean_val > median_val:
            ratio = mean_val / median_val
            print(f"   The mean is {ratio:.2f} times greater than the median.")
            print(f"   The distribution is positively skewed.")
        
        # Interpretation of the Skewness Index
        if skew_val > 1:
            print(f"   Skewness > 1 ({skew_val:.2f}). The data is highly skewed.")
        elif skew_val > 0:
            print(f"   Positive skewness. The tail is skewed toward the higher values.")        


        maOrder = MissingnessAnalyzer(order_df, 'order_ma')
        print(' Missingness Summary:')
        print(maOrder.summary().to_string())
        print()



    # Is order_approved_at missingness independent of order_status ? 
    chi2_result_order_approved = maOrder.test_mcar_chi2('order_approved_at', 'order_status')
    # Is order_delivered_carrier_date missingness independent of order_status ? 
    chi2_result_carrier_date = maOrder.test_mcar_chi2('order_delivered_carrier_date', 'order_status')
    # Is order_delivered_customer_date missingness independent of order_status ? 
    chi2_result_customer_date = maOrder.test_mcar_chi2('order_delivered_customer_date', 'order_status')


    print(f"\n🔬 Chi-² MCAR test — approved_at vs order_status:")
    for k, v in chi2_result_order_approved.items():
        print(f"   {k:<15}: {v}")
    contingency_table1 = pd.crosstab(order_df['order_status'], order_df['order_approved_at'].isna())
    print(contingency_table1)

    print(f"\n🔬 Chi-² MCAR test — order_delivered_carrier_date vs order_status:")
    for k, v in chi2_result_carrier_date.items():
        print(f"   {k:<15}: {v}")
    contingency_table2 = pd.crosstab(order_df['order_status'], order_df['order_delivered_carrier_date'].isna())
    print(contingency_table2)

    print(f"\n🔬 Chi-² MCAR test — order_delivered_customer_date vs order_status:")
    for k, v in chi2_result_customer_date.items():
        print(f"   {k:<15}: {v}")
    contingency_table1 = pd.crosstab(order_df['order_status'], order_df['order_delivered_customer_date'].isna())
    print(contingency_table1)

    # Let's convert ‘order_status’ into numerical columns (0 and 1)
    status_dummies = pd.get_dummies(order_df['order_status'], prefix='status', drop_first=True)
    dummy_cols = status_dummies.columns.tolist()
    # Let's add these new columns to the original DataFrame for analysis
    order_df_with_dummies = pd.concat([order_df, status_dummies], axis=1)
    # Let's create a new analyzer on this “enriched” DataFrame
    maOrderExtended = MissingnessAnalyzer(order_df_with_dummies, 'order_extended')
    # Let's run the test using dummy variables as predictors
    log_result_approved = maOrderExtended.test_mar_logistic('order_approved_at', dummy_cols)
    log_result_carrier = maOrderExtended.test_mar_logistic('order_delivered_carrier_date', dummy_cols)
    log_result_customer = maOrderExtended.test_mar_logistic('order_delivered_customer_date', dummy_cols)

    print(f"\n🔬 Logistic MAR test — order_approved_at:")
    for k, v in log_result_approved.items():
        print(f"   {k:<15}: {v}")

    print(f"\n🔬 Logistic MAR test — order_delivered_carrier_date:")
    for k, v in log_result_carrier.items():
        print(f"   {k:<15}: {v}")

    print(f"\n🔬 Logistic MAR test — order_delivered_customer_date:")
    for k, v in log_result_customer.items():
        print(f"   {k:<15}: {v}")


    product_with_seller = pd.merge(
        product_df, 
        order_item_df[['product_id', 'seller_id']], 
        on='product_id', 
        how='left'
    )
    product_with_seller = product_with_seller.drop_duplicates(subset=['product_id', 'seller_id'])



    # Plots: Missing % by order_status (MAR signal) 
    # Identifying the three columns with missing values in the Order table
    order_missing_cols = [
        'order_approved_at', 
        'order_delivered_carrier_date', 
        'order_delivered_customer_date'
    ]


    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.suptitle('Missing % by order_status (MAR signal)', fontsize=16, fontweight='bold')

    for i, col in enumerate(order_missing_cols):
        status_miss = (
            order_df.groupby('order_status')[col]
            .apply(lambda x: x.isna().mean() * 100)
            .sort_values(ascending=True)
        )
        
        
        status_miss.plot(
            kind='barh', 
            ax=axes[i], 
            color=['#e74c3c' if v > 10 else '#3498db' for v in status_miss]
        )
        
        
        overall_mean = order_df[col].isna().mean() * 100
        axes[i].axvline(overall_mean, color='black', linestyle='--', alpha=0.6, label='Overall mean')
        
        axes[i].set_title(f'Missing %: {col}', fontweight='bold')
        axes[i].set_xlabel('% missing')
        axes[i].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(BASE_PATH, "orders_missing_by_status.png"), dpi=300)
    print(f"✔ Plot saved: {os.path.join(BASE_PATH, 'orders_missing_by_status.png')}")
    plt.show()

    # Logistic Regression

    product_geo = pd.merge(product_with_seller, seller_df[['seller_id', 'seller_state']], on='seller_id')

    # Creating dummies for states
    state_dummies = pd.get_dummies(product_geo['seller_state'], prefix='state', drop_first=True)
    dummy_cols = state_dummies.columns.tolist()
    product_geo_logistic = pd.concat([product_geo, state_dummies], axis=1)

    # Calculating economic averages (price and freight_value) from sales
    avg_shipping_data = order_item_df.groupby('product_id')[['price', 'freight_value']].mean().reset_index()


    full_prod_data = pd.merge(product_geo_logistic, avg_shipping_data, on='product_id', how='left')


    full_prod_data['log_price'] = np.log1p(full_prod_data['price'])
    full_prod_data['log_freight'] = np.log1p(full_prod_data['freight_value'])


    # Definition of the complete list of predictors
    new_predictors = ['log_price', 'log_freight'] + dummy_cols 

    maEnhanced = MissingnessAnalyzer(full_prod_data, 'Enhanced_Full_Analysis')
    res = maEnhanced.test_mar_logistic('product_category_name', new_predictors)

    print(f"\n🔬 Logistic MAR test — product_category_name (All Attributes):")
    for k, v in res.items():
        if k != 'predictors':  
            print(f"   {k:<15}: {v}")
        else:
            
            dummies = [p for p in v if p.startswith('state_')]
            econ = [p for p in v if p in ['log_price', 'log_freight']]
            print(f"   {k:<15}: {len(v)} total variables")
            print(f"     └─ Economiche (Price/Freight): {len(econ)}")
            print(f"     └─ Geografiche (States)      : {len(dummies)}")

    # ── MNAR proxy — does product_category_name missingness correlate with values? ──
    mnar_result = maEnhanced.test_mnar_correlation('product_category_name', new_predictors)
    print(f"\n🔬 MNAR proxy test — product_category_name:")
    print(f" verdict: {mnar_result['verdict']}")
    # Print overall MNAR interpretation
    for col, v in mnar_result['comparisons'].items():
        sig = '⚠️ ' if v['significant'] else ' '
        # Highlight statistically significant differences
        print(f" {sig} {col:<18}: mean_present={v['mean_present']}, "
            f"mean_missing={v['mean_missing']}, p={v['p_value']}")


    # OUTLIER DETECTION

    cols_transazionali = ['price', 'freight_value']

    od_transazioni = ( 
        OutlierDetector(order_item_df, 'Olist_Transactions')
        .detect_iqr(cols_transazionali, multiplier=3.0) 
        .detect_modified_zscore(cols_transazionali, threshold=3.5) 
        .detect_isolation_forest(cols_transazionali, contamination=0.02)
    )

    print("\n📊 Outlier Detection Scorecard - Olist Transactions:")
    print(od_transazioni.scorecard().to_string(index=False))
    print('\n🔎 IQR Details per column:')
    for col, det in od_transazioni.results['iqr']['details'].items():
        if det['outlier_count'] > 0:
            print(f" {col:<18}: fence=[{det['lower_fence']}, {det['upper_fence']}], "
                f"outliers={det['outlier_count']}")
            

    consensus_outliers = od_transazioni.consensus(min_methods=2)
    print(f"\n✅ Anomalies identified by consensus: {consensus_outliers.sum()}")

    top_outliers = order_item_df[consensus_outliers].sort_values(by='price', ascending=False).head(10)
    print(top_outliers[['order_id', 'product_id', 'price', 'freight_value']])