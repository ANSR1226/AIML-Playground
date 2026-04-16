# Customer Segmentation with K-Means

This project performs **customer segmentation** using **K-Means clustering** on transactional data.  
We engineer **RFM-style features** (Recency, Frequency, Monetary), scale them, train a **K-Means model**, and evaluate the clustering quality using **silhouette scores**. Finally, we visualize clusters using **PCA**.

---

## 1. Project Overview

The goal is to group customers into a small number of **behaviourally similar segments** so that we can:

- Identify **high-value / VIP** customers  
- Detect **inactive / churn-risk** customers  
- Understand overall **spend and visit patterns**

We use:

- **RecencyDays** – how many days since the customer’s last purchase  
- **Frequency** – number of purchases per customer  
- **TotalAmount** – total monetary value spent by the customer  

These features are used as input to **K-Means**.

---

## 2. Data & Feature Engineering

Assume we start from a transaction-level DataFrame: `df` with at least:

- `CustomerID`
- `InvoiceDate` (datetime)
- `Amount` (order value)

### 2.1 Aggregate to customer level

```python
# Group by customer to build RFM-like features
customer_df = df.groupby("CustomerID").agg(
    TotalAmount=("Amount", "sum"),
    Frequency=("InvoiceNo", "nunique"),  # or "count" depending on data
    LastPurchase=("InvoiceDate", "max")
).reset_index()
```

### 2.2 Compute Recency (in days)

```python
reference_date = customer_df["LastPurchase"].max()
customer_df["RecencyDays"] = (reference_date - customer_df["LastPurchase"]).dt.days
```

- `reference_date`: most recent purchase date in the dataset (used as "today").
- `RecencyDays`: number of days since each customer's last purchase  
  (smaller = more recent = more active).

---

## 3. Preparing the Feature Matrix

We build a feature matrix **`X`** from numeric behaviour features:

```python
X = customer_df[["TotalAmount", "Frequency", "RecencyDays"]]
```

Because K-Means is distance-based, features are **standardized** so they contribute fairly:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

- `X`: original features in their natural units  
- `X_scaled`: standardized features (mean ≈ 0, std_deviation ≈ 1)

---

## 4. Training the K-Means Model

We fit **K-Means** on the scaled features:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=23)  # k=3 chosen via silhouette
kmeans.fit(X_scaled)

# Attach cluster labels back to customer_df
customer_df["cluster"] = kmeans.labels_
```

- `n_clusters`: number of segments (here 3, see Section 6)  
- `random_state`: makes results reproducible  
- `customer_df["cluster"]`: cluster assignment for each customer

---

## 5. Cluster Visualization (PCA)

To visualize high-dimensional data, we reduce **3D → 2D** using **PCA**, then plot clusters:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

customer_df["pc1"] = X_pca[:, 0]
customer_df["pc2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
for c in sorted(customer_df["cluster"].unique()):
    subset = customer_df[customer_df["cluster"] == c]
    plt.scatter(subset["pc1"], subset["pc2"], alpha=0.6, label=f"Cluster {c}")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Customer Segments (K-Means + PCA)")
plt.legend()
plt.show()
```

Notes:

- PCA is **only for visualization**; K-Means itself runs on the **full scaled feature space**.
- The 2D PC1–PC2 plot provides intuition on how clusters are separated.

---

## 6. Evaluating Cluster Quality (Silhouette Score)

Because this is **unsupervised**, there is no accuracy in the supervised sense.  
We use **silhouette score** to measure cluster quality:

```python
from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X_scaled, kmeans.labels_)
print("Silhouette score:", sil_score)
```

To choose a good `k`, we compute silhouette scores for different values:

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=23)
    km.fit(X_scaled)
    score = silhouette_score(X_scaled, km.labels_)
    print(f"k: {k}, silhouette_score: {score}")
```

Example results:

- k=2 → 0.5455  
- **k=3 → 0.5608** (highest)  
- k=4 → 0.5237  
- k=5 → 0.5221  
- k=6 → 0.5232  
- k=7 → 0.5275  
- k=8 → 0.5300  
- k=9 → 0.5288  

We choose **k=3** because it has the **best average silhouette score** and remains interpretable (3 segments).

---

## 7. Interpreting the Segments

For interpretation, we inspect the **mean RFM features per cluster**:

```python
cluster_profile = customer_df.groupby("cluster")[["TotalAmount", "Frequency", "RecencyDays"]].mean()
print(cluster_profile)
```

This allows us to label segments such as:

- **Cluster 0** – low spend, low frequency, high recency → dormant / churn-risk  
- **Cluster 1** – medium spend and frequency → regular customers  
- **Cluster 2** – high spend, high frequency, low recency → VIP / loyal customers  

(Exact descriptions depend on the dataset; adjust based on `cluster_profile`.)

---

## 8. How to Run

1. Install dependencies (example):

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. Load your raw transactional data into `df`.

3. Run the notebook / script that:
   - Aggregates to `customer_df`
   - Computes `RecencyDays`
   - Builds `X`, `X_scaled`
   - Trains K-Means
   - Computes silhouette scores
   - Plots PCA clusters
   - Prints cluster profiles

4. Adjust:
   - The set of features
   - `n_clusters`
   - Any preprocessing (e.g., log transforms) based on your data and business needs.

---

## 9. Next Steps / Improvements

- Add **log transforms** for highly skewed features (e.g. `log1p(TotalAmount)`).
- Try other clustering algorithms (e.g., **Gaussian Mixture Models**, **DBSCAN**).
- Use more features (e.g., average order value, time since first purchase).
- Integrate the segments into marketing or recommendation systems.

---