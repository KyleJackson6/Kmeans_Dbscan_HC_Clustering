# Clustering API with FastAPI

This project is a FastAPI application that allows you to upload a dataset, explore basic statistics, and perform clustering using three popular algorithms: **KMeans**, **DBSCAN**, and **Hierarchical Clustering**.

## Features

- Upload `.csv` or `.xlsx` datasets.
- Compute basic statistics about the dataset.
- Run clustering with customizable hyperparameters.
- Visualize data before and after clustering.
- Automatically save clustering results in folders.


### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install dependencies

Make sure you have Python 3.8+ installed. Then install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI server

```bash
uvicorn main:app --reload
```

This will start the server at `http://127.0.0.1:8000`.

## 📫 API Endpoints

### Upload Data

**POST /upload-data**  
Upload a `.csv` or `.xlsx` file. (mall_customer_data.csv)


### View Statistics

**GET /statistics**  
Returns shape, data types, and basic stats.


### Perform Clustering

**POST /cluster**  
Provide the algorithm and hyperparameters in JSON format.

#### KMeans Example

{
    "algorithm": "kmeans",
    "params": {
      "n_clusters": 3,
      "init": "k-means++",
      "random_state": 42
    }
  }

#### DBSCAN Example

{
    "algorithm": "dbscan",
    "params": {
      "eps": 5,
      "min_samples": 3
    }
  }

#### Hierarchical Example

{
    "algorithm": "hierarchical",
    "params": {
      "n_clusters": 4,
      "linkage": "ward"
    }
  }

## 📁 Output

Clustering visualizations will be saved in the `Visulization (Algorithms)` folder.

- `before_clustering`
- `after_clustering`

---
