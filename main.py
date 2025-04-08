import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import uvicorn
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.decomposition import PCA

app = FastAPI()
data = None

# Base folder to save clustering results
BASE_FOLDER = "Visulization (Algorithms)"

# Folders for different clustering algorithms
CLUSTER_FOLDERS = {
    "kmeans": os.path.join(BASE_FOLDER, "Kmeans"),
    "dbscan": os.path.join(BASE_FOLDER, "DBscan"),
    "hierarchical": os.path.join(BASE_FOLDER, "Hierarchical Clustering"),
}

# Ensure folders exist
for folder in CLUSTER_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

class ClusteringParams(BaseModel):
    algorithm: str
    params: dict

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    global data
    try:
        if file.filename.endswith(".csv"):
            content = await file.read()
            data = pd.read_csv(BytesIO(content))
        elif file.filename.endswith(".xlsx"):
            content = await file.read()
            data = pd.read_excel(BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        rows, cols = data.shape
        return {"status": "success", "rows": rows, "columns": cols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
def get_statistics():
    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded.")
    stats = {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "data_types": data.dtypes.astype(str).to_dict(),
        "numeric_stats": data.describe().to_dict()
    }
    return stats

@app.post("/cluster")
async def cluster_data(params: ClusteringParams):
    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded.")
    
    try:
        # Selecting only numerical data
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise HTTPException(status_code=400, detail="No numeric columns available.")

        # Scaling the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        # Dimensionality reduction for visualization (2D)
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        # Visualizing the data before clustering (in 2D)
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='gray', marker='o', edgecolor='k', s=50)
        plt.title('Data Before Clustering')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        # Save to appropriate folder
        before_folder = os.path.join(CLUSTER_FOLDERS[params.algorithm.lower()], "before_clustering")
        os.makedirs(before_folder, exist_ok=True)
        before_plot_path = os.path.join(before_folder, "data_before_clustering.png")
        plt.savefig(before_plot_path)
        plt.close()

        # Initialize the model based on the selected algorithm
        if params.algorithm == "kmeans":
            model = KMeans(**params.params)
        elif params.algorithm == "dbscan":
            model = DBSCAN(**params.params)
        elif params.algorithm == "hierarchical":
            model = AgglomerativeClustering(**params.params)
        else:
            raise HTTPException(status_code=400, detail="Unsupported algorithm.")

        # Fitting the model
        labels = model.fit_predict(scaled_data)
        data["cluster_label"] = labels

        # Visualizing the data after clustering (in 2D)
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
        plt.title(f'Data After {params.algorithm.capitalize()} Clustering')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(label='Cluster Label')

        # Save to appropriate folder
        after_folder = os.path.join(CLUSTER_FOLDERS[params.algorithm.lower()], "after_clustering")
        os.makedirs(after_folder, exist_ok=True)
        after_plot_path = os.path.join(after_folder, f"{params.algorithm.lower()}_clustering_result.png")
        plt.savefig(after_plot_path)
        plt.close()

        # Prepare response
        result = {
            "labels": labels.tolist(),
            "sample": data.head(10).to_dict(orient="records"),
            "before_clustering_plot": before_plot_path,
            "after_clustering_plot": after_plot_path
        }

        if params.algorithm == "kmeans":
            result["cluster_centers"] = model.cluster_centers_.tolist()
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
