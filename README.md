# **Mitigation of Over-Squashing in Graph Neural Networks**  

## **Overview**  
Over-squashing is a well-known issue in Graph Neural Networks (GNNs), where long-range dependencies struggle to propagate effectively due to bottlenecks in the graph structure. This project explores techniques to mitigate over-squashing, focusing on **graph rewiring** methods. Specifically, we compare the **State-of-the-Art SDRF method** with our proposed **Commute Time-based SDRF (CT-SDRF)**, evaluating their performance on benchmark datasets **QM9 and ZINC**.  

## **Methods**  
We employ several experimental setups to analyze different factors affecting over-squashing, including **impact of width, depth, and fully connected graph **. Our approach involves **graph rewiring** techniques to improve connectivity and reduce information bottlenecks while preserving key structural properties.  

## **Experiments**  
The experiments are organized into the following files:  

- **`width_comparison.py`** – Ablation study on the impact of **network width** in mitigating over-squashing.  
- **`depth_comparison.py`** – Ablation study on the impact of **network depth** in mitigating over-squashing.  
- **`fully_connected.py`** – Ablation study on **fully connected nodes** and their effect on graph representation.  
- **`qm9.py`** – Main experiment comparing **SDRF** and our proposed **CT-SDRF** method on the **QM9 dataset**.  

## **Installation**  
To set up the project, follow these steps:  

1. Clone the repository:  
   ```bash
   git clone https://github.com/nasibhuseynzade/Over-squashing-in-GNNs.git
   cd your-repo-name
   ```  
2. Create a virtual environment and activate it:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

## **Usage**  
Run individual experiments using:  
```bash
python experiments/qm9.py  
```  

## **Results**  
Our findings demonstrate that **CT-SDRF** effectively enhances graph connectivity while preserving key structural features, leading to improved model performance compared to standard SDRF. 

