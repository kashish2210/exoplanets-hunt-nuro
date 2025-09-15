# A World Away: Hunting for Exoplanets with AI  

This repository provides an open-source framework for **automated exoplanet detection** using data from space-based missions such as **Kepler, K2, and TESS**.  
The system applies **machine learning and deep learning methods** to stellar light curves in order to identify periodic dimming events (transits) that may indicate the presence of exoplanets.  

---

## Problem Statement  
Traditional exoplanet discovery has relied on **manual inspection of light curve data**. With the increasing scale of modern surveys producing **billions of photometric measurements**, manual approaches are no longer feasible. A scalable, automated solution is required to ensure timely and accurate identification of new worlds.  

---

## Objectives  
- Build a **scalable preprocessing pipeline** for light curve data.  
- Train and evaluate **deep learning models** (CNNs, RNNs, LSTMs) for transit detection.  
- Provide a **benchmarking framework** with established metrics.  
- Ensure **reproducibility** through an open-source, extensible implementation.  

---

## Key Features  
- **Multi-mission support**: Kepler, K2, TESS (extendable to JWST and beyond).  
- **Automated preprocessing**: normalization, detrending, noise filtering, outlier removal.  
- **ML/DL-based classification**: convolutional and recurrent neural networks for light curve analysis.  
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC.  
- **Visualization tools**: light curve plots with transit overlays, prediction confidence scores.  
- **Research extensibility**: modular structure for new datasets and models.  

---

## Technology Stack  
- **Programming**: Python 3.10+  
- **Data Processing**: NumPy, Pandas, SciPy  
- **Machine Learning**: TensorFlow / PyTorch, Scikit-learn  
- **Visualization**: Matplotlib, Seaborn, Plotly  
- **Optional UI**: Streamlit / Dash for interactive exploration  

---

## Datasets  
The following datasets are supported:  
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)  
- [Kepler Labeled Time Series Data (Kaggle)](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)  
- [TESS Mission Data (MAST)](https://archive.stsci.edu/tess/)  

---

## Installation  

```bash
# Clone the repository
git clone https://github.com/your-username/exoplanet-ai.git
cd exoplanet-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```
## References & Resources  

### Exoplanet Data Sources  
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)  
- [Kepler/K2 Mission Data](https://archive.stsci.edu/kepler/)  
- [TESS Mission Data](https://archive.stsci.edu/tess/)  
- [ExoFOP (Exoplanet Follow-up Observing Program)](https://exofop.ipac.caltech.edu/)  

### Machine Learning & Astronomy  
- [NASA Frontier Development Lab](https://frontierdevelopmentlab.org/)  
- [Google AI Exoplanet ML](https://ai.googleblog.com/2018/12/machine-learning-helps-discover-new.html)  
- [Lightkurve (Python package)](https://docs.lightkurve.org/)  
- [Astropy](https://www.astropy.org/)  

### deployment & Infrastructure  
- [Google Cloud AI/ML Platform](https://cloud.google.com/ai-platform)  
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)  
- [Microsoft Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/)  

### Visualization Tools  
- [Plotly Dash](https://plotly.com/dash/)  
- [Bokeh](https://bokeh.org/)  
- [Grafana](https://grafana.com/)  

### Research Papers  
- Shallue & Vanderburg (2018): *Identifying Exoplanets with Deep Learning*  
  [https://www.pnas.org/doi/10.1073/pnas.1717066115](https://www.pnas.org/doi/10.1073/pnas.1717066115)  
- Armstrong et al. (2021): *Exoplanet detection with deep learning from TESS*  
  [https://arxiv.org/abs/2101.11356](https://arxiv.org/abs/2101.11356)  
- Osborn et al. (2020): *AI for Exoplanet Transit Detection*  
  [https://arxiv.org/abs/2001.03460](https://arxiv.org/abs/2001.03460)  
