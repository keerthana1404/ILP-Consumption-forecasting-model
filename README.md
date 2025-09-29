-> ILP (Interleaving Paper) Consumption Forecasting
-->Project Overview

Website Link -https://ilp-consumption-forecasting-model.streamlit.app/ 

This project focuses on predicting the consumption of Fresh Interleaving Paper (FILP) used in the stainless steel production process at Jindal Stainless Limited (JSL). FILP is a critical consumable placed between stainless steel sheets to prevent scratches during rolling and dispatch.

Accurate forecasting helps achieve:
📉 Reduced Inventory Costs – Just-in-time procurement
📦 Optimized Storage Space
⏳ Better Production Planning
✅ Improved Supply Chain Efficiency

-->Example Workflow
-Upload monthly CSV files.
-Preprocessing step computes FILP ILP Tonnage Weight.
-Select forecasting model (XGBoost).
View next 7–30 days FILP demand forecast with confidence intervals.

-->Results
-Achieved RMSE ~0.53 MT with XGBoost (short-term forecast).

-->Tech Stack
Languages: Python (3.10+)
Libraries: Pandas, Numpy, Scikit-learn, XGBoost, TensorFlow/Keras, Matplotlib
Dashboard: Streamlit

