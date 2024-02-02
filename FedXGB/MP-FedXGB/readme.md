# STV 

This repository contains the code for a forecasting framework with *vertical federated learning*.  The methods cover a linear autoregressive forecaster, SARIMAX, and an autoregressive tree-based forecaster: https://github.com/JoaquinAmatRodrigo/skforecast


# Relevant Files
 
1. Secret-shared matrix operations and communication experiments: ```ssmatrix.py```
2. Forecasting tests: ```forecast_<DATASET>.py```
3. Secret-sharing primitives: ```SSCalculation.py``` and ```SSCalculate_Alternate.py```
4. VFL XGBoost code based on **MP-FedXGB**: ```VerticalXGBoost.py```, with documentation: https://github.com/HikariX/MP-FedXGB
5. Original diffusion model for time series repository: https://github.com/AI4HealthUOL/SSSD. Additional results and minor extensions in: https://github.com/adis98/SSSD/tree/master

# Datasets
1. Rossman Sales: https://www.kaggle.com/c/rossmann-store-sales/data
2. Air Quality: https://archive.ics.uci.edu/dataset/360/air+quality
3. Flight Passengers: https://www.kaggle.com/datasets/chirag19/air-passengers
4. SML 2010: https://archive.ics.uci.edu/dataset/274/sml2010
5. PV Power: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
