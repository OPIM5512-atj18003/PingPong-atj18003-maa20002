# A02 Ping Pong — California Housing MLP Regression

This project was completed as part of a collaborative “Ping Pong” GitHub workflow assignment. We trained a neural network regression model on the California Housing dataset and evaluated its performance using predicted vs actual plots for both training and test data.

## Partners

- Mohamad Ali Hamadi (maa20002)  
- Alex Jones (atj18003)

Hi everyone this is Alex. Welcome to our repo!

Hey everyone this is Mo. Nice to meet you! 

## How to Run:

From the repository root, run:

```bash
pip install -r requirements.txt
python3 src/ds_pipeline.py
```

## Output

After running the script, the following files will be saved in the figures/ directory:

* train_actual_vs_pred.png — Predicted vs Actual values for the training set
* test_actual_vs_pred.png — Predicted vs Actual values for the test set

These plots include a red reference line (y = x) to visually assess model performance.

### Expected Output:

```bash
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23        4.526
1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22        3.585
2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24        3.521
3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25        3.413
4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25        3.422
(20640, 9)
X_train: (16512, 8) X_test: (4128, 8)
y_train: (16512,) y_test: (4128,)
Scaled X_train: (16512, 8)
Scaled X_test: (4128, 8)
MLP training complete.
Train score (R^2): 0.7963554783358725
Test score (R^2): 0.7743853554640314
```
**Train score (R^2): 0.7963554783358725**

**Test score (R^2): 0.7743853554640314**