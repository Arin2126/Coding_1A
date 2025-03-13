import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def main():
    """
    Question 5:
    -----------
    Compare Linear and Polynomial Regression (degrees 2, 3, 4) on the 
    Concrete Compressive Strength dataset provided in an Excel (.xls) file.
    
    The dataset (Concrete_Data.xls) should be extracted from the provided zip 
    folder and saved in the same folder as this script.
    
    We use only one feature ("Cement (component 1)(kg in a m^3 mixture)") for 
    simplicity and visualize the regression curves over a scatter plot of the test data.
    """
    # ---------------------------------------------------
    # 1. Load the Dataset from the Excel File
    # ---------------------------------------------------
    try:
        df = pd.read_excel('Concrete_Data.xls')
    except FileNotFoundError:
        print("Error: 'Concrete_Data.xls' not found. Make sure the file is in the same folder as this script.")
        return

    # Remove any leading/trailing spaces from the column names
    df.columns = df.columns.str.strip()
    print("Columns in the Excel file:", df.columns)

    # ---------------------------------------------------
    # 2. Extract the Feature and Target Columns
    # ---------------------------------------------------
    # Based on your output, the columns are:
    #   Feature: "Cement (component 1)(kg in a m^3 mixture)"
    #   Target:  "Concrete compressive strength(MPa, megapascals)"
    try:
        X = df[['Cement (component 1)(kg in a m^3 mixture)']].values
        y = df['Concrete compressive strength(MPa, megapascals)'].values
    except KeyError:
        print("Error: Expected columns not found. Please verify the column names in the Excel file.")
        return

    # ---------------------------------------------------
    # 3. Split the Data into Training and Testing Sets (80/20 split)
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ---------------------------------------------------
    # 4. Linear Regression
    # ---------------------------------------------------
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    print("=== Linear Regression ===")
    print(f"MSE (Linear): {mse_linear:.4f}")
    print(f"R^2  (Linear): {r2_linear:.4f}")

    # ---------------------------------------------------
    # 5. Polynomial Regression (Degrees 2, 3, 4)
    # ---------------------------------------------------
    degrees = [2, 3, 4]
    poly_models = {}

    for deg in degrees:
        poly = PolynomialFeatures(degree=deg)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model_poly = LinearRegression()
        model_poly.fit(X_train_poly, y_train)
        y_pred_poly = model_poly.predict(X_test_poly)

        mse_poly = mean_squared_error(y_test, y_pred_poly)
        r2_poly = r2_score(y_test, y_pred_poly)
        poly_models[deg] = model_poly

        print(f"\n=== Polynomial Regression (Degree = {deg}) ===")
        print(f"MSE (Degree {deg}): {mse_poly:.4f}")
        print(f"R^2  (Degree {deg}): {r2_poly:.4f}")

    # ---------------------------------------------------
    # 6. Visualize the Results
    # ---------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual Data')

    # For a smooth line, sort the test data by the "Cement" feature
    sort_idx = np.argsort(X_test.ravel())
    X_test_sorted = X_test[sort_idx]
    
    # Plot the linear regression line
    y_pred_linear_sorted = linear_model.predict(X_test_sorted)
    plt.plot(X_test_sorted, y_pred_linear_sorted, color='red', label='Linear Regression')

    # Plot polynomial regression curves for each degree
    colors = ['green', 'orange', 'purple']
    for i, deg in enumerate(degrees):
        poly = PolynomialFeatures(degree=deg)
        X_test_poly_sorted = poly.fit_transform(X_test_sorted)
        y_pred_poly_sorted = poly_models[deg].predict(X_test_poly_sorted)
        plt.plot(X_test_sorted, y_pred_poly_sorted, color=colors[i],
                 label=f'Polynomial (deg={deg})')

    plt.xlabel('Cement (kg in a m^3 mixture)')
    plt.ylabel('Concrete Compressive Strength (MPa)')
    plt.title('Linear vs. Polynomial Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
