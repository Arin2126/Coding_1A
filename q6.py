import numpy as np
import matplotlib.pyplot as plt

# Define a function to compute the normal PDF
def pdf_normal(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def main():
    # -------------------------------
    # Part (a): Data Simulation
    # -------------------------------
    # Simulate 1000 samples from N(50, 10)
    np.random.seed(42)
    mu_true = 50
    sigma_true = 10
    n_samples = 1000
    data = np.random.normal(loc=mu_true, scale=sigma_true, size=n_samples)
    
    # Plot histogram of the simulated data
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    plt.title('Histogram of Simulated Data (Normal Distribution)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    
    # -------------------------------
    # Part (b): Normal Distribution Fitting (Without Outliers)
    # -------------------------------
    # Fit a normal distribution using MLE:
    # For a normal distribution, the MLE for the mean is the sample mean and for the
    # standard deviation is the population standard deviation (ddof=0).
    mu_est = np.mean(data)
    sigma_est = np.std(data, ddof=0)
    
    print("Fitted Normal Distribution (Without Outliers):")
    print(f"Estimated Mean: {mu_est:.4f}")
    print(f"Estimated Std:  {sigma_est:.4f}")
    
    # Create a range of x values for plotting the PDF
    x_range = np.linspace(min(data) - 10, max(data) + 10, 500)
    pdf_values = pdf_normal(x_range, mu_est, sigma_est)
    
    # Plot histogram with the fitted PDF overlay
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Histogram')
    plt.plot(x_range, pdf_values, 'r-', lw=2, label='Fitted Normal PDF')
    plt.title('Histogram and Fitted Normal PDF (Without Outliers)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # -------------------------------
    # Part (c): Handling Outliers
    # -------------------------------
    # Simulate 50 samples from a uniform distribution over [100, 150]
    outliers = np.random.uniform(100, 150, 50)
    # Add these outliers to the original dataset
    data_with_outliers = np.concatenate([data, outliers])
    
    # Fit a normal distribution to the new dataset (with outliers)
    mu_est_out = np.mean(data_with_outliers)
    sigma_est_out = np.std(data_with_outliers, ddof=0)
    
    print("\nFitted Normal Distribution (With Outliers):")
    print(f"Estimated Mean: {mu_est_out:.4f}")
    print(f"Estimated Std:  {sigma_est_out:.4f}")
    
    # Create a range of x values for the new dataset
    x_range_out = np.linspace(min(data_with_outliers) - 10, max(data_with_outliers) + 10, 500)
    pdf_values_out = pdf_normal(x_range_out, mu_est_out, sigma_est_out)
    
    # Plot histogram with the fitted PDF overlay for data with outliers
    plt.figure(figsize=(10, 6))
    plt.hist(data_with_outliers, bins=30, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label='Histogram with Outliers')
    plt.plot(x_range_out, pdf_values_out, 'b-', lw=2, label='Fitted Normal PDF (With Outliers)')
    plt.title('Histogram and Fitted Normal PDF (With Outliers)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # -------------------------------
    # Discussion
    # -------------------------------
    print("\nDiscussion:")
    print("Without outliers, the estimated mean and standard deviation are close to the true values (50 and 10).")
    print("When outliers (from a uniform distribution between 100 and 150) are added, the estimated mean and standard deviation both increase,")
    print("indicating that the outliers significantly affect the parameter estimates.")
    print("\nOne approach to detect outliers is the Interquartile Range (IQR) method:")
    print(" - Compute Q1 and Q3 (the 25th and 75th percentiles).")
    print(" - Calculate the IQR as Q3 - Q1.")
    print(" - Flag any data points that are below Q1 - 1.5*IQR or above Q3 + 1.5*IQR as potential outliers.")
    print("Alternatively, you can use z-scores (flagging values with |z| > 3) to identify outliers.")

if __name__ == "__main__":
    main()
