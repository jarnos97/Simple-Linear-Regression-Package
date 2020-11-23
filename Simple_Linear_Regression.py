import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


class LinearRegression:
    """
    This package enables creating an OLS model and testing its assumptions.
    The function linear_model initializes an Ordinary Least Squares Linear Regression model and fits it to data.
    The linear regression has 5 assumptions (linearity, no (or little) multicollinearity, normality, no autocorrelation,
    homoscedasticity). Each assumptions has an eponymous function to test the assumption.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.lm = None
        self.pred = None
        self.results = None

    def linear_model(self):
        self.x = sm.tools.tools.add_constant(self.x)  # Adds constant to X
        self.lm = sm.OLS(self.y, self.x).fit()  # Create model and fit to data

    def predict(self):
        self.pred = self.lm.predict(self.x)
        self.results = pd.DataFrame({"Actual": self.y, "Predicted": self.pred})
        self.results["Residuals"] = abs(self.y - self.pred)
        return self.results

    def mae(self):
        return f"Mean Absolute Error: {self.results['Residuals'].mean()}"

    def linearity(self):
        sns.lmplot(x='Actual', y='Predicted', data=self.results, fit_reg=False, height=7)
        line_coords = np.arange(self.results.min().min(), self.results.max().max())
        plt.plot(line_coords, line_coords, color='darkorange', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.show()

    def multicollinearity(self):
        """ Function displays correlation matrix and calculates Variance Inflation Factor.
        VIF > 10 gives an indication that multicollinearity may be present. Vif > 100 mean there is certain
        multicollinearity among variables. High correlation between variables results in parameter estimates having
        large standard errors."""
        plt.subplots(figsize=(10, 10))
        corr_2 = np.triu(self.x.corr())
        sns.heatmap(abs(self.x.corr()), annot=True, mask=corr_2, fmt='.2g', cmap='coolwarm')
        plt.title('Heatmap correlations')
        plt.show()
        x_dict = {}
        for feature in list(self.x.columns):
            x_dict[feature] = self.x[feature].to_list()
        feature_names = list(x_dict.keys())
        x_stack = np.column_stack([v for v in x_dict.values()])
        vif = [variance_inflation_factor(x_stack, s) for s in range(x_stack.shape[1])]
        return_dict = {}
        for index, vif in enumerate(vif):
            return_dict[feature_names[index]] = vif
        return return_dict

    def normality(self):
        """
        First we plot the residuals, then we test the normality using the Anderson-Darling test.
        p-value should be > 0.05
        """
        plt.subplots(figsize=(12, 6))
        plt.title('Distribution of Residuals')
        sns.distplot(self.results['Residuals'])
        plt.show()
        p_value = normal_ad(self.results['Residuals'])[1]  # Second value is the p-value
        if p_value < 0.05:
            return "Assumption not met", p_value
        return "Assumption met", p_value

    def autocorrelation(self):
        """
        Assumes no autocorrelation of the error terms. The value should be between 1.5 and 2.5.
        < 1.5 = positive autocorr. > 2.5 = negative autocorr
        """
        dw = durbin_watson(self.results['Residuals'])
        if dw < 1.5:
            return "Assumption not met - Positive Autocorrelation", dw
        elif dw > 2.5:
            return "Assumption not met - Negative Autocorrelation", dw
        return "Assumption met", dw

    def homoscedasticity(self):
        """
        Function first plots the residuals, then a Breusch-Pagan  test is performed. A p-value < 0.05 indicates the
        model is heteroscedastistic. We assume heteroscedasticity, thus we aim for p-value < 0.05.
        """
        plt.subplots(figsize=(12, 6))
        ax = plt.subplot(111)  # To remove spines
        plt.scatter(x=self.results.index, y=self.results.Residuals, alpha=0.5)
        plt.plot(np.repeat(0, self.results.index.max()), color='darkorange', linestyle='--')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine
        plt.title('Residuals')
        plt.show()
        breuschpagan_test = het_breuschpagan(self.lm.resid, self.lm.model.exog)
        labels = ['Lm Statistic', 'LM Test - p-value', 'F-Statistic', 'F-Test p-value']
        if breuschpagan_test[1] > 0.05:
            return 'Assumption not met', dict(zip(labels, breuschpagan_test))
        return 'Assumption met', dict(zip(labels, breuschpagan_test))


def remove_outliers(xy, column_list):
    """ xy should be a pandas data frame. The function loops through the dependent variables, calculates mean and
    standard deviation, and removes all rows that are more or less than two standard deviations from the mean. A list
    of columns to which the function should be applied is provided by the user The new data frame is returned."""
    for column in list(xy.columns.values):
        mu = np.mean(xy[column])
        s = np.std(xy[column])
        if column in column_list:
            xy = xy[(xy[column] > (mu - 3 * s)) & (xy[column] < (mu + 3 * s))]
    return xy
