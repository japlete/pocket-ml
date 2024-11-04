# Pocket ML

Pocket ML is a tabular AutoML (Automated Machine Learning) tool that runs entirely in the browser. It's designed to provide an easy way to prototype machine learning models for tabular data without the need for setting up a developing environment, paying for server-side processing or worrying about data privacy issues.

You can use the app [here](https://japlete.github.io/pocket-ml).

## Current Status

This project is in early development. Current functionality includes:

- CSV file upload and parsing
- Basic data preprocessing (categorical feature encoding, missing value imputation, scaling)
- Training a regression or classification model
- Displaying some performance metrics
- Automated hyperparameter tuning
- Save trained models in browser storage

## Planned Features

- Prediction from saved models
- Feature importances and other plots in results
- More model types (e.g. time series forecasting)
- Multiclass metrics
- Preprocessing and hypertuning with spline transformations
- More missing value imputation methods

## Deployment

This app is deployed using GitHub Pages. To deploy your own instance:

1. Fork this repository
2. Update the "homepage" field in package.json with your GitHub username
3. Install dependencies: `npm install`
4. Deploy: `npm run deploy`
5. Enable GitHub Pages in your repository settings using the gh-pages branch
