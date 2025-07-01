import numpy as np
from sklearn.metrics import mean_squared_error



def calculate_r_squared(y_true, y_pred):
    """
    Calculate R-squared (coefficient of determination) for regression.
    
    Parameters:
    y_true : array-like, true values
    y_pred : array-like, predicted values
    
    Returns:
    r_squared : float, R-squared value
    """
    # Calculate the mean of the true values
    mean_true = np.mean(y_true)
    
    # Calculate the total sum of squares (TSS)
    total_sum_squares = np.sum((y_true - mean_true)**2)
    
    # Calculate the residual sum of squares (RSS)
    residual_sum_squares = np.sum((y_true - y_pred)**2)
    
    # Calculate R-squared
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    
    return r_squared

def compute_metrics(y_pred, y_gt):
    mpe = abs(y_pred - y_gt)
    ae = (y_pred-y_gt)
    mae = mpe.mean()
    me = ae.mean()

    rmse = mean_squared_error(y_gt, y_pred, squared=False)
    rmspe = np.sqrt(np.mean(np.square(((y_gt - y_pred) / abs(y_gt))), axis=0)) * 100
    percent_rmse = (rmse/np.mean(y_gt))*100
    percent_rmse2 = (rmse/(max(y_gt)-min(y_gt)))*100

    r2 = calculate_r_squared(y_pred, y_gt)

    return np.array([mae, me, rmse, percent_rmse, rmspe, r2])

def print_metrics(metrics):
    mae, me, rmse, percent_rmse, rmspe, r2 = metrics
    merics_printout = f'''
        "MAE": {mae},
        "Mean Error": {me},
        "RMSE": {rmse},
        "% RMSE": {percent_rmse},
        "RMSPE": {rmspe},
        "R²": {r2}
    '''
    # print(merics_printout)
    return merics_printout

def get_metrics_dict(metrics, model):
    mae, me, rmse, percent_rmse, rmspe, r2 = metrics
    return {
        "model": model,
        "MAE": mae,
        "Mean Error": me,
        "RMSE": rmse,
        "% RMSE": percent_rmse,
        "RMSPE": rmspe,
        "R²": r2
    }