# finding log var of attention distribution to help prune the head

# from .. import logging
import numpy as np
from scipy.optimize import curve_fit

# logger = logging.get_logger(__name__)

def get_fitted_log_var(bin_centers, weights) -> float:
    """
    Calculates variance for a distribution given bin centers and weights (counts).
    
    Args:
        bin_centers (list or np.array): The midpoint or value representing each bin.
        weights (list or np.array): The population count or frequency for each bin.
        
    Returns:
        float: The calculated variance.
    """
    def lognormal_func(x, amplitude, mu, sigma):
        """
        x: bin centers
        amplitude: scaling factor for the histogram height
        mu: mean of the underlying normal distribution (mean of ln(x))
        sigma: standard deviation of the underlying normal distribution (std of ln(x))
        """
        # Avoid division by zero if x contains 0
        # The PDF has a 1/x term
        return (amplitude / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))
    
    def calculate_safe_variance(mu, sigma):
        # 1. Calculate log of variance to avoid overflow
        # Formula: ln(Var) = 2*mu + 2*sigma^2 + ln(1 - exp(-sigma^2))
        
        # Term A: The dominant part (2*mu + 2*sigma^2)
        log_var_term = 2 * mu + 2 * (sigma ** 2)
        
        # Term B: The small adjustment ln(1 - exp(-sigma^2))
        # We use np.log1p(-x) which computes ln(1-x) accurately for small x
        adjustment = np.log1p(-np.exp(-(sigma ** 2)))
        
        total_log_variance = log_var_term + adjustment
        
        # 2. Convert to Scientific Notation (Base 10) for display
        # V = e^L  ->  log10(V) = L / ln(10)
        log10_variance = total_log_variance / np.log(10)
        
        return total_log_variance

    bin_centers = np.array(bin_centers)
    counts = np.array(weights)

    # Guess mu as log of the mean of your bin centers
    guess_mu = np.log(np.average(bin_centers, weights=counts))
    # Guess sigma (start with a standard guess like 0.5 or 1.0)
    guess_sigma = 0.5
    # Guess amplitude (roughly the max count)
    guess_amp = max(counts)

    p0_guess = [guess_amp, guess_mu, guess_sigma]

    params, covariance = curve_fit(lognormal_func, bin_centers, counts, p0=p0_guess, maxfev=500)
    fit_amp, fit_mu, fit_sigma = params
    calculated_variance = calculate_safe_variance(fit_mu, fit_sigma)

    return calculated_variance