import numpy as np
from scipy.signal import find_peaks

def extract_top_k_seasonalities(series, k, L):
    """
    Extracts the top-k seasonality periods and frequencies, following Algorithm 3
    from the paper: "From Tables to Time: How TabPFN-v2 Outperforms Specialized Time Series Forecasting Models"

    Args:
        series (np.ndarray): The input time series data.
        k (int): The number of top seasonalities to extract.
        L (int): The window length for the Hann window.

    Returns:
        top_k_periods (list): A list of the top-k seasonality periods.
        top_k_freqs (list): A list of the corresponding frequencies.
    """
    series = np.asarray(series)
    N = len(series)
    
    # 1. Preprocess: Find and remove linear trend
    t = np.arange(N)
    alpha, beta = np.polyfit(t, series, 1)
    series_detrended = series - (alpha * t + beta)
    
    # 2. Preprocess: Apply Hann window via convolution
    w_hann = np.hanning(L)
    series_smoothed = np.convolve(series_detrended, w_hann, mode='same')
    
    # 3. Symmetrically zero-pad to length 2N
    pad_left = N // 2
    pad_right = N - pad_left
    series_padded = np.pad(series_smoothed, (pad_left, pad_right), mode='constant', constant_values=0)
    
    # 4. Spectral Analysis: Fast Fourier Transform
    fft_result = np.fft.rfft(series_padded)
    mags = np.abs(fft_result)
    freqs = np.fft.rfftfreq(len(series_padded))
    
    # 5. Remove DC component
    mags[0] = 0 
    
    # 6. Peak Selection: Identify local maxima
    # find_peaks handles the "midpoint of multi-point peaks" requirement natively
    peak_indices, _ = find_peaks(mags)
    
    # 7 & 8 & 9. Period Conversion & Build Top-K Set
    unique_periods = {}
    
    for idx in peak_indices:
        freq = freqs[idx]
        if freq <= 0:  
            continue
            
        # Step 7: Convert to period and cast strictly to integer
        period = int(np.round(1 / freq))
        if period <= 0:
            continue
            
        mag = mags[idx]
        
        # Step 8: Remove duplicates (keep the one with the highest magnitude)
        if period not in unique_periods or mag > unique_periods[period]['mag']:
            unique_periods[period] = {'mag': mag, 'freq': freq}
            
    # Step 9: Build top k index set
    sorted_items = sorted(unique_periods.items(), key=lambda x: x[1]['mag'], reverse=True)
    top_k_items = sorted_items[:k]
    
    # Step 10: Return {p_i | i in T}
    top_k_periods = [item[0] for item in top_k_items]
    top_k_freqs = [item[1]['freq'] for item in top_k_items]
    
    return top_k_periods, top_k_freqs

def build_fourier_features(t, top_k_freqs):
    """
    Builds the Phi_auto(t) feature vector based on detected frequencies.
    """
    t = np.asarray(t)
    features = []
    
    # Construct the alternating cos/sin array for each frequency
    for f in top_k_freqs:
        features.append(np.cos(2 * np.pi * f * t))
        features.append(np.sin(2 * np.pi * f * t))
        
    # If t is a single scalar, return a 1D vector (length 2k). 
    # If t is an array, return a 2D matrix of shape (len(t), 2k)
    if t.ndim > 0:
        return np.column_stack(features)
    else:
        return np.array(features)
    

if __name__ == "__main__":

    # Create a dummy signal with a mixture of two seasonalities (periods of 10 and 25)
    np.random.seed(42)
    t_signal = np.arange(200)
    # Base signal + linear trend + noise
    signal = np.sin(2 * np.pi * (1/10) * t_signal) + 0.5 * np.cos(2 * np.pi * (1/25) * t_signal) 
    signal += 0.05 * t_signal + np.random.normal(0, 0.2, len(t_signal))

    # Parameters
    k = 2  # Extract top 2 seasonalities
    L = 10 # Hann window size

    # Run Algorithm
    periods, frequencies = extract_top_k_seasonalities(signal, k, L)
    print(f"Top {k} Detected Periods: {periods}")
    print(f"Associated Frequencies: {frequencies}")

    # Build features for a specific time step (e.g., t = 201)
    t_target = 201
    phi_auto = build_fourier_features(t_target, frequencies)
    print(f"\nFeature Vector Shape: {phi_auto.shape}")
    print(f"Feature Vector Phi_auto(201): \n{phi_auto}")

    # Build features for a range of time steps (e.g., t = 201 to 210)
    t_range = np.arange(201, 211)
    phi_auto_range = build_fourier_features(t_range, frequencies)
    print(f"\nFeature Matrix Shape: {phi_auto_range.shape}")
    print(f"Feature Matrix Phi_auto(201 to 210): \n{phi_auto_range}")
