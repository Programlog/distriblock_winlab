import numpy as np
import torch

def debug_computation(p_ctc_prob, batch_idx):
    """Helper function to debug computations"""
    print(f"\nBatch {batch_idx} Statistics:")
    print(f"Shape of p_ctc_prob: {p_ctc_prob.shape}")
    print(f"Contains NaN: {np.isnan(p_ctc_prob).any()}")
    print(f"Contains Inf: {np.isinf(p_ctc_prob).any()}")
    print(f"Min value: {np.min(p_ctc_prob)}")
    print(f"Max value: {np.max(p_ctc_prob)}")
    return not (np.isnan(p_ctc_prob).any() or np.isinf(p_ctc_prob).any())

def safe_computations(p_ctc_prob):
    """Safely compute characteristics handling edge cases"""
    # Remove any remaining infinite values
    p_ctc_prob = np.nan_to_num(p_ctc_prob, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Add small epsilon to zero values to avoid log(0)
    epsilon = 1e-10
    p_ctc_prob = np.clip(p_ctc_prob, epsilon, 1.0 - epsilon)
    
    # Ensure probabilities sum to 1 along axis 1
    p_ctc_prob = p_ctc_prob / p_ctc_prob.sum(axis=1, keepdims=True)
    
    return p_ctc_prob

def characteristics(batch, stage, device='cuda'):
    """Compute characteristics with additional error checking"""
    measurements = {
        'Entropy': [], 'Max': [], 'Min': [], 'Median': [],
        'JSD': [], 'KLD': []
    }
    
    with torch.no_grad():
        for i, b in enumerate(batch):
            # Move batch to device
            b = b.to(device)
            
            # Get probabilities
            p_ctc = torch.squeeze(b[0], dim=0)
            p_ctc_prob = torch.exp(p_ctc).cpu().numpy()
            
            # Debug print
            is_valid = debug_computation(p_ctc_prob, i)
            if not is_valid:
                print(f"Invalid values detected in batch {i}, attempting to fix...")
                p_ctc_prob = safe_computations(p_ctc_prob)
            
            # Compute characteristics
            try:
                # Entropy
                entropy_vals = -np.sum(p_ctc_prob * np.log(p_ctc_prob + 1e-10), axis=1)
                measurements['Entropy'].append(np.mean(entropy_vals))
                
                # Max probability
                max_prob = np.max(p_ctc_prob, axis=1)
                measurements['Max'].append(np.mean(max_prob))
                
                # Min probability
                min_prob = np.min(p_ctc_prob, axis=1)
                measurements['Min'].append(np.mean(min_prob))
                
                # Median probability
                median_prob = np.median(p_ctc_prob, axis=1)
                measurements['Median'].append(np.mean(median_prob))
                
                # JSD
                jsds = []
                for j in range(p_ctc_prob.shape[0] - 1):
                    p = p_ctc_prob[j]
                    q = p_ctc_prob[j + 1]
                    m = 0.5 * (p + q)
                    jsd = 0.5 * (np.sum(p * np.log(p/m + 1e-10)) + np.sum(q * np.log(q/m + 1e-10)))
                    jsds.append(jsd)
                if jsds:
                    measurements['JSD'].append(np.mean(jsds))
                
                # KLD
                klds = []
                for j in range(p_ctc_prob.shape[0] - 1):
                    p = p_ctc_prob[j]
                    q = p_ctc_prob[j + 1]
                    kld = np.sum(p * np.log(p/(q + 1e-10) + 1e-10))
                    klds.append(kld)
                if klds:
                    measurements['KLD'].append(np.mean(klds))
                
            except Exception as e:
                print(f"Error in batch {i}: {str(e)}")
                continue
    
    return measurements

