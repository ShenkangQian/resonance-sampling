"""Numerical checks for random features and frequency-response sampling."""
from pathlib import Path
import json
import math
import numpy as np

out = Path(__file__).resolve().parent
rng = np.random.default_rng(20260911)
a, b = rng.normal(size=(2, 100000))
centers = -b / a
scaled_centers = -(3 * b) / (3 * a)
assert np.allclose(centers, scaled_centers)
epsilon = 0.05
prob_single = 2 * math.atan(epsilon) / math.pi
prob_at_least_one = 1 - (1 - prob_single) ** 20

# A fixed one-feature model: F=(1,2), y=(1,2), noise variance=1, lambda=1.
readout_mean = 5 / 6
readout_variance = 1 / 6
test_feature = 3.0
function_variance = test_feature ** 2 * readout_variance
assert abs(function_variance - 1.5) < 1e-12

# A fixed-free mass-spring chain. The full modal decomposition is evaluator-only.
n = 48
stiffness = np.diag(np.full(n, 2.0)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)
stiffness[-1, -1] = 1.0
mass = np.eye(n)
alpha, beta = 0.001, 0.0002
damping = alpha * mass + beta * stiffness
eigenvalues, modes = np.linalg.eigh(stiffness)
input_vector = np.zeros(n)
input_vector[-1] = 1.0
modal_weights = (modes.T @ input_vector) ** 2

def response(omega):
    omega = np.atleast_1d(omega)
    denominator = eigenvalues[:, None] - omega[None, :] ** 2 + 1j * omega[None, :] * (alpha + beta * eigenvalues[:, None])
    return np.sum(modal_weights[:, None] / denominator, axis=0)

for omega in [0.027, 0.12, 0.43, 0.88]:
    matrix = stiffness - omega ** 2 * mass + 1j * omega * damping
    direct = input_vector @ np.linalg.solve(matrix, input_vector)
    assert np.allclose(direct, response([omega])[0], rtol=1e-9, atol=1e-9)

coarse_omega = np.linspace(0.01, 1.0, 25)
dense_omega = np.linspace(0.01, 1.0, 30001)
coarse_amp = np.abs(response(coarse_omega))
dense_amp = np.abs(response(dense_omega))
peak_idx = int(np.argmax(dense_amp))
assert coarse_amp.max() < dense_amp.max()
np.savetxt(out / 'toy_frequency_response.csv', np.column_stack([dense_omega[::10], dense_amp[::10]]), delimiter=',', header='dimensionless_angular_frequency,dimensionless_amplitude', comments='')
result = {
    'status': 'synthetic numerical checks',
    'center_scale_check': {
        'samples': len(a), 'ratio_identity_checked': True,
        'interval': [-epsilon, epsilon], 'one_feature_center_probability': prob_single,
        'twenty_feature_at_least_one_center_probability': prob_at_least_one,
        'warning': 'Center presence alone neither ensures adequate width nor approximation accuracy.'
    },
    'conditional_readout_variance_check': {
        'readout_mean': readout_mean, 'readout_variance': readout_variance,
        'test_prediction_mean': test_feature * readout_mean,
        'latent_function_variance': function_variance,
        'observation_prediction_variance': function_variance + 1,
        'variance_of_repeated_conditional_mean_at_fixed_features': 0
    },
    'frequency_response_probe': {
        'degrees_of_freedom': n, 'direct_modal_cross_checks': 4,
        'coarse_queries': len(coarse_omega), 'reference_grid_points': len(dense_omega),
        'reference_peak_frequency': float(dense_omega[peak_idx]),
        'reference_grid_maximum': float(dense_amp[peak_idx]),
        'coarse_sample_maximum': float(coarse_amp.max()),
        'coarse_max_to_reference_max_ratio': float(coarse_amp.max() / dense_amp.max()),
        'warning': 'Reference is a dense-grid maximum, not a certified continuous maximum. No learned model or adaptive strategy was tested. This toy does not prove an expensive-simulation bottleneck.'
    }
}
(out / 'checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
