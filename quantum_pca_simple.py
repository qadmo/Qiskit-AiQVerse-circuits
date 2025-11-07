"""
Quantum PCA - Phase Estimation Algorithm
Simplified version for QuantumOps Portal

This script demonstrates quantum phase estimation to compute principal components
of a covariance matrix using IBM Quantum or Aer Simulator.

Based on: https://github.com/bregueral-quantonomiqs/quantum_PCA_Celine
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qiskit_aer import AerSimulator
from scipy.linalg import expm
import json
import math
from time import perf_counter


def generate_sample_covariance_matrix(n_features: int = 4, seed: int = 42) -> np.ndarray:
    """
    Generate a sample positive semi-definite covariance matrix.
    
    Parameters:
    -----------
    n_features : int
        Dimension of the matrix (default: 4)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        A symmetric positive semi-definite matrix
    """
    np.random.seed(seed)
    
    # Generate random data
    data = np.random.randn(100, n_features)
    
    # Compute covariance matrix
    covariance = np.cov(data, rowvar=False)
    
    return covariance


def pad_matrix_to_power_of_two(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Pad a matrix to the next power of 2 dimension.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input square matrix
        
    Returns:
    --------
    tuple[np.ndarray, int]
        (padded_matrix, original_dimension)
    """
    dim = matrix.shape[0]
    target_dim = 1 if dim == 0 else 2 ** (dim - 1).bit_length()
    
    if target_dim == dim:
        return matrix, dim
    
    padded = np.zeros((target_dim, target_dim), dtype=matrix.dtype)
    padded[:dim, :dim] = matrix
    return padded, dim


def pad_vector_to_dimension(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Pad a vector to target dimension and normalize.
    
    Parameters:
    -----------
    vector : np.ndarray
        Input vector
    target_dim : int
        Target dimension
        
    Returns:
    --------
    np.ndarray
        Padded and normalized vector
    """
    vector = np.asarray(vector, dtype=complex)
    
    if len(vector) > target_dim:
        raise ValueError("Vector length exceeds target dimension")
    
    padded = np.zeros(target_dim, dtype=complex)
    padded[:len(vector)] = vector
    
    norm = np.linalg.norm(padded)
    if norm == 0:
        raise ValueError("Vector must be non-zero")
        
    return padded / norm


def build_phase_estimation_circuit(
    unitary_matrix: np.ndarray,
    initial_state: np.ndarray,
    num_evaluation_qubits: int = 6
) -> QuantumCircuit:
    """
    Build a quantum phase estimation circuit.
    
    Parameters:
    -----------
    unitary_matrix : np.ndarray
        The unitary operator to estimate eigenvalues of
    initial_state : np.ndarray
        Initial state (eigenvector)
    num_evaluation_qubits : int
        Number of qubits for phase precision
        
    Returns:
    --------
    QuantumCircuit
        The quantum circuit for phase estimation
    """
    system_qubits = int(math.log2(len(initial_state)))
    total_qubits = num_evaluation_qubits + system_qubits
    
    qc = QuantumCircuit(total_qubits, num_evaluation_qubits)
    
    # Define qubit ranges
    ancilla_qubits = list(range(num_evaluation_qubits))
    system_qubit_indices = list(range(num_evaluation_qubits, total_qubits))
    
    # Step 1: Apply Hadamard to evaluation qubits
    for qubit in ancilla_qubits:
        qc.h(qubit)
    
    # Step 2: Initialize system qubits with the eigenvector
    qc.initialize(initial_state, system_qubit_indices)
    
    # Step 3: Apply controlled unitary operations
    current_power = unitary_matrix.copy()
    for idx, ancilla in enumerate(ancilla_qubits):
        gate = UnitaryGate(current_power, label=f"U^{2**idx}")
        qc.append(gate.control(), [ancilla] + system_qubit_indices)
        current_power = current_power @ current_power
    
    # Step 4: Apply inverse QFT
    qft_dag = QFT(num_evaluation_qubits, do_swaps=False, approximation_degree=0).inverse()
    qc.append(qft_dag, ancilla_qubits)
    
    # Step 5: Measure evaluation qubits
    qc.measure(ancilla_qubits, ancilla_qubits)
    
    return qc


def bitstring_to_phase(bitstring: str) -> float:
    """Convert a bitstring to a phase value."""
    return int(bitstring, 2) / (2 ** len(bitstring))


def extract_eigenvector(
    state: Statevector,
    num_evaluation_qubits: int,
    system_qubits: int,
    original_dim: int
) -> np.ndarray:
    """
    Extract the eigenvector from the quantum state using partial trace.
    
    Parameters:
    -----------
    state : Statevector
        The quantum state after phase estimation
    num_evaluation_qubits : int
        Number of evaluation qubits
    system_qubits : int
        Number of system qubits
    original_dim : int
        Original dimension before padding
        
    Returns:
    --------
    np.ndarray
        The extracted eigenvector
    """
    try:
        # Create density matrix and trace out evaluation qubits
        density = DensityMatrix(state)
        reduced = partial_trace(density, list(range(num_evaluation_qubits)))
        matrix = np.asarray(reduced.data)
        
        # Extract dominant eigenvector
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        idx = int(np.argmax(eigenvalues))
        vector = eigenvectors[:, idx]
        
        # Truncate to original dimension
        vector = vector[:original_dim]
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        else:
            return np.zeros(original_dim)
            
    except Exception as e:
        print(f"Warning: Could not extract eigenvector: {e}")
        return np.zeros(original_dim)


def run_quantum_pca(
    covariance: np.ndarray,
    num_components: int = 3,
    num_evaluation_qubits: int = 6,
    shots: int = 4096,
    backend_name: str = "aer_simulator"
) -> dict:
    """
    Run Quantum PCA using phase estimation.
    
    Parameters:
    -----------
    covariance : np.ndarray
        Covariance matrix to analyze
    num_components : int
        Number of principal components to estimate
    num_evaluation_qubits : int
        Number of qubits for phase precision (higher = more accurate)
    shots : int
        Number of measurement shots
    backend_name : str
        'aer_simulator' for local simulation
        
    Returns:
    --------
    dict
        Results including eigenvalues, eigenvectors, and metadata
    """
    start_time = perf_counter()
    
    # Validate input
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be square")
    
    # Compute classical eigendecomposition for reference
    trace = float(np.trace(covariance))
    if trace <= 0:
        raise ValueError("Covariance matrix must have positive trace")
    
    eigenvalues_classical, eigenvectors_classical = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues_classical)[::-1]
    eigenvalues_classical = eigenvalues_classical[order]
    eigenvectors_classical = eigenvectors_classical[:, order]
    
    # Pad matrix to power of 2
    padded_covariance, original_dim = pad_matrix_to_power_of_two(covariance)
    
    # Create density operator and unitary
    rho = padded_covariance / trace
    unitary = expm(2j * np.pi * rho)
    
    system_qubits = int(math.log2(padded_covariance.shape[0]))
    
    # Initialize backend
    backend = AerSimulator(method="statevector")
    
    # Results storage
    results = []
    
    # Process each component
    for component_idx in range(min(num_components, original_dim)):
        component_start = perf_counter()
        
        # Prepare initial state (eigenvector)
        initial_state = pad_vector_to_dimension(
            eigenvectors_classical[:, component_idx],
            padded_covariance.shape[0]
        )
        
        # Build circuit
        circuit = build_phase_estimation_circuit(
            unitary,
            initial_state,
            num_evaluation_qubits
        )
        
        # Simulate without measurement for statevector
        circuit_no_measure = circuit.remove_final_measurements(inplace=False)
        state = Statevector.from_instruction(circuit_no_measure)
        
        # Run circuit with measurements
        transpiled = transpile(circuit, backend, optimization_level=0)
        job = backend.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Find most likely measurement
        probabilities = {bitstring: count / shots for bitstring, count in counts.items()}
        most_likely_bitstring, probability = max(probabilities.items(), key=lambda x: x[1])
        
        # Convert to phase and eigenvalue
        phase = bitstring_to_phase(most_likely_bitstring)
        estimated_eigenvalue = phase * trace
        
        # Extract eigenvector
        eigenvector = extract_eigenvector(
            state,
            num_evaluation_qubits,
            system_qubits,
            original_dim
        )
        
        # Convert eigenvector to real if it's complex
        if eigenvector is not None:
            eigenvector = np.real_if_close(eigenvector)
            if np.iscomplexobj(eigenvector):
                eigenvector = np.real(eigenvector)
            eigenvector_list = eigenvector.tolist()
        else:
            eigenvector_list = None
        
        component_time = perf_counter() - component_start
        
        # Store results
        results.append({
            "component_index": component_idx,
            "component_name": f"PC{component_idx + 1}",
            "bitstring": most_likely_bitstring,
            "phase": float(phase),
            "estimated_eigenvalue": float(estimated_eigenvalue),
            "classical_eigenvalue": float(eigenvalues_classical[component_idx]),
            "eigenvalue_error": float(abs(estimated_eigenvalue - eigenvalues_classical[component_idx])),
            "probability": float(probability),
            "variance_share": float(estimated_eigenvalue / trace) if trace > 0 else 0.0,
            "eigenvector": eigenvector_list,
            "runtime_seconds": float(component_time)
        })
        
        print(f"Component {component_idx + 1}: λ_quantum={estimated_eigenvalue:.6f}, "
              f"λ_classical={eigenvalues_classical[component_idx]:.6f}, "
              f"probability={probability:.3f}")
    
    total_time = perf_counter() - start_time
    
    # Compute explained variance
    total_variance = eigenvalues_classical.sum()
    explained_variance_ratio = (eigenvalues_classical / total_variance)[:num_components]
    
    return {
        "success": True,
        "backend": backend_name,
        "shots": shots,
        "num_evaluation_qubits": num_evaluation_qubits,
        "num_components": num_components,
        "matrix_dimension": original_dim,
        "trace": float(trace),
        "total_runtime_seconds": float(total_time),
        "classical_eigenvalues": eigenvalues_classical[:num_components].tolist(),
        "explained_variance_ratio": explained_variance_ratio.tolist(),
        "cumulative_variance": np.cumsum(explained_variance_ratio).tolist(),
        "components": results
    }


def main():
    """
    Main execution function - demonstrates Quantum PCA.
    """
    print("=" * 60)
    print("Quantum PCA - Phase Estimation Demo")
    print("=" * 60)
    
    # Generate sample covariance matrix
    print("\n1. Generating sample 4×4 covariance matrix...")
    covariance = generate_sample_covariance_matrix(n_features=4, seed=42)
    print(f"Covariance matrix:\n{covariance}")
    print(f"Trace: {np.trace(covariance):.6f}")
    
    # Run Quantum PCA
    print("\n2. Running Quantum Phase Estimation...")
    results = run_quantum_pca(
        covariance=covariance,
        num_components=3,
        num_evaluation_qubits=6,
        shots=4096,
        backend_name="aer_simulator"
    )
    
    # Display results
    print("\n3. Results:")
    print(f"   Total runtime: {results['total_runtime_seconds']:.2f}s")
    print(f"   Backend: {results['backend']}")
    print(f"   Shots: {results['shots']}")
    
    print("\n4. Principal Components:")
    for comp in results['components']:
        print(f"\n   {comp['component_name']}:")
        print(f"      Quantum eigenvalue:    {comp['estimated_eigenvalue']:.6f}")
        print(f"      Classical eigenvalue:  {comp['classical_eigenvalue']:.6f}")
        print(f"      Error:                 {comp['eigenvalue_error']:.6f}")
        print(f"      Variance explained:    {comp['variance_share']:.2%}")
        print(f"      Measurement prob:      {comp['probability']:.3f}")
        print(f"      Runtime:               {comp['runtime_seconds']:.3f}s")
    
    print("\n5. Cumulative Variance Explained:")
    for i, (ratio, cumul) in enumerate(zip(results['explained_variance_ratio'], 
                                           results['cumulative_variance'])):
        print(f"   PC{i+1}: {ratio:.2%} (cumulative: {cumul:.2%})")
    
    # Export to JSON
    output_file = "quantum_pca_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n6. Results exported to: {output_file}")
    
    print("\n" + "=" * 60)
    print("Quantum PCA Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
