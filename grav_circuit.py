import numpy as np 
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Iterable, Any
import os, json, time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile 
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler 
from qiskit_ibm_runtime.fake_provider import FakeTorino 
from qiskit.quantum_info import Pauli, DensityMatrix, Statevector
from qiskit.circuit.library import UnitaryGate
from itertools import product
from qiskit_aer import AerSimulator 
from collections import Counter
from functools import reduce
from dotenv import load_dotenv

# setting the font for matplotlib
plt.rcParams.update({
    "font.family": "serif", 
})

load_dotenv()
token = os.getenv("IBM_QUANTUM_TOKEN")
USE_REAL_BACKEND = True
SHOTS_PER_CIRCUIT = 2048
# 98364 is an arbitrary number I chose for reproducability
TRANSPILE_SEED = 98364
backend_name = "ibm_fez"
PAULI_BASES = ["Z", "X", "Y"]
DEBUG = False

qubit_0 = 0
qubit_1 = 1

if USE_REAL_BACKEND:
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
    )
    backend = service.backend(backend_name)
    transpilation_backend = backend
    
    # updating qubits to the optimal ibm_fez qubits on 11/27/25 
    qubit_0 = 4
    qubit_1 = 5
else:
    transpilation_backend = AerSimulator(noise_model = None)
    backend = AerSimulator(
        coupling_map=transpilation_backend.configuration().coupling_map,
        basis_gates=transpilation_backend.configuration().basis_gates,
    )
    backend.set_options(seed_simulator = 98364)
    backend_name = "aer_simulator_ideal"

print(f"Transpilation backend: {backend_name}")

# Qiskit only measures in the Z basis, so rotations are needed for the Pauli X and Y bases
def basis_pre_rotation(circ: QuantumCircuit, qubit: int, axis: str) -> None:
    if axis == "X":
        # applies the Hadamard gate
        circ.h(qubit)
    elif axis == "Y":
        # applies the hermitian conjugate of the phase gate, S^dagger, then the Hadamard
        circ.sdg(qubit)
        circ.h(qubit)

_POVM_CACHE: Dict[Tuple[str, str], np.ndarray] = {}

# builds the single-qubit measurement projectors for measuring along the Pauli axes X, Y, and Z
def _one_qubit_projectors(axis: str) -> Tuple[np.ndarray, np.ndarray]:
    if axis == "Z":
        # |z;+/-> = (1/sqrt(2))(|0> +/- |1>)
        # |0><0|
        proj_0 = np.array([[1, 0], [0, 0]], dtype = complex)
        # |1><1|
        proj_1 = np.array([[0, 0], [0, 1]], dtype = complex)
    elif axis == "X":
        # |x;+/-> = (1/sqrt(2))(|z;+> +/- |z;->)
        # |+><+|
        proj_0 = 0.5 * np.array([[1, 1], [1, 1]], dtype = complex)
        # |-><-|
        proj_1 = 0.5 * np.array([[1, -1], [-1, 1]], dtype = complex)
    elif axis == "Y":
        # |y;+/-> = (1/sqrt(2))(|z;+> +/- i|z;->)
        # |i><i|
        proj_0 = 0.5 * np.array([[1, -1j], [1j, 1]], dtype = complex)
        # |-i><-i|
        proj_1 = 0.5 * np.array([[1, 1j], [-1j, 1]], dtype = complex)
    else:
        raise ValueError("the axis must be X, Y, or Z")
    return proj_0, proj_1

# builds a mulitple-qubit measurement operator for a given Pauli basis and a given measurment outcome bitstring
def _tensor_projector(basis: str, bitstring: str) -> np.ndarray:
    key = (basis, bitstring)
    if key in _POVM_CACHE:
        return _POVM_CACHE[key]
    
    single_qubit_projs = [_one_qubit_projectors(ax)[int(bt)] for ax, bt in zip(basis, bitstring)]
    proj = reduce(np.kron, reversed(single_qubit_projs))
    _POVM_CACHE[key] = proj

    return proj

def diluted_mle_algorithm(counts_by_basis: Dict[str, Dict[str, int]], 
                          max_iterations: int = 400, 
                          convergence_tolerance: float = 1e-7,
                          model_probability_floor: float = 1e-12, 
                          renormalize_trace_floor: float | None = None,
                          dilution: float = 1.0, 
                          debug: bool = DEBUG) -> DensityMatrix:
    if renormalize_trace_floor is None:
        renormalize_trace_floor = model_probability_floor

    # calculating hilbert space dimension from passed in number of qubits
    first_basis = next(iter(counts_by_basis))
    n_qubits = len(first_basis)
    dimension = 2 ** n_qubits
    # creates a list of the 2^d possible measurement outcome bitstrings {00, 01, 10, 11}
    outcomes = [''.join(bits) for bits in product('01', repeat = n_qubits)]

    # initializing initial variables
    rho = np.eye(dimension, dtype = complex) / dimension
    projectors: List[np.ndarray] = []
    shot_counts: List[int] = []
    total_shots = 0

    # this loops through every (b, m) pair to create the quantities needed for the iterative algorithm
    # let indices i and j represent qubit 0 and qubit 1
    # joint pauli basis b = b_{ij} = {XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ}
    # joint measurement outcome m = m_{ij} = {00, 01, 10, 11}
    # N = total_shots, f_j = n_shots, and projectors[outcome] = proj^{hat}_j 
    for basis_label, outcome_hist in counts_by_basis.items():
        for outcome in outcomes:
            n_shots = int(outcome_hist.get(outcome, 0))
            if n_shots == 0:
                continue
            projectors.append(_tensor_projector(basis_label, outcome))
            shot_counts.append(n_shots)
            total_shots += n_shots

    if total_shots == 0:
        return DensityMatrix(rho)
    
    # diluted iterative algorithm
    for iteration in range(1, max_iterations + 1):
        model_probabilities = np.array([float(np.real(np.trace(proj @ rho))) for proj in projectors], dtype = float)
        model_probabilities = np.clip(model_probabilities, model_probability_floor, None)

        # creates the state-dependent R operator for use in the iterative step (equation 4)
        R = np.zeros((dimension, dimension), dtype = complex)
        for n_j, p_j, proj in zip(shot_counts, model_probabilities, projectors):
            R = R + (n_j / p_j) * proj
        R = R / total_shots

        # defining the unity operator for the diluted iterative algorithm
        epsilon_ = dilution
        # d x d identity operator (2 x 2 in this case)
        I = np.eye(dimension, dtype = complex)
        # defining the diluted map generator
        G_eps = (I + epsilon_ * R) / (1.0 + epsilon_)
        # applying the non-linear update (equation 9)
        rho_updated_unnorm = G_eps @ rho @ G_eps

        # numerically enforcing hermiticity of G_eps
        rho_updated_unnorm = (rho_updated_unnorm + rho_updated_unnorm.conj().T) / 2

        # normalizing the denisty matrix
        trace_updated = float(np.trace(rho_updated_unnorm).real)

        if trace_updated <= renormalize_trace_floor:
            rho_updated = np.eye(dimension, dtype = complex) / dimension
        else:
            rho_updated = rho_updated_unnorm / trace_updated

        # applying the dilution to the state density matrix
        rho_next = rho_updated

        # checking convergence using the frobenius norm ||rho_(k + 1) - rho-k||
        change = np.linalg.norm(rho_next - rho, ord = 'fro')

        if debug:
            # for the current {f_j}, this calculates the negative of the paper's equation 2
            neg_log_likelihood = -1 * np.sum(np.array(shot_counts) * np.log(model_probabilities))
            print(f"MLE iteration {iteration:3d}: change = {change:3e}, neg log likelihood = {neg_log_likelihood:.3f}")
        
        rho = rho_next
        if change < convergence_tolerance:
            break

    # setting rho equal to its hermitian conjugate
    rho = (rho + rho.conj().T) / 2
    # normalizing the trace so that Tr(rho) = 1
    rho = rho / np.trace(rho)
    # spectral decomposition of rho: rho is hermitian, so this diagonalizes it to find its eigenvalues and eigenvectors
    eigenvals, eigenvects = np.linalg.eigh(rho)
    # positive semi-definite condition: clipping negative values
    eigenvals = np.clip(eigenvals, 0, None)
    # reconstructing the eigenspectrum into a density matrix
    rho = eigenvects @ np.diag(eigenvals) @ eigenvects.conj().T
    # normalizing the trace Tr(rho) again
    rho = rho / np.trace(rho)

    print(f"Performed Diluted MLE Quantum Tomography on the State resulting from the gravitational interaction experiment with N = {total_shots} measurements with qubits {qubit_0}, {qubit_1}")
    return DensityMatrix(rho)

# helper function that separates a bitstring by its classical bit indices for Qiskit's little-endian format
def select_cbits_from_key(bitstring: str, indices: Iterable[int], total_bits: int) -> str:
    s = bitstring.replace(' ', '')
    return ''.join(s[total_bits - 1 - i] for i in indices)

def marginalize_counts_by_indices(counts: Dict[str, int],
                                  system_bit_indices: Iterable[int],
                                  total_classical_bits: int) -> Dict[str, int]:
    result = Counter()
    for key, count in counts.items():
        sys_bits = select_cbits_from_key(key, system_bit_indices, total_classical_bits)
        result[sys_bits] = result[sys_bits] + count
    
    return result

# Quantum Circuit that simulates the gravitational interaction experiment
def gravitational_circuit(circ: QuantumCircuit, qubit0: int = 0, qubit1: int = 1, phi_1_val: float = 0.0, delta_phi_val: float = 0.0) -> None:
    # 1. Applying Beam Splitter 1 for qubit 0
    circ.h(qubit0)
    # 2. Applying Beam Splitter 1 for qubit 1
    circ.h(qubit1)

    # Defining the Unitary matrix that models the gravitational interaction betwene each qubit as they travel through the interferometer
    U = np.array([[1, 0, 0, 0], 
                  [0, np.exp(1j * phi_1_val), 0, 0], 
                  [0, 0, np.exp(1j * phi_1_val), 0], 
                  [0, 0, 0, np.exp(1j * (phi_1_val + delta_phi_val))]], dtype=complex)
    grav_unitary = UnitaryGate(U)

    # 3. Applying the U_grav to the joint system
    circ.append(grav_unitary, [qubit0, qubit1])

    # 4. Applying Beam Splitter 2 for qubit 0
    circ.h(qubit0)

    # 5. Applying Beam Splitter 2 for qubit 1
    circ.h(qubit1)

# returns a list of all 9 possible two-qubit Pauli settings: XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
def two_qubit_pauli_settings() -> List[str]:
    return [''.join(bs) for bs in product(PAULI_BASES, repeat = 2)]

# This function
#   1. prepares the bell state |Omega_0> = (1/sqrt(2))(|00> + |11>)
#   2. applies basis pre-rotation
#   3. measures the qubits and stores the readout into 2 classical bits
#   4. adds results to program's metadata as {"basis": basis_label, "sys_index": [0, 1]}
def build_tomography_circuit(basis_label: str, phi_1_val: float, delta_phi_val: float) -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name = f"grav_{basis_label}")
    gravitational_circuit(qc, 0, 1, phi_1_val, delta_phi_val)
    basis_pre_rotation(qc, 0, basis_label[0])
    basis_pre_rotation(qc, 1, basis_label[1])

    # measuring 
    qc.measure([0, 1], [0, 1])
    qc.metadata = {"basis": basis_label, "sys_index": [0, 1]}

    return qc

# returns a list of tomography circuits for each possible Pauli basis pair
def build_tomography_circuits(phi_1_val: float, delta_phi_val: float) -> List[QuantumCircuit]:
    return [build_tomography_circuit(basis, phi_1_val, delta_phi_val) for basis in two_qubit_pauli_settings()]

def run_tomography(backend: Any, 
                   initial_layout: List[int], 
                   shots: int,
                   phi_1_val: float,
                   delta_phi_val: float,
                   optimization_level: int = 3) -> Tuple[List[QuantumCircuit], List[Dict[str, int]]]:
    circuits = build_tomography_circuits(phi_1_val, delta_phi_val)

    transpiled_circuits = transpile(
        circuits,
        backend = backend,
        initial_layout = initial_layout,
        optimization_level = optimization_level,
        seed_transpiler = TRANSPILE_SEED
    )

    if USE_REAL_BACKEND:
        sampler = Sampler(mode = backend)
        job = sampler.run(transpiled_circuits, shots = shots)
        result = job.result()
        counts_per_circuit = [res.join_data().get_counts() for res in result]
    else:
        job = backend.run(transpiled_circuits, shots = shots)
        result = job.result()
        counts_per_circuit = [result.get_counts(i) for i in range(len(transpiled_circuits))]

    return transpiled_circuits, counts_per_circuit

# This function creates a map of each basis label and its 4-valued histogram of measurement outcomes
def counts_grouped_by_basis(transpiled_circuits: List[QuantumCircuit],
                            counts_per_circuit: List[Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    basis_counts: Dict[str, Dict[str, int]] = {}
    for qc, count in zip(transpiled_circuits, counts_per_circuit):
        basis_label = qc.metadata["basis"]
        system_indices = qc.metadata["sys_index"]
        system_counts = marginalize_counts_by_indices(count, system_indices, qc.num_clbits)
        basis_counts[basis_label] = system_counts

    return basis_counts

# returns |psi 5> as a Statevector on two qubits
def ideal_gravitational_statevector(phi_1_val: float, delta_phi_val: float) -> Statevector:
    c0 = 0.25 + 0.5 * np.exp(1j * phi_1_val) + 0.25 * np.exp(1j * (phi_1_val + delta_phi_val))
    c1 = 0.25 - 0.25 * np.exp(1j * (phi_1_val + delta_phi_val))
    c2 = 0.25 - 0.25 * np.exp(1j * (phi_1_val + delta_phi_val))
    c3 = 0.25 - 0.5 * np.exp(1j * phi_1_val) + 0.25 * np.exp(1j * (phi_1_val + delta_phi_val))

    vect = np.array([c0, c1, c2, c3], dtype = complex)

    vect = vect / np.linalg.norm(vect)

    return Statevector(vect)

# returns the expectation value for measuring denisty matrix rho in a specified two-qubit Pauli basis 
def pauli_expectation_value(rho: DensityMatrix, label: str) -> float:
    pauli_gate = Pauli(label).to_matrix()

    # born probability of a given density matrix rho (equation 1 in Diluted maximum-likelihood algorithm for QT paper)
    exp_value = float(np.real(np.trace(pauli_gate @ rho.data)))

    return exp_value

def joint_pauli_expectation_values(rho: DensityMatrix) -> Dict[str, float]:
    labels = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
    return {label: pauli_expectation_value(rho, label) for label in labels}

# returns a 3 x 3 matrix of Pauli basis combinations for two qubits
def pauli_bases_matrix(pauli_strings: Dict[str, float]) -> np.ndarray:
    bases = ["X", "Y", "Z"]
    matrix = np.zeros((3, 3))

    for i, basis_i in enumerate(bases):
        for j, basis_j in enumerate(bases):
            joint_basis = basis_i + basis_j
            matrix[i, j] = pauli_strings[joint_basis]

    return matrix

def single_mass_detector_probabilities(rho: DensityMatrix) -> Tuple[float, float]:
    diagonal_elements = np.real(np.diag(rho.data))
    p_0 = float(diagonal_elements[0] + diagonal_elements[1])
    p_1 = float(diagonal_elements[2] + diagonal_elements[3])

    return p_0, p_1

# calculates the partial transpose of a 2-qubit density matrix
def partial_transpose(rho: np.ndarray, dimensions: Tuple[int, int] = (2, 2), sub_system: int = 1) -> np.ndarray:
    dimension_A, dimension_B = dimensions
    rho_reshaped = rho.reshape(dimension_A, dimension_B, dimension_A, dimension_B)
    
    if sub_system == 0:
        rho_partial_trace = rho_reshaped.transpose(2, 1, 0, 3)
    else:
        rho_partial_trace = rho_reshaped.transpose(0, 3, 2, 1)

    return rho_partial_trace.reshape(dimension_A * dimension_B, dimension_A * dimension_B)

def negativity(rho: DensityMatrix) -> float:
    rho_partial_trace = partial_transpose(rho.data, sub_system = 1)
    eigen_vals = np.linalg.eigvalsh(rho_partial_trace)
    neg = -np.sum(eigen_vals[eigen_vals < 0.0])

    return float(neg)

def run_tomography_for_phases(backend: Any,
                              initial_layout: List[int],
                              phi_1_val: float,
                              delta_phi_val: float,
                              shots: int = SHOTS_PER_CIRCUIT,
                              optimization_level: int = 3) -> Tuple[DensityMatrix, float, float, float]:
    tomography_circuits, counts_list = run_tomography(
        backend = backend,
        initial_layout = initial_layout,
        shots = shots,
        phi_1_val = phi_1_val,
        delta_phi_val = delta_phi_val,
        optimization_level = optimization_level,
    )

    basis_counts = counts_grouped_by_basis(tomography_circuits, counts_list)
    rho_estimate = diluted_mle_algorithm(basis_counts, debug = DEBUG)

    p_0, p_1 = single_mass_detector_probabilities(rho_estimate)
    neg = negativity(rho_estimate)

    return rho_estimate, p_0, p_1, neg

if __name__ == "__main__":
    initial_layout = [qubit_0, qubit_1]

    # running diluted MLE tomography once for reference
    phi_1_reference = 0.0
    delta_phi_reference = 1.0
    
    rho_estimate, p0_ref, p1_ref, neg_ref = run_tomography_for_phases(
        backend = backend,
        initial_layout = initial_layout,
        phi_1_val = phi_1_reference,
        delta_phi_val = delta_phi_reference,
        shots = SHOTS_PER_CIRCUIT,
        optimization_level = 3
    )

    ideal_gravitational_state = ideal_gravitational_statevector(phi_1_reference, delta_phi_reference)
    # calculating the fidelity to the gravitationally-induced entangled state as F = <psi 5|rho|psi 5>
    gravitational_state_fidelity = float((ideal_gravitational_state.data.conj().T @ rho_estimate.data @ ideal_gravitational_state.data).real)

    print(f"Reference phases: phi_1 = {phi_1_reference:.4f}, delta_phi_reference = {delta_phi_reference:.4f}")
    print(f"Calculated Fidelity to ideal Gravitationally-Induced Entangled State: F = {gravitational_state_fidelity:.8f}")
    print(f"Reference Single-mass Detector Probabilities: p0 = {p0_ref:.6f}, p1 = {p1_ref:.6f}")
    print(f"Entanglement at reference phase: N = {neg_ref:.6f}")

    # plotting a heatmap of joint Pauli expectation values
    rho_ideal = DensityMatrix(ideal_gravitational_state)
    joint_pauli_exp_vals = joint_pauli_expectation_values(rho_estimate)
    ideal_joint_pauli_exp_vals = joint_pauli_expectation_values(rho_ideal)

    experimental_pauli_matrix = pauli_bases_matrix(joint_pauli_exp_vals)
    ideal_pauli_matrix = pauli_bases_matrix(ideal_joint_pauli_exp_vals)

    axes_labels = ["X", "Y", "Z"]
    os.makedirs("figures", exist_ok = True)

    heat_map = plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(ideal_pauli_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(3), axes_labels)
    plt.yticks(range(3), axes_labels)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{ideal_pauli_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=10)
    plt.title("Ideal Expectation Values")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(experimental_pauli_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(3), axes_labels)
    plt.yticks(range(3), axes_labels)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{experimental_pauli_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=10)
    plt.title("Experimental Expectation Values")
    plt.colorbar()

    plt.tight_layout()
    heatmap_path = os.path.join("figures", f"pauli_heatmap_phi_1{phi_1_reference:.2f}_d_phi_{delta_phi_reference:.2f}.png")
    heat_map.savefig(heatmap_path, dpi=300)
    plt.close(heat_map)

    joint_bases_expectation_values = {
        "XX": pauli_expectation_value(rho_estimate, "XX"),
        "XY": pauli_expectation_value(rho_estimate, "XY"),
        "XZ": pauli_expectation_value(rho_estimate, "XZ"),
        "YX": pauli_expectation_value(rho_estimate, "YX"),
        "YY": pauli_expectation_value(rho_estimate, "YY"),
        "YZ": pauli_expectation_value(rho_estimate, "YZ"),
        "ZX": pauli_expectation_value(rho_estimate, "ZX"),
        "ZY": pauli_expectation_value(rho_estimate, "ZY"),
        "ZZ": pauli_expectation_value(rho_estimate, "ZZ")
    }

    # defining a constant phi_1 for the variable delta_phi plot
    fixed_phi_1 = 0.0

    # sampling 30 phase values between 0 and 2 pi to plot
    n_phase_values = 15
    delta_phi_values = np.linspace(0.0, 2.0 * np.pi, n_phase_values)

    p0_values = []
    p1_values = []
    neg_values = []

    for delta_phi in delta_phi_values:
        rho_value, p0, p1, neg = run_tomography_for_phases(
            backend = backend,
            initial_layout = initial_layout,
            phi_1_val = fixed_phi_1,
            delta_phi_val = delta_phi,
            shots = SHOTS_PER_CIRCUIT,
            optimization_level = 3
        )

        p0_values.append(p0)
        p1_values.append(p1)
        neg_values.append(neg)

    p0_values = np.array(p0_values)
    p1_values = np.array(p1_values)
    neg_values = np.array(neg_values)

    # theoretical detector probabilities to use for comparison in the plot (equation 2 from the paper)
    theoretical_p0 = 0.5 * (np.cos(fixed_phi_1 / 2.0) ** 2 + np.cos(delta_phi_values / 2.0) ** 2)
    theoretical_p1 = 0.5 * (np.sin(fixed_phi_1 / 2.0) ** 2 + np.sin(delta_phi_values / 2.0) ** 2)

    # plot 2: Detector probabilities for each mass as a function of relative phase
    detector_probs_plot = plt.figure()
    plt.plot(delta_phi_values, p0_values, "o", label = r"$p_0$ (experimental)")
    plt.plot(delta_phi_values, p1_values, "o", label = r"$p_1$ (experimental)")
    plt.plot(delta_phi_values, theoretical_p0, "--", label = r"$p_0$ (theoretical)")
    plt.plot(delta_phi_values, theoretical_p1, "--", label = r"$p_1$ (theoretical)")
    plt.xlabel(r"$\Delta\phi$ (radians)", fontsize=12)
    plt.ylabel("Single-Mass Detector Probability", fontsize=12)
    plt.title(r"Detector Probabilities vs Relative Phase", fontsize=14)
    plt.legend()
    plt.tight_layout()
    detector_probs_plot_path = os.path.join("figures", "detector_probs_vs_delta_phi.png")
    detector_probs_plot.savefig(detector_probs_plot_path, dpi=300)
    plt.close(detector_probs_plot)

    # plot 3: Entanglement strength vs relative phase
    entanglement_plot = plt.figure()
    plt.plot(delta_phi_values, neg_values, "o-")
    plt.xlabel(r"$\Delta\phi$ (radians)", fontsize=12)
    plt.ylabel(r"Negativity N($\rho$)", fontsize=12)
    plt.title(r"Entanglement Strength vs Relative Phase", fontsize=14)
    plt.tight_layout()
    entanglement_plot_path = os.path.join("figures", "entanglement_vs_delta_phi.png")
    entanglement_plot.savefig(entanglement_plot_path, dpi = 300)
    plt.close(entanglement_plot)

    # printing results
    print(f"Calculated Fidelity to Ideal Gravitationally-Induced State: F = {gravitational_state_fidelity:.6f}")
    print(f"Fidelity to Joint Pauli Bases:")
    print(f"\t<XX>: F = {joint_bases_expectation_values['XX']:.6f}")
    print(f"\t<XY>: F = {joint_bases_expectation_values['XY']:.6f}")
    print(f"\t<XZ>: F = {joint_bases_expectation_values['XZ']:.6f}")
    print(f"\t<YX>: F = {joint_bases_expectation_values['YX']:.6f}")
    print(f"\t<YY>: F = {joint_bases_expectation_values['YY']:.6f}")
    print(f"\t<YZ>: F = {joint_bases_expectation_values['YZ']:.6f}")
    print(f"\t<ZX>: F = {joint_bases_expectation_values['ZX']:.6f}")
    print(f"\t<ZY>: F = {joint_bases_expectation_values['ZY']:.6f}")
    print(f"\t<ZZ>: F = {joint_bases_expectation_values['ZZ']:.6f}")

    # saving program metadata
    os.makedirs("results", exist_ok = True)
    experimental_results = {
        "backend": backend_name,
        "used_real_backend": USE_REAL_BACKEND,
        "shots_per_circuit": SHOTS_PER_CIRCUIT,
        "layout": {"qubit_0": qubit_0, "qubit_1": qubit_1},
        "reference_phases": {"phi_1": phi_1_reference, "delta_phi": delta_phi_reference},
        "bases": two_qubit_pauli_settings(),
        "gravitationally_induced_state_fidelity": gravitational_state_fidelity,
        "joint_bases_expectation_values": joint_bases_expectation_values,
        "reference_detector_probabilities": {"p0": p0_ref, "p1": p1_ref},
        "reference_entanglement": neg_ref,
        "phase_values": {
            "phi_1_fixed": fixed_phi_1,
            "delta_phi_values": delta_phi_values.tolist(),
            "p0_values": p0_values.tolist(),
            "p1_values": p1_values.tolist(),
            "entanglement_strength_values": neg_values.tolist(),
        },
    }

    file_path = f"results/{backend_name}_grav_state_tomography_{int(time.time())}.json"
    with open(file_path, "w") as file:
        json.dump(experimental_results, file, indent = 2)
    print(f"Saved experimental results to {file_path}")
