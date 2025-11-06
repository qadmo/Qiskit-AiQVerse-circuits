from qiskit import QuantumCircuit

# Create a simple 2-qubit circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])
