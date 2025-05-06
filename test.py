from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from my_surface import generate_single_qubit_lut, apply_correction, is_degenerate
import numpy as np
from itertools import combinations




x_stabilizer_matrix = [
    np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # X1 X2 X4 → 0 1 3
    np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  # X2 X3 X5 → 1 2 4
    np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]),  # X4 X6 X7 X9 → 3 5 6 8
    np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]),  # X5 X7 X8 X10 → 4 6 7 9
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]),  # X9 X11 X12 → 8 10 11
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]),  # X10 X12 X13 → 9 11 12
]

def is_degenerate(error_vector, stabilizer_matrix = x_stabilizer_matrix):
    error_vector = np.array(error_vector)
    if np.all(error_vector == 0):
        return False

    for r in range(len(stabilizer_matrix) + 1):
        for combo in combinations(stabilizer_matrix, r):
            total = np.zeros(len(error_vector), dtype=int)
            for vec in combo:
                total = (total + vec) % 2
            if np.array_equal(total, error_vector):
                return True
    return False

def generate_single_qubit_lut():
    NUM_QUBITS = 13
    NUM_STABILIZERS = 12

    # 0-indexed lists of which stabilizers each qubit affects
    x_stabilizers = { #qubit affetcted : stabilizer that returns -1
        0: [0],
        1: [0,1],
        2: [1],
        3: [0,2],
        4: [1,3],
        5: [2],
        6: [2,3],
        7: [3],
        8: [2,4],
        9: [3,5],
        10: [4],
        11: [4,5],
        12: [5],
    }

    z_stabilizers = {
        0: [6],
        1: [7],
        2: [8],
        3: [6,7],
        4: [7,8],
        5: [6,9],
        6: [7,10],
        7: [8,11],
        8: [9,10],
        9: [10,11],
        10: [9],
        11: [10],
        12: [11],
    }

    def syndrome_for_error(qubit, pauli):
        syndrome = [0] * NUM_STABILIZERS
        if pauli == 'X':
            for s in z_stabilizers.get(qubit, []):
                syndrome[s] = 1
        elif pauli == 'Z':
            for s in x_stabilizers.get(qubit, []):
                syndrome[s] = 1
        elif pauli == 'Y':
            for s in z_stabilizers.get(qubit, []):
                syndrome[s] = 1
            for s in x_stabilizers.get(qubit, []):
                syndrome[s] = 1
        return syndrome

    lut = {}

    for q in range(NUM_QUBITS):
        for pauli in ['X', 'Z', 'Y']:
            syn = syndrome_for_error(q, pauli)
            syndrome_str = (''.join(str(b) for b in syn))


            if syndrome_str not in lut:
                lut[syndrome_str] = [(q, pauli)]
    return lut


def apply_correction(logical, correction_ops):
    corrected = logical.copy()
    for qubit, pauli in correction_ops:
        if pauli in ['X', 'Y']:
            corrected[qubit] ^= 1

    return corrected

data = QuantumRegister(13, 'data')
ar_x= QuantumRegister(6, 'ar_x')
ar_z = QuantumRegister(6, 'ar_z')
cl_x = ClassicalRegister(6, 'cl_x')
cl_z = ClassicalRegister(6, 'cl_z')
cl_data= ClassicalRegister(13, 'cl_data')

qc = QuantumCircuit(data,ar_z, ar_x, cl_x, cl_z, cl_data)
#everythng is |0> though we have to add phase to our code space

qc.h(ar_x[0])
qc.cx(ar_x[0], data[0])
qc.cx(ar_x[0], data[1])
qc.cx(ar_x[0], data[3])
qc.h(ar_x[0])

qc.h(ar_x[1])
qc.cx(ar_x[1], data[1])
qc.cx(ar_x[1], data[2])
qc.cx(ar_x[1], data[4])
qc.h(ar_x[1])

qc.h(ar_x[2])
qc.cx(ar_x[2], data[3])
qc.cx(ar_x[2], data[5])
qc.cx(ar_x[2], data[6])
qc.cx(ar_x[2], data[8])
qc.h(ar_x[2])

qc.h(ar_x[3])
qc.cx(ar_x[3], data[4])
qc.cx(ar_x[3], data[6])
qc.cx(ar_x[3], data[7])
qc.cx(ar_x[3], data[9])
qc.h(ar_x[3])

qc.h(ar_x[4])
qc.cx(ar_x[4], data[8])
qc.cx(ar_x[4], data[10])
qc.cx(ar_x[4], data[11])
qc.h(ar_x[4])

qc.h(ar_x[5])
qc.cx(ar_x[5], data[9])
qc.cx(ar_x[5], data[11])
qc.cx(ar_x[5], data[12])
qc.h(ar_x[5])

"""
Depolarizing noise
"""
qc.x(data[6])

"""
stabilizer measurements
"""


"""
x-stabilizer
"""

qc.h(ar_x[0])
qc.cx(ar_x[0], data[0])
qc.cx(ar_x[0], data[1])
qc.cx(ar_x[0], data[3])
qc.h(ar_x[0])

qc.h(ar_x[1])
qc.cx(ar_x[1], data[1])
qc.cx(ar_x[1], data[2])
qc.cx(ar_x[1], data[4])
qc.h(ar_x[1])

qc.h(ar_x[2])
qc.cx(ar_x[2], data[3])
qc.cx(ar_x[2], data[5])
qc.cx(ar_x[2], data[6])
qc.cx(ar_x[2], data[8])
qc.h(ar_x[2])

qc.h(ar_x[3])
qc.cx(ar_x[3], data[4])
qc.cx(ar_x[3], data[6])
qc.cx(ar_x[3], data[7])
qc.cx(ar_x[3], data[9])
qc.h(ar_x[3])

qc.h(ar_x[4])
qc.cx(ar_x[4], data[8])
qc.cx(ar_x[4], data[10])
qc.cx(ar_x[4], data[11])
qc.h(ar_x[4])

qc.h(ar_x[5])
qc.cx(ar_x[5], data[9])
qc.cx(ar_x[5], data[11])
qc.cx(ar_x[5], data[12])
qc.h(ar_x[5])

qc.measure(ar_x, cl_x)


"""
 z-stabilizer
"""

qc.cx(data[0], ar_z[0])
qc.cx(data[3], ar_z[0])
qc.cx(data[5], ar_z[0])

qc.cx(data[1], ar_z[1])
qc.cx(data[3], ar_z[1])
qc.cx(data[4], ar_z[1])
qc.cx(data[6], ar_z[1])

qc.cx(data[2], ar_z[2])
qc.cx(data[4], ar_z[2])
qc.cx(data[7], ar_z[2])

qc.cx(data[5], ar_z[3])
qc.cx(data[8], ar_z[3])
qc.cx(data[10], ar_z[3])

qc.cx(data[6], ar_z[4])
qc.cx(data[8], ar_z[4])
qc.cx(data[9], ar_z[4])
qc.cx(data[11], ar_z[4])

qc.cx(data[7], ar_z[5])
qc.cx(data[9], ar_z[5])
qc.cx(data[12], ar_z[5])

qc.measure(ar_z, cl_z)

qc.measure(data, cl_data)
# qc.x().c_if()....

LUT = generate_single_qubit_lut()

# for axis in ['Z', 'X', 'Y']:
#     for idx in range(13):  # from 0 to 12
#         for key, value in LUT.items():
#             if value[0] == (idx, axis):
#                 print(f"{key} ({idx},{axis})")


simulator = AerSimulator()
result = simulator.run(qc, shots=1).result()
measurement = result.get_counts()
#order of measurements is c_n .. c_0
logical = list(measurement.keys())[0][:13][::-1]
logical = [int(i) for i in logical]
z_stab = list(measurement.keys())[0][14:20][::-1]
x_stab = list(measurement.keys())[0][21::][::-1]


print("syndrome:",x_stab + z_stab)
for_Lut = x_stab + z_stab
#manipulating output
print("data before correction: ",logical)
if for_Lut in LUT:
    print("correction applied: ",LUT[for_Lut])
    res = apply_correction(logical, LUT[for_Lut])
else:
    res = logical

if(is_degenerate(res)):
    print("is degen, no error: ",res)
elif any(res):
    print("Error", res)
else:
    print("no error",res)