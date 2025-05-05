from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from my_surface import generate_single_qubit_lut, apply_correction, is_degenerate

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



simulator = AerSimulator()
result = simulator.run(qc, shots=1).result()
measurement = result.get_counts()
#order of measurements is c_n .. c_0
logical = list(measurement.keys())[0][:13][::-1]
logical = [int(i) for i in logical]
z_stab = list(measurement.keys())[0][14:20][::-1]
x_stab = list(measurement.keys())[0][21::][::-1]
for_Lut = int(x_stab + z_stab,2)
print(bin(for_Lut))
#manipulating output
LUT = generate_single_qubit_lut()
print(logical)
if for_Lut in LUT:
    res = apply_correction(logical, LUT[for_Lut])
else:
    res = logical
print(res)
print(is_degenerate(res))
print(any(res))
