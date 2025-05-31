# Quantum Error Correction

### Summary and Intro

These are scripts that simulate quantum circuits in Qiskit across a depolarizing channel of:

$$
\mathcal{E}(\rho) = (1 - p)\rho + \frac{p}{3}\left( X\rho X^\dagger + Y\rho Y^\dagger + Z\rho Z^\dagger \right)
$$

This means that theres a probability, $1-p$, of no error occuring and a probability of $p/3$ of either a Pauli-X ,Y or Z occuring.

We iterate through different values of $p$ and measure the errorate against the depolarizing channel which is plotted.

All codes used are of minimum distance, $d$=3, and are only able to correct up to one error.

## Files

### my_surface.py

This is simulates the [[13,1,3]] surface code with stabilizers:


| X Stabilizers       | Z Stabilizers          |
| ------------------- | ---------------------- |
| s₀ = X₀X₁X₃     | s₆ = Z₀Z₃Z₅        |
| s₁ = X₁X₂X₄     | s₇ = Z₁Z₃Z₄Z₆     |
| s₂ = X₃X₅X₆X₈  | s₈ = Z₂Z₄Z₇        |
| s₃ = X₄X₆X₇X₉  | s₉ = Z₅Z₈Z₁₀      |
| s₄ = X₈X₁₀X₁₁ | s₁₀ = Z₆Z₈Z₉Z₁₁ |
| s₅ = X₉X₁₁X₁₂ | s₁₁ = Z₇Z₉Z₁₂    |

### my_color.py

[[7,1,3]] color AKA Steane Code


| X-Stabilizer       | Z-Stabilizer       |
| ------------------ | ------------------ |
| s₀ = X₀X₁X₂X₃ | s₃ = Z₀Z₁Z₂Z₃ |
| s₁ = X₂X₃X₄X₅ | s₄ = Z₂Z₃Z₄Z₅ |
| s₂ = X₁X₃X₅X₆ | s₅ = Z₁Z₃Z₅Z₆ |

### Rotated

[[9,1,3]] rotated surface code


| X-Stabilizers      | Z-Stabilizers      |
| ------------------ | ------------------ |
| s₀ = X₁X₂       | s₄ = Z₀Z₃       |
| s₁ = X₀X₁X₃X₄ | s₅ = Z₁Z₂Z₄Z₅ |
| s₂ = X₄X₅X₇X₈ | s₆ = Z₃Z₄Z₆Z₇ |
| s₃ = X₆X₇       | s₇ = Z₅Z₈       |

## How to run files

please make sure each file is is in the same directory before running, as some functions are shared across them. as long as you have **qiskit** and **matplotlib** installed you should be able to run them. I do utilize multiprocessing libraries (concurrent futures) to parrallelize the code.

When you run them it may take a while as i do ALOT of runs (**n**), if you wish to change this go ahead but be aware of really inconsistent outputs.

Under the **main** heading, where the code is executed, the graphs should be displayed and the numerical outputs should be saved in a *.npy* file, once you run and save each file you can go to **plots.py** and run either of the functions with input parameters as the .npy files and some colors. The variables you want to save are highlighted in the code but to be clear they are:

***qbers*** - array that stores the QBER values (indexed across probability)

***degen_ratio*** - Array that stores degeneracy ratios (0 when disabled, again, indexed across probability)

If you change probability values (***p_values***) be sure to change it in the rest of the files and **plots.py** (that is, if you want to visualise all of them together) 

I have tried to make this as simple and as customizable as possible, but as one can imagine, they're kind of a trade-off with eachother.

Feel free to mess around with it, just make sure file names and variable names are kept consistent

To enable **degenracy** just uncomment the code within the **simulate_circuit** function.

**test.py** contains the raw surface code except only with a single run, run it to understand what the functions and how the surface code corrects errors (the circuit is written out in full in this script). This is essential to understand before running other scripts (alot of the functions here are implemented in other files as well).

## How a typical run should go

Run test.py to understand everything (if you want) OR

1. run each file my_color, my_surface and rotated_surface - each should have their own respective output plot as well
3) save outputs from each file
4) run a function from "plots.py" to visualise outputs
