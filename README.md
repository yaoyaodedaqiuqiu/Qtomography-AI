# Qtomography-AI
Utilizing AI to explore Quantum Many-body system


## updata_1.0 : 2024/11/26
### dataset.py 

A series of classes have been designed to simulate and generate Gibbs states, implementing the following functions:
- An interface for importing the Hamiltonian of different many-body models, with the Ising model currently defined.
- The ability to perform a temperature gradient scan under the Gibbs state, with customizable temperature range and step size.

### model.py

A simple Multi-Layer Perceptron (MLP) architecture is employed to predict the temperature of Gibbs states. The model consists of two hidden layers and an output layer, using the ReLU activation function and Dropout to prevent overfitting. The input is a flattened matrix of Gibbs states, and the output is the corresponding temperature value, making it a regression problem.

### train.py

The following functions are defined in `train.py`:
- **load_gibbs_states()**: Loads Gibbs state data from a `.pkl` file.
- **preprocess_data()**: Flattens and standardizes the Gibbs state matrix to prepare it as input for the neural network.
- **train()**: Defines the training process, including forward propagation, loss calculation, backpropagation, and optimization steps.
- **save_model()**: Saves the trained model to a file for later use.


## updata_0.0 : 2024/11/22
upload the previous code and trial for exploring GCN in Ising model

### Ising_dataset.py

This Python script generates a dataset for the 9-qubit Ising model, simulating quantum measurements and the computation of the ground state energy for random Hamiltonians. The script includes functions to:

1. **Construct the Ising Hamiltonian**: The `build_ising_hamiltonian` function generates the Ising Hamiltonian matrix for a given set of coupling strengths `J` and transverse field strengths `h`.
2. **Compute the Kronecker product**: The `kron_n` function computes the Kronecker product of multiple matrices, which is essential for constructing the Hamiltonian in the tensor product space.
3. **Simulate Random Pauli Measurements**: The `measure_random_pauli` function simulates measurements on randomly chosen qubits with random Pauli operators (X, Y, Z). The outcome is determined based on the expectation value of the Pauli operator.
4. **Generate the Ising Dataset**: The `generate_ising_dataset` function creates a dataset of quantum states, where each sample consists of the measurement qubits, the measurement result, the Hamiltonian matrix, and the corresponding target eigenvalue (ground state energy).


### Ising_train.py

This Python script implements a Graph Neural Network (GNN) using PyTorch to predict the minimum eigenvalue (ground state energy) of the Ising model Hamiltonian based on quantum measurement results. The key components of the code include:

1. **GNN Model**: The `GNNModel` class defines a graph neural network with two layers of `GCNConv`, followed by a global mean pooling layer and a fully connected layer to predict a scalar value (the ground state energy). It also learns edge features via a parameterized tensor.

2. **Graph Construction**: The `create_full_graph` function constructs a full graph representation of the quantum system, where each qubit is a node, and edges are created between all pairs of qubits. The measurement results are used as node features, and all edges are initialized with the same feature.

3. **Training and Testing**: 
   - The `train` function performs a forward pass, computes the loss using the Mean Squared Error (MSE) between the model's prediction and the target (the ground state energy), and updates the model's parameters.
   - The `test` function evaluates the model's prediction on the first sample of the dataset and compares it with the ground truth.

4. **Data Loading**: The `load_dataset` function loads the Ising model dataset from a pickle file, and the `get_batches` function yields data in mini-batches for training.

5. **Training Loop**: The model is trained over a specified number of epochs, and the loss is printed after each epoch.


### draw.py
### trial.ipynb
### requirements.txt


