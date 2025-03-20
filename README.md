# Qtomography-AI

Utilizing AI to explore Quantum Many-body Systems

## Welcome

Welcome to **Qtomography-AI**! This project leverages artificial intelligence to investigate quantum many-body systems, specifically focusing on the Ising model and its Gibbs states.

### Getting Started

Follow these steps to set up and run the project:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yaoyaodedaqiuqiu/Qtomography-AI.git
    cd Qtomography-AI
    ```

2. **Set Up a Virtual Environment**:
    - **Using `venv`**:
        ```bash
        python -m venv venv
        ```
    - **Activate the Virtual Environment**:
        - **On Windows**:
            ```bash
            .\venv\Scripts\activate
            ```
        - **On macOS/Linux**:
            ```bash
            source venv/bin/activate
            ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Training Script**:
    ```bash
    python train.py
    ```
    - **Optional: Specify Model Type**:
        ```bash
        python train.py --model transformer
        python train.py --model cnn
        python train.py --model gcn
        ```

5. **Run the Testing Script**:
    ```bash
    python test.py
    ```

6. **Using TensorBoard for Visualization**:
    ```bash
    # Install TensorBoard if not already installed
    pip install tensorboard
    
    # Start TensorBoard server
    tensorboard --logdir=runs
    ```
    
    - **For Remote Servers**:
        ```bash
        # On server
        tensorboard --logdir=runs --host=0.0.0.0 --port=6006
        
        # On local machine (in a new terminal)
        ssh -L 6006:localhost:6006 username@server_ip
        ```
        Then open your browser and navigate to: `http://localhost:6006`

7. **View Example Results**:

    After running the testing script, you can find the generated correlation plots in the `figure_test` directory. Below are some example results:

    #### Heatmap of Predicted Correlation `<σ₁ᶻ σ₂ᶻ>`

    ![Heatmap of Correlation](figure_test/heatmap_correlation.png)

    *Heatmap: Predicted Correlation `<σ₁ᶻ σ₂ᶻ>` Across Temperature T and Transverse Field h.*

    #### Correlation Functions for Specific h Values




    ![Correlation Functions h=0.5](figure_test/correlation_functions_h0.5.png)

    *Figures: Predicted vs True Correlation Functions for Specific h Values.*

    #### Precision Heatmap

    ![Precision Heatmap](figure_test/precision_heatmap.png)

    *Heatmap: Standard Deviation of Predicted Correlation Across Temperature T and Transverse Field h, indicating prediction precision.*

    #### TensorBoard Visualization Examples

    ![Training Loss](figure/loss.png)

    *Training loss curve showing model convergence over epochs. The decreasing trend indicates successful learning.*

    ![Gradient Flow](figure/gradients.png)

    *Gradient flow visualization showing the distribution of gradients across different layers during training.*

    ![Model Graph](figure/graph.png)

    *Interactive model architecture graph showing the network structure and connections between layers.*

    ![Parameter Distribution](figure/parameter.png)

    *Distribution of model parameters across training, providing insights into weight initialization and updates.*

#### Unsupervised Phase Transition Detection
![Phase Transition Analysis](figure/phase_transition_analysis.png)

*Heatmap: Unsupervised detection of quantum phase transitions using correlation features. Critical regions (yellow) align with theoretical critical point h/J=1, achieving 95.5% anomaly detection accuracy near criticality.*

#### Key Metrics:
- **Samples Analyzed**: 1,331 
- **Critical Region Detection Rate**: 95.5% 
- **Analysis Techniques**: PCA + DBSCAN clustering on correlation features

## Updates

### Update 2.4.1 : 2025/3/20

- **Enhanced Training Framework**:
  - Added support for multiple model architectures (Transformer, CNN, GCN)
  - Integrated TensorBoard for comprehensive training visualization
  - Improved model selection via command-line arguments

### Update 2.3.1 : 2025/3/13

- **Code Optimization & Maintenance**:
  - Removed redundant class `CorrelationDatasetPyTorch` from `test.py`

### Update 2.3 : 2025/3/10

- **Added Unsupervised Phase Analysis**:
  - Implemented `draw.py` for automated phase transition detection:
    - **Feature Engineering**: Extracts 4 key correlation features (mean strength, decay slope, variance, endpoint difference)
    - **Dimensionality Reduction**: Uses PCA for 2D visualization of high-dimensional correlation patterns
    - **Cluster Analysis**: Applies DBSCAN to identify anomalous regions in parameter space (h/J vs T)
  - **Validation Metrics**:
    - Achieved 95.5% detection rate near theoretical critical point (h/J=1)
    - Processed 1,331 samples across diverse Hamiltonian parameters
  - **Visualization**:
    - Generates dual plots showing feature space clustering and parameter space phase diagram
    - Highlights quantum critical regions using unsupervised learning

- **Updated Project Structure**:
  - Updated documentation with unsupervised learning capabilities


### Update 2.2 : 2025/1/9

- **Integrated `CorrelationDatasetPyTorch` into `dataset.py`**:
    - Moved the scripts related to training the AI models, including the main training loop, evaluation, and visualization tools.

##### `model/`

Houses the neural network architectures used for predicting correlations in the quantum many-body system.

##### `dataset/`

Includes scripts for generating and processing the Gibbs state data, simulating quantum measurements, and preparing datasets for training.

##### `test/`

Introduced for generating test data, loading trained models, evaluating performance, and visualizing results.

### Update 1.0 : 2024/11/26

##### `dataset.py`

A series of classes have been designed to simulate and generate Gibbs states, implementing the following functions:
- An interface for importing the Hamiltonian of different many-body models, with the Ising model currently defined.
- The ability to perform a temperature gradient scan under the Gibbs state, with customizable temperature range and step size.

##### `model.py`

A simple Multi-Layer Perceptron (MLP) architecture is employed to predict the temperature of Gibbs states. The model consists of two hidden layers and an output layer, using the ReLU activation function and Dropout to prevent overfitting. The input is a flattened matrix of Gibbs states, and the output is the corresponding temperature value, making it a regression problem.

##### `train.py`

The following functions are defined in `train.py`:
- **load_gibbs_states()**: Loads Gibbs state data from a `.pkl` file.
- **preprocess_data()**: Flattens and standardizes the Gibbs state matrix to prepare it as input for the neural network.
- **train()**: Defines the training process, including forward propagation, loss calculation, backpropagation, and optimization steps.
- **save_model()**: Saves the trained model to a file for later use.

### Update 0.0 : 2024/11/22

Uploaded the previous code and trial for exploring GCN in the Ising model.

##### `Ising_dataset.py`

This Python script generates a dataset for the 9-qubit Ising model, simulating quantum measurements and the computation of the ground state energy for random Hamiltonians. The script includes functions to:

1. **Construct the Ising Hamiltonian**: The `build_ising_hamiltonian` function generates the Ising Hamiltonian matrix for a given set of coupling strengths `J` and transverse field strengths `h`.
2. **Compute the Kronecker product**: The `kron_n` function computes the Kronecker product of multiple matrices, which is essential for constructing the Hamiltonian in the tensor product space.
3. **Simulate Random Pauli Measurements**: The `measure_random_pauli` function simulates measurements on randomly chosen qubits with random Pauli operators (X, Y, Z). The outcome is determined based on the expectation value of the Pauli operator.
4. **Generate the Ising Dataset**: The `generate_ising_dataset` function creates a dataset of quantum states, where each sample consists of the measurement qubits, the measurement result, the Hamiltonian matrix, and the corresponding target eigenvalue (ground state energy).

##### `Ising_train.py`

This Python script implements a Graph Neural Network (GNN) using PyTorch to predict the minimum eigenvalue (ground state energy) of the Ising model Hamiltonian based on quantum measurement results. The key components of the code include:

1. **GNN Model**: The `GNNModel` class defines a graph neural network with two layers of `GCNConv`, followed by a global mean pooling layer and a fully connected layer to predict a scalar value (the ground state energy). It also learns edge features via a parameterized tensor.

2. **Graph Construction**: The `create_full_graph` function constructs a full graph representation of the quantum system, where each qubit is a node, and edges are created between all pairs of qubits. The measurement results are used as node features, and all edges are initialized with the same feature.

3. **Training and Testing**: 
   - The `train` function performs a forward pass, computes the loss using the Mean Squared Error (MSE) between the model's prediction and the target (the ground state energy), and updates the model's parameters.
   - The `test` function evaluates the model's prediction on the first sample of the dataset and compares it with the ground truth.

4. **Data Loading**: The `load_dataset` function loads the Ising model dataset from a pickle file, and the `get_batches` function yields data in mini-batches for training.

5. **Training Loop**: The model is trained over a specified number of epochs, and the loss is printed after each epoch.

##### `draw.py`

##### `trial.ipynb`

##### `requirements.txt`