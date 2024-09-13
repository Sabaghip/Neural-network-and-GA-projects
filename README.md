# Neural Networks and Genetic Algorithm Projects

## Overview
This repository contains two subprojects focused on neural networks and a genetic algorithm. The first part involves implementing neural networks using PyTorch, while the second part focuses on developing a genetic algorithm to solve optimization problems. 

## Table of Contents
1. [Part 1: Implementing Neural Networks with PyTorch](#part-1-implementing-neural-networks-with-pytorch)
   - [Task 1: Digit Recognition with Fully Connected Neural Networks](#task-1-digit-recognition-with-fully-connected-neural-networks)
   - [Task 2: Digit Recognition with Convolutional Neural Networks](#task-2-digit-recognition-with-convolutional-neural-networks)
   - [Task 3: Brain Tumor Detection](#task-3-brain-tumor-detection)
2. [Part 2: Genetic Algorithm Implementation](#part-2-genetic-algorithm-implementation)
   - [Functions Overview](#functions-overview)
3. [Getting Started](#getting-started)
4. [License](#license)
5. [Acknowledgements](#acknowledgements)

## Part 1: Implementing Neural Networks with PyTorch

### Task 1: Digit Recognition with Fully Connected Neural Networks
Complete the `linear-mnist.ipynb` file

### Task 2: Digit Recognition with Convolutional Neural Networks
Complete the `cnn-mnist.ipynb` file

### Task 3: Brain Tumor Detection
This dataset contains images of brains with four labels (no tumor and three types of tumors). You will use transfer learning with the ResNet50 model. Complete the `Project_Tumor.ipynb`

## Part 2: Genetic Algorithm Implementation

This section contains a Jupyter Notebook that implements a genetic algorithm. The key components of the algorithm include:

### Functions Overview
- **create_individual()**: Creates a random chromosome.
- **generate_population(population_size)**: Generates an initial population of individuals.
- **parent_selection(population)**: Selects pairs of parents from the population.
- **next_generation_selection(children_population)**: Chooses the next generation from the children population.
- **crossover(parents)**: Implements the crossover algorithm to produce offspring.
- **mutate(seq)**: Implements mutation on a sequence.
- **fitness_score(seq)**: Calculates the fitness score of an individual.
- **population_fitness(population)**: Calculates the total fitness of the population.
- **check_end(population)**: Checks if the algorithm should terminate.

## Getting Started
To get started, clone or fork the repository and follow the instructions in the respective files to complete the tasks. Make sure to install the required libraries, especially PyTorch for the neural network tasks and Matplotlib for visualization.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
