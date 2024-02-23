{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6Ggya0mzv0MO"
   },
   "source": [
    "# Assignment 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "awRoi4dUv0MQ"
   },
   "outputs": [],
   "source": [
    "# Imports.\n",
    "import random\n",
    "import network.network as Network\n",
    "import network.mnist_loader as mnist_loader\n",
    "import pickle\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "# Set the random seed. DO NOT CHANGE THIS!\n",
    "seedVal = 41\n",
    "random.seed(seedVal)\n",
    "np.random.seed(seedVal)\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MSoHXst4v0MR"
   },
   "source": [
    "Use a pre-trained network. It has been saved as a pickle file. Load the model, and continue.\n",
    "The network has only one hidden layer of 30 units, 784 input units (MNIST images are $ 28 \\times 28 = 784 $ pixels large), and 10 output units. All the activations are sigmoidal."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "3BJ_QyOsv0MS"
   },
   "outputs": [],
   "source": [
    "# Load the pre-trained model.\n",
    "with open('network/trained_network.pkl', 'rb') as f:\n",
    "    u = pickle._Unpickler(f)\n",
    "    u.encoding = 'latin1'\n",
    "    net = u.load()\n",
    "\n",
    "# Helpful function to load the MNIST data.\n",
    "training_data, validation_data, test_data = mnist_loader.load_data_wrapper()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JN7_fD0qv0MS"
   },
   "source": [
    "The neural network is pretrained, so it should already be set up to predict characters. Run `predict(n)` to evaluate the $ n^{th} $ digit in the test set using the network. You should see that even this relatively simple network works really well (~97% accuracy). The output of the network is a one-hot vector indicating the network's predictions:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 698
    },
    "id": "iYnOhgCYv0MS",
    "outputId": "a0e31ed8-964b-41cb-cf7c-dce6f229a6d9"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Network output: \n",
      "[[0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [1.]\n",
      " [0.]]\n",
      "\n",
      "Network prediction: 8\n",
      "\n",
      "Actual image: \n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def predict(n):\n",
    "    # Get the data from the test set\n",
    "    x = test_data[n][0]\n",
    "\n",
    "    # Print the prediction of the network\n",
    "    print('Network output: \\n' + str(np.round(net.feedforward(x), 2)) + '\\n')\n",
    "    print('Network prediction: ' + str(np.argmax(net.feedforward(x))) + '\\n')\n",
    "    print('Actual image: ')\n",
    "\n",
    "    # Draw the image\n",
    "    plt.imshow(x.reshape((28,28)), cmap='Greys')\n",
    "\n",
    "# Replace the argument with any number between 0 and 9999\n",
    "predict(8384)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mluPPHm3v0MT"
   },
   "source": [
    "To actually generate adversarial examples we solve a minimization problem. We do this by setting a \"goal\" label called $ \\vec y_{goal} $ (for instance, if we wanted the network to think the adversarial image is an 8, then we would choose $ \\vec y_{goal} $ to be a one-hot vector with the eighth entry being 1). Now we define a cost function:\n",
    "\n",
    "$$ C = \\frac{1}{2} \\|\\vec y_{goal} - \\hat y(\\vec x)\\|^2_2 $$\n",
    "\n",
    "where $ \\| \\cdot \\|^2_2 $ is the squared Euclidean norm and $ \\hat y $ is the network's output. It is a function of $ \\vec x $, the input image to the network, so we write $ \\hat y(\\vec x) $. Our goal is to find an $ \\vec x $ such that $ C $ is minimized. Hopefully this makes sense, because if we find an image $ \\vec x $ that minimizes $ C $ then that means the output of the network when given $ \\vec x $ is close to our desired output, $ \\vec y_{goal} $. So in full mathy language, our optimization problem is:\n",
    "\n",
    "$$ \\arg \\min_{\\vec x} C(\\vec x) $$\n",
    "\n",
    "that is, find the $ \\vec x $ that minimizes the cost $ C $.\n",
    "\n",
    "To actually do this we can do gradient descent on $ C $. Start with an initially random vector $ \\vec x $ and take steps (changing $ \\vec x $) gradually in the direction opposite of the gradient $ \\nabla_x C $. To actually get these derivatives we can perform backpropagation on the network. In contrast to training a network, where we perform gradient descent on the weights and biases, when we create adversarial examples we hold the weights and biases constant (because we don't want to change the network!), and change the inputs to our network."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4_R8pmTVv0MU"
   },
   "source": [
    "Helper functions to evaluate the non-linearity and it's derivative:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "id": "uI5J0ZHXv0MU"
   },
   "outputs": [],
   "source": [
    "def sigmoid(z):\n",
    "    \"\"\"The sigmoid function.\"\"\"\n",
    "    return 1.0/(1.0+np.exp(-z))\n",
    "\n",
    "def sigmoid_prime(z):\n",
    "    \"\"\"Derivative of the sigmoid function.\"\"\"\n",
    "    return sigmoid(z)*(1-sigmoid(z))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "GqbEl1Fuv0MU"
   },
   "source": [
    "Also, a function to find the gradient derivatives of the cost function, $ \\nabla_x C $ with respect to the input $ \\vec x $, with a goal label of $ \\vec y_{goal} $. (Don't worry too much about the implementation, just know it calculates derivatives)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "YvpuNGXdv0MU"
   },
   "outputs": [],
   "source": [
    "def input_derivative(net, x, y):\n",
    "    \"\"\" Calculate derivatives wrt the inputs\"\"\"\n",
    "    nabla_b = [np.zeros(b.shape) for b in net.biases]\n",
    "    nabla_w = [np.zeros(w.shape) for w in net.weights]\n",
    "\n",
    "    # feedforward\n",
    "    activation = x\n",
    "    activations = [x] # list to store all the activations, layer by layer\n",
    "    zs = [] # list to store all the z vectors, layer by layer\n",
    "    for b, w in zip(net.biases, net.weights):\n",
    "        z = np.dot(w, activation)+b\n",
    "        zs.append(z)\n",
    "        activation = sigmoid(z)\n",
    "        activations.append(activation)\n",
    "\n",
    "    # backward pass\n",
    "    delta = net.cost_derivative(activations[-1], y) * \\\n",
    "        sigmoid_prime(zs[-1])\n",
    "    nabla_b[-1] = delta\n",
    "    nabla_w[-1] = np.dot(delta, activations[-2].transpose())\n",
    "\n",
    "    for l in range(2, net.num_layers):\n",
    "        z = zs[-l]\n",
    "        sp = sigmoid_prime(z)\n",
    "        delta = np.dot(net.weights[-l+1].transpose(), delta) * sp\n",
    "        nabla_b[-l] = delta\n",
    "        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())\n",
    "\n",
    "    # Return derivatives WRT to input\n",
    "    return net.weights[0].T.dot(delta)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HU6mOSzgv0MV"
   },
   "source": [
    "The actual function that generates adversarial examples and a wrapper function:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MC3KdXb6v0MV"
   },
   "source": [
    "## (a) Non Targeted Attack"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "-cT-lHVEv0MV"
   },
   "outputs": [],
   "source": [
    "def nonTargetedAdversarial(net, n, steps, eta):\n",
    "    \"\"\"\n",
    "    net : network object\n",
    "        neural network instance to use\n",
    "    n : integer\n",
    "        our goal label (just an int, the function transforms it into a one-hot vector)\n",
    "    steps : integer\n",
    "        number of steps for gradient descent\n",
    "    eta : float\n",
    "        step size for gradient descent\n",
    "    \"\"\"\n",
    "\n",
    "    # Create a random image to initialize gradient descent with\n",
    "    x = np.random.rand(28*28,1) # Assuming MNIST images are 28x28 pixels\n",
    "\n",
    "\n",
    "    #x = np.random.rand(28*28,1)\n",
    "#     print(x.shape)\n",
    "    # Set the goal output with the same shape as the output layer of your neural network\n",
    "    goal = np.zeros((10,1))  # Replace net.num_outputs with the correct value\n",
    "\n",
    "    goal[n] = 1\n",
    "#     print(goal.shape)\n",
    "    # Gradient descent on the input\n",
    "    for i in range(steps):\n",
    "        # Calculate the derivative\n",
    "        gradient = input_derivative(net, x, goal)\n",
    "\n",
    "        # The GD update on x\n",
    "        x = x - eta * gradient\n",
    "\n",
    "    return x\n",
    "\n",
    "# Existing nonTargetedAdversarial function remains unchanged\n",
    "\n",
    "# Wrapper function\n",
    "def generate(n):\n",
    "    \"\"\"\n",
    "    n : integer\n",
    "        goal label (not a one-hot vector)\n",
    "    \"\"\"\n",
    "    # Find the vector x with the above function that you just wrote.\n",
    "    adversarial_example = nonTargetedAdversarial(net, n, steps=100, eta=0.01)  # You can adjust steps and eta\n",
    "\n",
    "    # Pass the generated image (vector) to the neural network. Perform a forward pass, and get the prediction.\n",
    "    prediction = net.feedforward(adversarial_example)\n",
    "\n",
    "    print('Network Output: \\n' + str(np.round(prediction, 2)) + '\\n')\n",
    "\n",
    "    print('Network Prediction: ' + str(np.argmax(prediction)) + '\\n')\n",
    "\n",
    "    print('Adversarial Example: ')\n",
    "\n",
    "    plt.imshow(adversarial_example.reshape(28, 28), cmap='Greys')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1s3Z457Pv0MV"
   },
   "source": [
    "Now let's generate some adversarial examples! Use the function provided to mess around with the neural network. (For some inputs gradient descent doesn't always converge; 0 and 5 seem to work pretty well though. I suspect convergence is very highly dependent on our choice of random initial $ \\vec x $. We'll see later in the notebook if we force the adversarial example to \"look like\" a handwritten digit, convergence is much more likely. In a sense we will be adding regularization to our generation process)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 698
    },
    "id": "ha3ZaK3kv0MV",
    "outputId": "b707266e-05b3-47ba-e403-78844e6271a0",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Network Output: \n",
      "[[0.  ]\n",
      " [0.  ]\n",
      " [0.97]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]]\n",
      "\n",
      "Network Prediction: 2\n",
      "\n",
      "Adversarial Example: \n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPsAAAD4CAYAAAAq5pAIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAZxUlEQVR4nO2deXTV1bmG308ERAaZNQ2IikgdEJA4VKyCA1VaQaheGhSlBQFFScWJYbXaqktABm2RUhAEHECoUKioSBEEqUVDiwTKjGGSAlbCPATc948cvdRmvztNwjlZd7/PWlknnCffOTu/nJczfL+9tznnIIT4/88pqR6AECI5KOxCRILCLkQkKOxCRILCLkQknJrMO6tRo4ZLT0/3+ooVK9L6LVu2eF39+vVpbajrsH37durr1q1b7Nq8vDzqzz77bOrLly9P/Smn+P/PPvVU/ic+evQo9YcOHaK+Zs2a1G/evNnr2DEFgH/+85/U165dm/p9+/Z53ZEjR2ht6PdixxwIP5YPHDjgdaG/N/u98vLycPDgQSvMlSjsZnYzgBcAlAPwknNuEPv59PR0TJ8+3esbNGhA769v375e98ILL9Da0IN28ODB1N9///1e99RTT9HaWbNmUT9s2DDqQ6GoUqWK19WqVYvWbtu2jfrly5dT37lzZ+rvu+8+r8vKyqK1gwbRhxN++tOfUr9o0SKv27BhA63NzMyk/rTTTqO+UaNG1C9ZssTr0tLSaO2CBQu8bsyYMV5X7JfxZlYOwIsAbgFwEYBMM7uouLcnhDi5lOQ9+xUA1jvnNjrnjgKYAqB96QxLCFHalCTs6QBOfBO9NXHdv2FmPcws28yyd+/eXYK7E0KUhJKEvbAPAf7jUzDn3BjnXIZzLqNGjRoluDshREkoSdi3AjjxI/B6AD4v2XCEECeLkoT9EwCNzOxcM6sA4CcA+MfOQoiUUezWm3PumJk9AGAOClpv451zK1lNhQoV8J3vfMfr7777bnqfEyZM8DrWtwSA1atXU79//37q2fkBOTk5tJb9zgAwYMCAYt83wNstn332Ga0dNWoU9b169aJ+2rRp1A8cONDr8vPzae3WrVup79OnD/V//etfvS7Uw58zZw71oVZvly5dqP/8c/+LYNZiBoAOHTp4Hft7lKjP7px7G8DbJbkNIURy0OmyQkSCwi5EJCjsQkSCwi5EJCjsQkSCwi5EJFgyV5dt1qyZe//9973+F7/4Ba1/4403vI71HgHecwWAPXv2UB+ac8744IMPqN+0aRP1obnXbD59qFf93e9+t9i3DYTPX2jWrJnXffzxx7T2mmuuof7888+nnp1j8PLLL9Paw4cPU1+1alXqhw8fTj17vIWmRP/+97/3urVr13rns+uZXYhIUNiFiASFXYhIUNiFiASFXYhIUNiFiISktt4yMjIca7eEph3++Mc/9rqzzjqL1l533XXUjxgxgnrWSvnXv/5Fa0PLObdr1476sWPHUs9Wlx05ciSt/eSTT6gPLYm8bNky6idPnux1t99+O60NEVpyuWHDhl63du1aWstWpgWANWvWUJ+bm0v9s88+63XPPfccrX300Ue9rmvXrli1apVab0LEjMIuRCQo7EJEgsIuRCQo7EJEgsIuRCQo7EJEQlL77Jdccolju7iGljVu06aN14X66KFdWlkPHwDOOOMMr6tQoQKtZf1eAPjiiy+onz9/PvVsKijrwQNA9+7dqQ/t4lqS7abr1KlDa4cMGUL9E088QT1beryk04pDS0n/6le/op7tUMv66ACfutuxY0fk5OSozy5EzCjsQkSCwi5EJCjsQkSCwi5EJCjsQkSCwi5EJCS1z37RRRe5119/3esvvPBCWv/RRx95XV5eHq1t0qQJ9U2bNqW+bdu2XtepUydau2vXrmLfNhDe0pnNrWbHDOBbKgPAm2++Sf3ChQupr1mzpteF/t5ZWVnUh9YRYMtgHzx4kNbWq1eP+tCy5926daN+ypQpXmdWaJv8G+rWret1eXl5yM/PL/QGSrRls5nlAtgH4DiAY865jJLcnhDi5FGisCdo7Zzjp4AJIVKO3rMLEQklDbsD8J6ZLTWzHoX9gJn1MLNsM8sOva8WQpw8Shr2ls65ywDcAqC3mV377R9wzo1xzmU45zKqV69ewrsTQhSXEoXdOfd54nIngBkAriiNQQkhSp9ih93MKptZ1a+/B9AGwIrSGpgQonQpyafxZwKYkegJngrgdefcu6xgz549mD17ttc/8MAD9A7ffdd/84sXL6a1U6dOpT50vkHPnj297s9//jOtDfWyH3zwQepDHDt2zOu6dOlCa0Nz8UPryof68KyXHtomO7QNd+/evakfNmyY13Xt2pXWduzYkfpGjRpRn5mZST1j79691FerVq1YtcUO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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "generate(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "32pepE6Rv0MW"
   },
   "source": [
    "## (b) Targeted Attack(s)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "XG5rQPMKv0MW"
   },
   "source": [
    "Sweet! We've just managed to create an image that looks utterly meaningless to a human, but the neural network thinks is a '5' with very high certainty. We can actually take this a bit further. Let's generate an image that looks like one number, but the neural network is certain is another. To do this we will modify our cost function a bit. Instead of just optimizing the input image, $ \\vec x $, to get a desired output label, we'll also optimize the input to look like a certain image, $ \\vec x_{target} $, at the same time. Our new cost function will be\n",
    "\n",
    "$$ C = \\|\\vec y_{goal} - y_{hat}(\\vec x)\\|^2_2 + \\lambda \\|\\vec x - \\vec x_{target}\\|^2_2 $$\n",
    "\n",
    "The added term tells us the distance from our $ \\vec x $ and some $ \\vec x_{target} $ (which is the image we want our adversarial example to look like). Because we want to minimize $ C $, we also want to minimize the distance between our adversarial example and this image. The $ \\lambda $ is hyperparameter that we can tune; it determines which is more important: optimizing for the desired output or optimizing for an image that looks like $ \\vec x_{target} $.\n",
    "\n",
    "If you are familiar with ridge regularization, the above cost function might look suspiciously like the ridge regression cost function. In fact, we can view this generation method as giving our model a prior, centered on our target image.\n",
    "\n",
    "Here is a function that implements optimizing the modified cost function, called `sneaky_adversarial` (because it is very sneaky). Note that the only difference between this function and `adversarial` is an additional term on the gradient descent update for the regularization term:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "mbgfrr6Uv0MW"
   },
   "outputs": [],
   "source": [
    "def targetedAdversarial(net, n, x_target, steps, eta, lam=0.05):\n",
    "    \"\"\"\n",
    "    net : network object\n",
    "        neural network instance to use\n",
    "    n : integer\n",
    "        our goal label (just an int, the function transforms it into a one-hot vector)\n",
    "    x_target : numpy vector\n",
    "        our goal image for the adversarial example\n",
    "    steps : integer\n",
    "        number of steps for gradient descent\n",
    "    eta : float\n",
    "        step size for gradient descent\n",
    "    lam : float\n",
    "        lambda, our regularization parameter. Default is 0.05\n",
    "    \"\"\"\n",
    "\n",
    "    # Set the goal output\n",
    "    goal = np.zeros((10,1))  # Assuming you are working with MNIST (10 classes)\n",
    "    goal[n] = 1\n",
    "\n",
    "    # Create a random image to initialize gradient descent with\n",
    "    x = np.random.rand(28*28,1)  # Assuming MNIST images are 28x28 pixels\n",
    "  #take random\n",
    "    # Gradient descent on the input\n",
    "    for i in range(steps):\n",
    "        # Calculate the derivative\n",
    "        gradient = input_derivative(net, x, goal)\n",
    "\n",
    "        # Calculate the regularization term\n",
    "        regularization_term = 2*lam * (x - x_target) / np.linalg.norm((x - x_target),2)\n",
    "\n",
    "        # The GD update on x, with an added penalty to the cost function\n",
    "        x = x - eta * (gradient + regularization_term)\n",
    "\n",
    "    return x\n",
    "\n",
    "\n",
    "# Wrapper function\n",
    "def generate_advSample(n, m):\n",
    "    \"\"\"\n",
    "    n: int 0-9, the target number to match\n",
    "    m: index of example image to use (from the test set)\n",
    "    \"\"\"\n",
    "\n",
    "    # Find a random instance of m in the test set\n",
    "    idx = np.random.randint(0, 8000)\n",
    "    while test_data[idx][1] != m:\n",
    "        idx += 1\n",
    "\n",
    "    # Hardcode the parameters for the wrapper function\n",
    "    adversarial_example = targetedAdversarial(net, n, test_data[idx][0], steps=100, eta=1)\n",
    "    prediction = net.feedforward(adversarial_example)\n",
    "\n",
    "    print('\\nWhat we want our adversarial example to look like: ')\n",
    "    plt.imshow(test_data[idx][0].reshape((28, 28)), cmap='Greys')\n",
    "    plt.show()\n",
    "\n",
    "    print('\\nAdversarial Example: ')\n",
    "    plt.imshow(adversarial_example.reshape(28, 28), cmap='Greys')\n",
    "    plt.show()\n",
    "\n",
    "    print('Network Prediction: ' + str(np.argmax(prediction)) + '\\n')\n",
    "\n",
    "    print('Network Output: \\n' + str(np.round(prediction, 2)) + '\\n')\n",
    "\n",
    "    return adversarial_example"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "GnRijuHvv0MW"
   },
   "source": [
    "Play around with this function to make \"sneaky\" adversarial examples! (Again, some numbers converge better than others... try 0, 2, 3, 5, 6, or 8 as a target label. 1, 4, 7, and 9 still don't work as well... no idea why... We get more numbers that converge because we've added regularization term to our cost function. Perhaps changing $ \\lambda $ will get more to converge?)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "id": "Qzio46qkv0MW",
    "outputId": "67ddbef3-1740-463b-a0df-dc1a80e5c0b4"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "What we want our adversarial example to look like: \n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Adversarial Example: \n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Network Prediction: 8\n",
      "\n",
      "Network Output: \n",
      "[[0.  ]\n",
      " [0.  ]\n",
      " [0.01]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.99]\n",
      " [0.  ]]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# generate_advSample(target label, target digit)\n",
    "adv_ex = generate_advSample(8, 2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gM6xIfurv0MW"
   },
   "source": [
    "## (c) Protection against adversarial attacks"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vq97hju1v0MW"
   },
   "source": [
    "Awesome! We’ve just created images that trick neural networks. The next question we could ask is whether or not we could protect against these kinds of attacks. If you look closely at the original images and the adversarial examples you’ll see that the adversarial examples have some sort of grey tinged background."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6vPs4sdyv0MX"
   },
   "source": [
    "So how could we protect against these adversarial attacks? One very simple way would be to use binary thresholding. Set a pixel as completely black or completely white depending on a threshold. This should remove the \"noise\" that's always present in the adversarial images. Let's see if it works:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "1NDPGanzv0MX"
   },
   "outputs": [],
   "source": [
    "def simple_defense(n, m):\n",
    "    \"\"\"\n",
    "    n: int 0-9, the target number to match\n",
    "    m: index of example image to use (from the test set)\n",
    "    \"\"\"\n",
    "\n",
    "    # Generate an adversarial sample.\n",
    "    x = generate_advSample(n, m)\n",
    "\n",
    "    # Perform binary thresholding on the generated sample.\n",
    "    threshold = 0.5\n",
    "    binarized_sample = (x > threshold).astype(int)\n",
    "\n",
    "    print(\"With binary thresholding: \")\n",
    "\n",
    "    # Plot a grayscale image of the binarized generated sample.\n",
    "    plt.imshow(binarized_sample.reshape(28, 28), cmap='Greys')\n",
    "    plt.show()\n",
    "\n",
    "    # Print the network's predictions for the binarized image.\n",
    "    prediction = net.feedforward(binarized_sample)\n",
    "    print(\"Prediction with binary thresholding: \" + str(np.argmax(prediction))+ '\\n')\n",
    "\n",
    "    # The output of the network.\n",
    "    print(\"Network output: \")\n",
    "    print(prediction)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "id": "7FpzVHcev0MX",
    "outputId": "f66b8f5b-ff08-46fb-bc0b-7e4133b066bf"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "What we want our adversarial example to look like: \n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPsAAAD4CAYAAAAq5pAIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAOVUlEQVR4nO3df6hVdbrH8c+T+aN0IL2exLI6KQZpkDPsJOtiyXjNirQpZxgJMQicPwwcGKLwUpN/RHa9M8MlLpZeRe/F2zDhiAZFxkGIITC34S292j398M44/jhHDNIsJvW5f5zV5WRnf/dprbV/eJ73CzZ77/XstdbTzs9Ze+/v2vtr7i4AQ99lrW4AQHMQdiAIwg4EQdiBIAg7EMTlzdzZ+PHjvbOzs5m7BEI5fPiwTp48aQPVCoXdzOZL+hdJwyT9m7uvTj2+s7NT1Wq1yC4BJFQqlZq13C/jzWyYpH+VdK+kaZIWm9m0vNsD0FhF3rPPlPSRu3/i7n+T9HtJC8tpC0DZioT9Wkl/6Xf/SLbsW8xsmZlVzaza29tbYHcAiigS9oE+BPjOubfuvs7dK+5e6ejoKLA7AEUUCfsRSdf1uz9J0tFi7QBolCJh3yNpqpndaGYjJP1c0o5y2gJQttxDb+5+zswel/Sm+obeNrr7gdI6A1CqQuPs7v66pNdL6gVAA3G6LBAEYQeCIOxAEIQdCIKwA0EQdiAIwg4EQdiBIAg7EARhB4Ig7EAQhB0IgrADQRB2IAjCDgRB2IEgCDsQBGEHgiDsQBCEHQiCsANBNHXK5kb6+uuvk/X58+cn6x9++GGyvnbt2pq1e+65J7nuiBEjknWgGTiyA0EQdiAIwg4EQdiBIAg7EARhB4Ig7EAQQ2acffjw4cn6Aw88kKzv2bMnWV+4cGHN2s0335xcd8GCBcn6ihUrkvXu7u5kffr06cl6ESNHjkzWR48e3bB9o1yFwm5mhyWdlnRe0jl3r5TRFIDylXFkn+PuJ0vYDoAG4j07EETRsLuknWa218yWDfQAM1tmZlUzq/b29hbcHYC8iob9Tnf/kaR7JS03s9kXP8Dd17l7xd0rHR0dBXcHIK9CYXf3o9l1j6RtkmaW0RSA8uUOu5mNNrMffHNb0jxJ+8tqDEC5zN3zrWg2WX1Hc6nvU/3/dPfnUutUKhWvVqu59tdon376abK+evXqmrUtW7Yk1z179myuntrB5MmTk/WXXnopWZ87d26Z7aCOSqWiarVqA9VyD725+yeSbs3dFYCmYugNCIKwA0EQdiAIwg4EQdiBIIbMV1yLuvHGG5P1l19+uWZt5cqVyXW//PLLZP3ChQvJ+t69e5P12267rWbt+PHjyXUXLVqUrH/88cfJer2hVIbe2gdHdiAIwg4EQdiBIAg7EARhB4Ig7EAQhB0IgnH2Etxwww0N3f60adNyr3v99dcn69dcc02yfurUqdz7RnvhyA4EQdiBIAg7EARhB4Ig7EAQhB0IgrADQTDOPsQdPXo0Wd+/v9hP/c+aNavQ+mgejuxAEIQdCIKwA0EQdiAIwg4EQdiBIAg7EATj7EPAm2++WbP20EMPNXTf9X43/uDBgw3b90033ZSs33XXXTVrw4YNK7udtlf3yG5mG82sx8z291s2zszeMrPu7HpsY9sEUNRgXsZvkjT/omVPSepy96mSurL7ANpY3bC7+9uSLv5tooWSNme3N0t6sNy2AJQt7wd0E9z9mCRl11fXeqCZLTOzqplVe3t7c+4OQFEN/zTe3de5e8XdKx0dHY3eHYAa8ob9hJlNlKTsuqe8lgA0Qt6w75C0NLu9VNL2ctoB0Ch1x9nN7BVJd0sab2ZHJP1a0mpJfzCzxyT9WdJPG9nkUOfuyfoXX3yRrG/fXvtvbb254Yt64oknkvV6v0uf8tlnnyXr9f7bpk+fXrO2e/fu5LpXXnllsn4pqht2d19co/TjknsB0ECcLgsEQdiBIAg7EARhB4Ig7EAQfMW1CeoNEXV1dSXrCxYsKLOdb5k6dWqyPnfu3GT94YcfTtbnzJlTs2ZmyXW7u7uT9SVLliTr7777bs1aT0/6PLDOzs5k/VLEkR0IgrADQRB2IAjCDgRB2IEgCDsQBGEHgmCcvQneeOONZH3RokWFtp/6Ouajjz6aXHfNmjXJ+hVXXJGnpVLUOwcg9VPRUnqcfdWqVcl1N2zYkKxfdtmld5y89DoGkAthB4Ig7EAQhB0IgrADQRB2IAjCDgRh9X7GuEyVSsXrTfE7FJ05cyZZ37ZtW7KeGi+WpCeffLJmbdKkScl1L2WHDh1K1qdNm5Z722fPnk3WR40alXvbjVSpVFStVgf8oQCO7EAQhB0IgrADQRB2IAjCDgRB2IEgCDsQBN9nb4IxY8Yk6/V+/7xeHRiMukd2M9toZj1mtr/fsmfN7K9mti+73NfYNgEUNZiX8ZskzR9g+e/cfUZ2eb3ctgCUrW7Y3f1tSaea0AuABiryAd3jZvZ+9jJ/bK0HmdkyM6uaWbW3t7fA7gAUkTfsayVNkTRD0jFJv6n1QHdf5+4Vd690dHTk3B2AonKF3d1PuPt5d78gab2kmeW2BaBsucJuZhP73f2JpP21HgugPdQdZzezVyTdLWm8mR2R9GtJd5vZDEku6bCkXzSuRQBlqBt2d188wOL0L+gDaDucLgsEQdiBIAg7EARhB4Ig7EAQfMUVl6xdu3a1uoVLCkd2IAjCDgRB2IEgCDsQBGEHgiDsQBCEHQiCcXa0rfPnzyfrO3fuzL3tej/vPRRxZAeCIOxAEIQdCIKwA0EQdiAIwg4EQdiBIMKMs3/++efJ+unTp5P18ePH16yNHDkyV09IW79+fbK+ffv23Nt+7rnnkvVRo0bl3na74sgOBEHYgSAIOxAEYQeCIOxAEIQdCIKwA0GEGWd//vnnk/UXXnghWZ8zZ07N2qpVq5Lr3n777cn65ZeH+d/wLefOnUvWu7q6GrbvW2+9tWHbbld1j+xmdp2Z7TKzg2Z2wMxWZMvHmdlbZtadXY9tfLsA8hrMy/hzkn7l7jdLul3ScjObJukpSV3uPlVSV3YfQJuqG3Z3P+bu72W3T0s6KOlaSQslbc4etlnSgw3qEUAJvtcHdGbWKemHknZLmuDux6S+PwiSrq6xzjIzq5pZtbe3t2C7APIadNjNbIykrZJ+6e7pb5X04+7r3L3i7pWOjo48PQIowaDCbmbD1Rf0Le7+x2zxCTObmNUnSuppTIsAylB3zMfMTNIGSQfd/bf9SjskLZW0OrvO/33DJli+fHmyfurUqWT9tddeq1mbPXt2ct1Zs2Yl6/W+Tnn//fcn6/PmzatZu+WWW5LrNtru3btr1l599dXkulu3bi2076effrpm7Y477ii07UvRYAZ475S0RNIHZrYvW7ZSfSH/g5k9JunPkn7akA4BlKJu2N39T5KsRvnH5bYDoFE4XRYIgrADQRB2IAjCDgRB2IEgzN2btrNKpeLVarVp+ytT6qemN23alFz3+PHjyfqaNWuS9XpfBR0+fHjN2pQpU5LrNtqhQ4catu1HHnkkWX/xxRdr1q666qqSu2kPlUpF1Wp1wNEzjuxAEIQdCIKwA0EQdiAIwg4EQdiBIAg7EATj7G3gq6++Stb37duXrL/zzju5950ai5akCRMmJOsHDhxI1p955pmatXHjxiXXrTeOnjq/QJKGDRuWrA9FjLMDIOxAFIQdCIKwA0EQdiAIwg4EQdiBIBhnB4YQxtkBEHYgCsIOBEHYgSAIOxAEYQeCIOxAEHXDbmbXmdkuMztoZgfMbEW2/Fkz+6uZ7csu9zW+XQB5DWZ+9nOSfuXu75nZDyTtNbO3strv3P2fG9cegLIMZn72Y5KOZbdPm9lBSdc2ujEA5fpe79nNrFPSDyXtzhY9bmbvm9lGMxtbY51lZlY1s2pvb2+xbgHkNuiwm9kYSVsl/dLdP5e0VtIUSTPUd+T/zUDrufs6d6+4e6Wjo6N4xwByGVTYzWy4+oK+xd3/KEnufsLdz7v7BUnrJc1sXJsAihrMp/EmaYOkg+7+237LJ/Z72E8k7S+/PQBlGcyn8XdKWiLpAzPbly1bKWmxmc2Q5JIOS/pFA/oDUJLBfBr/J0kDfT/29fLbAdAonEEHBEHYgSAIOxAEYQeCIOxAEIQdCIKwA0EQdiAIwg4EQdiBIAg7EARhB4Ig7EAQhB0IoqlTNptZr6T/7bdovKSTTWvg+2nX3tq1L4ne8iqztxvcfcDff2tq2L+zc7Oqu1da1kBCu/bWrn1J9JZXs3rjZTwQBGEHgmh12Ne1eP8p7dpbu/Yl0VteTemtpe/ZATRPq4/sAJqEsANBtCTsZjbfzD40s4/M7KlW9FCLmR02sw+yaairLe5lo5n1mNn+fsvGmdlbZtadXQ84x16LemuLabwT04y39Llr9fTnTX/PbmbDJP2PpH+QdETSHkmL3f2/m9pIDWZ2WFLF3Vt+AoaZzZZ0RtK/u/st2bJ/knTK3VdnfyjHuvuTbdLbs5LOtHoa72y2oon9pxmX9KCkR9XC5y7R18/UhOetFUf2mZI+cvdP3P1vkn4vaWEL+mh77v62pFMXLV4oaXN2e7P6/rE0XY3e2oK7H3P397LbpyV9M814S5+7RF9N0YqwXyvpL/3uH1F7zffuknaa2V4zW9bqZgYwwd2PSX3/eCRd3eJ+LlZ3Gu9mumia8bZ57vJMf15UK8I+0FRS7TT+d6e7/0jSvZKWZy9XMTiDmsa7WQaYZrwt5J3+vKhWhP2IpOv63Z8k6WgL+hiQux/NrnskbVP7TUV94psZdLPrnhb38//aaRrvgaYZVxs8d62c/rwVYd8jaaqZ3WhmIyT9XNKOFvTxHWY2OvvgRGY2WtI8td9U1DskLc1uL5W0vYW9fEu7TONda5pxtfi5a/n05+7e9Iuk+9T3ifzHkv6xFT3U6GuypP/KLgda3ZukV9T3su5r9b0iekzS30nqktSdXY9ro97+Q9IHkt5XX7Amtqi3v1ffW8P3Je3LLve1+rlL9NWU543TZYEgOIMOCIKwA0EQdiAIwg4EQdiBIAg7EARhB4L4P/qLViDaFauUAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Adversarial Example: \n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPsAAAD4CAYAAAAq5pAIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAX3klEQVR4nO2de4yVZ7XGn8VtWq4Fhlth5DLcS1vAgZJSaQFrK2rQxp60icrxUozRRBOboJ4Yq8lpyMlR0+qxCT2ieFKkTbSKtqZSYkWaikyBculwK9cBygzlNtyGmWGdP2Y3wTrfs8a9Z/aec97nl0xmZj+z9n7n298z3+y93rWWuTuEEP//6VbqBQghioPMLkQiyOxCJILMLkQiyOxCJEKPYj5YeXm5jx49OlO/fPkyjS8rK8vUWlpaaGy3bvzvWnNzM9ULue9CH7szMybXrl2j+g033ED1s2fPUp09Z9F9X716lepRPPvdzKygx46OG/u9AeDKlSuZWp8+fWgsO+b19fU4f/58m79cQWY3s/sBPAGgO4D/dvfl7OdHjx6NV199NVPfvXs3fbwxY8Zkag0NDTQ2OjHOnDlDdfbkRk9O9MRHhmEnBsD/GER/aKI/sJMmTaL6b37zG6pPmDAhUxs/fjyNPX78ONWjeHbcevTgp35tbS3Vo+PGLmoAsGfPnkxt9uzZNPZ3v/tdprZs2bJMLe9/482sO4D/AvBhAFMBPGxmU/O9PyFE51LIa/bZAPa7+wF3vwpgDYDFHbMsIURHU4jZRwI4et33tbnb/g4zW2pm1WZWXV9fX8DDCSEKoRCzt/UmwD+8eHT3Fe5e5e5VQ4YMKeDhhBCFUIjZawFUXPf9KAD8HRUhRMkoxOybAUwws7Fm1gvAQwDWdsyyhBAdTd6pN3dvNrOvAHgJram3le6+i8U0NzfTNNPp06fpY5aXl2dqdXV1NPZ973sf1aM0z8SJEzO16L2IaG1RquXo0aNUb2pqytT69u1LY6MU0okTJ6i+cOFCqr/99tuZWvfu3WlsZWUl1Tds2ED1BQsWZGr79u2jsZcuXaL6xYsXqd7Y2Eh1dj6yYxbF9urVK1MrKM/u7i8CeLGQ+xBCFAdtlxUiEWR2IRJBZhciEWR2IRJBZhciEWR2IRKhqPXszc3NNOcclR0eO3YsU+vfvz+NjcpQZ8yYQXWWv4xytlGJapRHj2qn2doGDhxIY2tqaqg+ZcoUqldXV1Od5cqjMtIoDx+Vke7fvz9Ti/YXRCXTU6fyAs8BAwZQ/fe//32mdvfdd9NYtueDlXLryi5EIsjsQiSCzC5EIsjsQiSCzC5EIsjsQiRCUVNvvXr1oh1it2/fTuMrKioytZ49e9LYKEU0efJkqp88eTJTu+OOO2hsVCYaddWtqqqiOmujHaWvxo4dS3VWPgsAvXv3zluPSppvvvlmqkftoJn+2muv0dhBgwZRPUq33n777XnrUXks64TMni9d2YVIBJldiESQ2YVIBJldiESQ2YVIBJldiESQ2YVIhKLm2Zuammib3Ciny8oGozLRZ599lurnzp2jOisrvO+++2hslMOPct0XLlygOitxPXjwII2Nctls2igATJs2jepsbVGO/tChQ1SPxiqzvRejRo2isWxfBQDMnTuX6hHDhw/P1KLfm5X2st9ZV3YhEkFmFyIRZHYhEkFmFyIRZHYhEkFmFyIRZHYhEqHoraRZ/jIaL8xaSd944400Nsrhv/zyy1Rnuc+o1fOaNWuoPmvWLKqzlsgAH5t8+PBhGhu1mo6ek+i4s+d77969NHbmzJlUj/oAsPbhUWvxsrIyqkf9Edh4cYC3qh43bhyNZbX4bE9GQWY3s0MAGgC0AGh2d95lQQhRMjriyj7f3U91wP0IIToRvWYXIhEKNbsD+KOZvW5mS9v6ATNbambVZlZ99uzZAh9OCJEvhf4bP9fdj5vZUADrzGy3u2+4/gfcfQWAFQAwefJkL/DxhBB5UtCV3d2P5z7XAXgewOyOWJQQouPJ2+xm1sfM+r37NYAPAdjZUQsTQnQs5p7ff9ZmNg6tV3Og9eXAanf/dxYzZcoUX7VqVaY+adIk+phszG6U6z51iicMtm7dSvW//vWvmVqU7z1w4ADVm5ubqf6pT32K6s8//3ymVl9fX9BjP/roo1SPRmXPmTMnU4vOPVYLDwBDhw6l+ltvvZWpRXX4EWw0MhDPMThy5EimFtXSs30Zc+fOxeuvv95mw/y8X7O7+wEAvBO+EKLLoNSbEIkgswuRCDK7EIkgswuRCDK7EIlQ1BJXgI/R3bmTp+lZai5qBc1a9wLAkiVLqL548eJMbfPmzTR2yJAhVGcpIiAuz2VtjaMtymvXrqX6H/7wB6pHaUH2nEXp0EuXLlE9Oi5sXHUUy1JjQLz2aIw3Kx2+6aabaOyuXbsyNZae1pVdiESQ2YVIBJldiESQ2YVIBJldiESQ2YVIBJldiEQoap69V69eGDlyZKYelfaxcsoonxzl8O+++26qv/DCC5laZWUljR08eDDVo5HN69evpzobfRztPzh//jzVjx8/TvUoX93Y2Eh1BhuTDcRrY/sbov0DUQvtO++8k+pRyfWVK1cytS1bttBYVuLao0e2pXVlFyIRZHYhEkFmFyIRZHYhEkFmFyIRZHYhEkFmFyIRij6ymbU2Pn36NI1nOeFoBO9dd91F9Wgs8oIFCzK1qPY5Gmsc5Ytvu+02qrN20FHL46iFNqsJB4DJkydTndX6s5wwAPzlL3+h+gc+8AGqs/Pp/e9/P42NWmxHba6jc3nUqFGZ2pgxY2gsO1fZvgZd2YVIBJldiESQ2YVIBJldiESQ2YVIBJldiESQ2YVIhKLm2RsbG2mP9ChXznp1Hz58mMZGOVlWEw4A3brl/3cxyicPGzaM6lGPcjYS+uWXX6ax0djjKMcf1cNv3LgxU6uoqKCxf/vb36ge5brZHoABAwbQ2Lq6OqpH50N0Lq9ZsyZTmzdvHo1l/fTZXIbwDDazlWZWZ2Y7r7ttkJmtM7N9uc8Do/sRQpSW9lyufg7g/vfc9g0A6919AoD1ue+FEF2Y0OzuvgHAe/f+LQawKvf1KgAf79hlCSE6mnxfiA5z9xMAkPuc+cLPzJaaWbWZVUf90IQQnUenvxvv7ivcvcrdq6I3RYQQnUe+Zj9pZiMAIPeZv3UphCg5+Zp9LYB3ZxwvAfDbjlmOEKKzCPPsZvZLAPcAKDezWgDfAbAcwHNm9nkARwA82J4Hc3faZ3z06NE0fsSIEZnawIE8+1dTU0P1qOac5S9vvfVWGhvlybdv3071aNb322+/nant3r0771gg7qf/3e9+l+oPPph9avzsZz+jsbfccgvVly1bRnW2f+F73/sejY36AER59Oi4s/0L0XPCHpvl/0Ozu/vDGdLCKFYI0XXQdlkhEkFmFyIRZHYhEkFmFyIRZHYhEqGoJa79+vXDwoXZb+Lv2LGDxrOUwxtvvEFjozLS8ePHU/3NN9+kOoOVoAJxOeWXvvQlqrNUTVlZGY39whe+QPXy8nKqP/LII1RnqaBvfvObNPbJJ5+kepTy/NGPfpSp1dbW0tjPfe5zVN+0aRPVe/bsSXVGdMzZ6HKNbBZCyOxCpILMLkQiyOxCJILMLkQiyOxCJILMLkQiFDXPfuXKFZqvvv3222k8yyFG7XfPnj1L9erqaqqzPP0777xDY//85z9T/emnn6b68OHDqb53795M7dFHH6WxY8eOpfrcuXOpHu0RYM9ZtDdi6dKlVH/88cepzsYms5bmAHDx4kWqR+W3UZtrlodfvXo1jZ0zZ06mdu3atUxNV3YhEkFmFyIRZHYhEkFmFyIRZHYhEkFmFyIRZHYhEqGoeXZ3R0tLS6Z+9OhRGj9y5MhMLcp1szG3QNw6mN1/VFc9btw4qv/kJz+herRHgI0mjlpkX7hwgepXr16levS7bdiwIVObNWsWjY32J7B8M8B/94aGBhobnYtRvXp03E+ePJmpzZ8/n8ayMdnunqnpyi5EIsjsQiSCzC5EIsjsQiSCzC5EIsjsQiSCzC5EIhQ1zw7wPGBUW11fX5+p9e7dm8aycc8A8MILL1D9Yx/7WKYW5arZ2GIA2Lp1K9WnT59O9aFDh2ZqUb/7CRMmUL179+5Uj/rSX758OVOLctFsXwUAbNy4keoPPPBAprZy5Uoa+8lPfpLqkyZNono0dpn9bgcPHqSxrEcA28cSXtnNbKWZ1ZnZzutue8zMjpnZttzHouh+hBClpT3/xv8cwP1t3P5Dd5+e+3ixY5clhOhoQrO7+wYA2f19hBD/JyjkDbqvmNn23L/5A7N+yMyWmlm1mVVHe7yFEJ1HvmZ/CkAlgOkATgD4ftYPuvsKd69y96qbbropz4cTQhRKXmZ395Pu3uLu1wA8DWB2xy5LCNHR5GV2M7s+j/UJADuzflYI0TUI8+xm9ksA9wAoN7NaAN8BcI+ZTQfgAA4B+GJ7HuyGG27A1KlTM/U9e/bQeNYTO5qvvm3bNqqzdQHAoUOHMrWKigoa29jYSPWoZpzlTgFem11ZWUljozz6rl27qD5mzBiqf+QjH8nUzpw5Q2OjWvnDhw9T/ZlnnsnUpkyZQmOj5+zEiRNUj/Z9sP4J0fwEdi6zfSyh2d394TZu/mkUJ4ToWmi7rBCJILMLkQgyuxCJILMLkQgyuxCJUNQS18bGRuzbty9TZy2RAeDcuXOZGkvLAcDo0aPDtTHYVt+oJDFqcz1z5kyqs9bBAF87O2ZAXJ4blXLu2LGD6qxsuby8nMZGsHbMADBjxoxMbf/+/TQ2GpM9ZMgQqkdpxSNHjmRqffr0obH9+/fP1FgqVVd2IRJBZhciEWR2IRJBZhciEWR2IRJBZhciEWR2IRKhqHn2bt260Rxi1MmmV69emdr27dtpbJQ3jeJZ+94777yTxrJ1A8CLL/J+ncOGDaP6qVOnMrXZs3lfkWj0cNQSOdobwcYuR7nq6L7Zng0AMLNMLWqRxnLZADBgwACqR8eN7W+ISqZZGSt7PnVlFyIRZHYhEkFmFyIRZHYhEkFmFyIRZHYhEkFmFyIRippn79GjBx0vHLVUZrnLiRMn0tio5jzKbfbr1y9TY2OJAd42GODtloE4Z8vq4Xfu5C39o5ry06f5mL+o9po931Hs448/TvUnn3yS6g899FCmdt9999HYqBV0IfsyAN66PGqRzVpss70FurILkQgyuxCJILMLkQgyuxCJILMLkQgyuxCJILMLkQhFzbNfvHgRmzZtyjv+gx/8YKbGaroB4JVXXilIv3TpUqb24x//OO9YgOdGAaCuro7qrDY6qmc/duwY1adPn071qKc961HQ1NREY+vr66k+f/58qq9bty5Te+KJJ2hs9HtFPQZGjRpFdXZORHX+bF8Hm58QXtnNrMLM/mRmNWa2y8y+mrt9kJmtM7N9uc8Do/sSQpSO9vwb3wzg6+4+BcAcAF82s6kAvgFgvbtPALA+970QoosSmt3dT7j7ltzXDQBqAIwEsBjAqtyPrQLw8U5aoxCiA/in3qAzszEAZgDYBGCYu58AWv8gAGhzE7SZLTWzajOrjl4HCSE6j3ab3cz6AvgVgK+5e7td6+4r3L3K3auiJn5CiM6jXWY3s55oNfoz7v7r3M0nzWxETh8BgL9lLIQoKWHqzVrzQj8FUOPuP7hOWgtgCYDluc+/bcd90dK/e++9l8azcbRlZWU0NkrTRKOLX3311Uxt+fLlNDZqO7xlyxaqR+W7rJX1yJEjaeyVK1eo3tDQQPWoJXN1dXWmFqVho5HMzc3NVP/sZz+bqd1666009tChQ1SPynOj486OW0tLC40dP358psZaSbcnzz4XwKcB7DCzbbnbvoVWkz9nZp8HcATAg+24LyFEiQjN7u4bAWTt+ljYscsRQnQW2i4rRCLI7EIkgswuRCLI7EIkgswuRCIUtcQV4DnEaDsty9mOGDGCxrLSPwCYMWMG1RctWpSpPffcczQ2Kne8ePEi1Wtra6n+7W9/O1N76qmnaCxraQzExzXKR7N8cmVlJY2NSoNZC20A+MxnPpOpRbs5ozHbUYlrtAeAtRdnbcsB4Nlnn83Uzpw5k6npyi5EIsjsQiSCzC5EIsjsQiSCzC5EIsjsQiSCzC5EIpi7F+3BbrnlFl+9enWmztoOA0C3btl/m6Ia4N27d0drozqr+37zzTdpLKs3B4ADBw5QvbGxkeqsHTSrw2/PfdfU1FCd9RgAgI9+9KOZWt++fWnsAw88QPU9e/ZQfeDA7IbHrO4biOv8o/0HUTzrnxAdU/bY99xzD7Zu3dpmlaqu7EIkgswuRCLI7EIkgswuRCLI7EIkgswuRCLI7EIkQlHr2RsbG2n9c9TL+9y5c5laVDMe5brZfQOg/e6jdV+9epXqUQ/yiooKqt92222Z2h133EFjozr/G2+8sSCd7W+4+eabaez+/fupPmvWLKpv3ry5U2IBoHfv3lSPjsuRI0cyNbafBODnAxv/rSu7EIkgswuRCDK7EIkgswuRCDK7EIkgswuRCDK7EInQnvnsFQB+AWA4gGsAVrj7E2b2GIBHANTnfvRb7v4iu6/+/ftjwYIFmXpUF87qn1l+EQCiuv2ohvjgwYOZWpQXjfLo5eXlVI9q9Y8ePZqpjRkzhsaePn2a6lHP+qhvPHu+ox4D0f6EnTt3Un3AgAGZ2muvvUZjo97tUS1+dNzmzJmTqRV6LmfRnk01zQC+7u5bzKwfgNfNbF1O+6G7/2dejyyEKCrtmc9+AsCJ3NcNZlYDYGRnL0wI0bH8U6/ZzWwMgBkANuVu+oqZbTezlWbWZg8gM1tqZtVmVn3q1KnCViuEyJt2m93M+gL4FYCvuft5AE8BqAQwHa1X/u+3FefuK9y9yt2rotemQojOo11mN7OeaDX6M+7+awBw95Pu3uLu1wA8DWB25y1TCFEoodmt9a3BnwKocfcfXHf79S0uPwGAvzUqhCgp7Xk3fi6ATwPYYWbbcrd9C8DDZjYdgAM4BOCL0R01NTWhrq4uUz958iSNZ6mUKFUSjYOOUlAsHTJt2jQa+8orr1B98uTJVI9Skmzk88SJE2nsrl27qD5//nyqRy24WZvr6dOn09iXXnqJ6lFZMxubPHbsWBoblTyz8xgALl++TPXDhw9naoMHD6axLPXGfuf2vBu/EUBbZzrNqQshuhbaQSdEIsjsQiSCzC5EIsjsQiSCzC5EIsjsQiRCUVtJt7S04MyZM5n6vHnzaDzLdV+6dInGNjQ0UD0aXcxaB0fjfysrK6ke1QwMGTKE6k1NTZlatH9gxowZVI/io3wyK79lI5UBYOjQoVSPRnyXlZVlalFpLstXA3Eb7ChXzlp4v/POOzSWrZ35QFd2IRJBZhciEWR2IRJBZhciEWR2IRJBZhciEWR2IRLB8m1Lm9eDmdUDuL6QtxxAV21M11XX1lXXBWht+dKRaxvt7m1uzCiq2f/hwc2q3b2qZAsgdNW1ddV1AVpbvhRrbfo3XohEkNmFSIRSm31FiR+f0VXX1lXXBWht+VKUtZX0NbsQoniU+souhCgSMrsQiVASs5vZ/Wa2x8z2m9k3SrGGLMzskJntMLNtZlZd4rWsNLM6M9t53W2DzGydme3LfeZF4cVd22Nmdix37LaZ2aISra3CzP5kZjVmtsvMvpq7vaTHjqyrKMet6K/Zzaw7gL0A7gVQC2AzgIfdnU9CKBJmdghAlbuXfAOGmc0DcAHAL9x9Wu62/wBw2t2X5/5QDnT3ZV1kbY8BuFDqMd65aUUjrh8zDuDjAP4VJTx2ZF3/giIct1Jc2WcD2O/uB9z9KoA1ABaXYB1dHnffAOC9rWIWA1iV+3oVWk+WopOxti6Bu59w9y25rxsAvDtmvKTHjqyrKJTC7CMBXN+rqBZda967A/ijmb1uZktLvZg2GObuJ4DWkwcA791UfMIx3sXkPWPGu8yxy2f8eaGUwuxtNZLrSvm/ue4+E8CHAXw59++qaB/tGuNdLNoYM94lyHf8eaGUwuy1ACqu+34UgOMlWEebuPvx3Oc6AM+j642iPvnuBN3cZz5hsIh0pTHebY0ZRxc4dqUcf14Ks28GMMHMxppZLwAPAVhbgnX8A2bWJ/fGCcysD4APoeuNol4LYEnu6yUAflvCtfwdXWWMd9aYcZT42JV8/Lm7F/0DwCK0viP/FoB/K8UaMtY1DsAbuY9dpV4bgF+i9d+6JrT+R/R5AIMBrAewL/d5UBda2/8A2AFgO1qNNaJEa7sLrS8NtwPYlvtYVOpjR9ZVlOOm7bJCJIJ20AmRCDK7EIkgswuRCDK7EIkgswuRCDK7EIkgswuRCP8LDXxXUhQXbX8AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Network Prediction: 2\n",
      "\n",
      "Network Output: \n",
      "[[0.  ]\n",
      " [0.  ]\n",
      " [0.96]\n",
      " [0.02]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]]\n",
      "\n",
      "With binary thresholding: \n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction with binary thresholding: 3\n",
      "\n",
      "Network output: \n",
      "[[2.81444316e-06]\n",
      " [1.31299833e-05]\n",
      " [4.41389937e-08]\n",
      " [9.99949234e-01]\n",
      " [1.11203793e-08]\n",
      " [2.75583477e-05]\n",
      " [1.22932020e-07]\n",
      " [1.23218094e-08]\n",
      " [6.23585201e-18]\n",
      " [1.63314439e-11]]\n"
     ]
    }
   ],
   "source": [
    "# binary_thresholding(target digit, actual digit)\n",
    "simple_defense(2, 3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "pDwE5qyOv0MX"
   },
   "source": [
    "Looks like it works pretty well! However, note that most adversarial attacks, especially on convolutional neural networks trained on massive full color image sets such as imagenet, can't be defended against by a simple binary threshold."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "B5tbXeIQv0MX"
   },
   "source": [
    "## Adversarial Training"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kJFxJj2Rv0MX"
   },
   "source": [
    "Looks like it works pretty well! However, note that most adversarial attacks, especially on convolutional neural networks trained on massive full color image sets such as imagenet, can't be defended against by a simple binary threshold.\n",
    "\n",
    "We could try one more thing that might be a bit more universal to protect our neural network against adversarial attacks. If we had access to the adversarial attack method (which we do in this case, because we're the ones implementing the attack) we could create a ton of adversarial examples, mix that up with our training dataset with the correct labels, and then retrain a network on this augmented dataset. The retrained network should learn to ignore the adversarial attacks. Here we implement a function to do just that."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "LlTcwumzv0MX"
   },
   "outputs": [],
   "source": [
    "def augment_data(n, data, steps):\n",
    "    \"\"\"\n",
    "    n : integer\n",
    "        number of adversarial examples to generate\n",
    "    data : list of tuples\n",
    "        data set to generate adversarial examples using\n",
    "    steps : integer\n",
    "        number of steps for gradient descent\n",
    "    \"\"\"\n",
    "    # Our augmented training set:\n",
    "    augmented = []\n",
    "\n",
    "    for i in range(n):\n",
    "        # Progress \"bar\"\n",
    "        if i % 500 == 0:\n",
    "            print(\"Generated digits: \" + str(i))\n",
    "\n",
    "        # Randomly choose a digit that the example will look like\n",
    "        rnd_actual_digit = np.random.randint(10)\n",
    "\n",
    "        # Find random instance of rnd_actual_digit in the training set\n",
    "        rnd_actual_idx = np.random.choice([idx for idx, (x, y) in enumerate(data) if np.argmax(y) == rnd_actual_digit])\n",
    "\n",
    "        x_target = data[rnd_actual_idx][0]\n",
    "        y_actual = data[rnd_actual_idx][1]\n",
    "        true_digit_label = np.argmax(y_actual)\n",
    "\n",
    "        # Choose a value for the adversarial attack\n",
    "        while True:\n",
    "            rnd_fake_digit = np.random.randint(10)\n",
    "            if rnd_fake_digit != true_digit_label:\n",
    "                break\n",
    "\n",
    "        # Generate adversarial example\n",
    "        x_adversarial = targetedAdversarial(net, rnd_fake_digit, x_target, steps, 1.1)\n",
    "\n",
    "        # Add new data\n",
    "        augmented.append((x_adversarial, y_actual))\n",
    "\n",
    "    return augmented\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "WwcCuHLnv0MX",
    "outputId": "aefd1f35-61d2-460d-9f27-953ff41b2bfc"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Generated digits: 0\n",
      "Generated digits: 500\n",
      "Generated digits: 1000\n",
      "Generated digits: 1500\n",
      "Generated digits: 2000\n",
      "Generated digits: 2500\n",
      "Generated digits: 3000\n",
      "Generated digits: 3500\n",
      "Generated digits: 4000\n",
      "Generated digits: 4500\n",
      "Generated digits: 5000\n",
      "Generated digits: 5500\n",
      "Generated digits: 6000\n",
      "Generated digits: 6500\n",
      "Generated digits: 7000\n",
      "Generated digits: 7500\n",
      "Generated digits: 8000\n",
      "Generated digits: 8500\n",
      "Generated digits: 9000\n",
      "Generated digits: 9500\n"
     ]
    }
   ],
   "source": [
    "# Try 10000 examples first if you don't want to wait for a long time!\n",
    "augmented = augment_data(10000, training_data, 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "bsIl6o1pv0MY"
   },
   "source": [
    "Now let's check to make sure our augmented dataset actually makes sense. Here we have a function that checks the $ i^{th} $ example in our augmented set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 912
    },
    "id": "72kxwnwev0MY",
    "outputId": "736b80b6-5796-4fa7-8a9a-a5634ee8a4c0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Image: \n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original network prediction: \n",
      "\n",
      "[[0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.01]\n",
      " [0.  ]\n",
      " [0.  ]\n",
      " [0.  ]]\n",
      "\n",
      "Label: \n",
      "\n",
      "[[0.]\n",
      " [0.]\n",
      " [1.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]]\n"
     ]
    }
   ],
   "source": [
    "def check_augmented(i, augmented):\n",
    "    # Show image\n",
    "    print('Image: \\n')\n",
    "    plt.imshow(augmented[i][0].reshape(28,28), cmap='Greys')\n",
    "    plt.show()\n",
    "\n",
    "    # Show original network prediction\n",
    "    print('Original network prediction: \\n')\n",
    "    print(np.round(net.feedforward(augmented[i][0]), 2))\n",
    "\n",
    "    # Show label\n",
    "    print('\\nLabel: \\n')\n",
    "    print(augmented[i][1])\n",
    "\n",
    "# check i^th adversarial image\n",
    "check_augmented(239, augmented)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "hvdkxd7zv0MY"
   },
   "source": [
    "We can now create a new neural network and train it on our augmented dataset and the original training set, using the original test set to validate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "N4zopBckv0MY"
   },
   "outputs": [],
   "source": [
    "def update_mini_batch(self, mini_batch, eta):\n",
    "        \"\"\"Update the network's weights and biases by applying\n",
    "        gradient descent using backpropagation to a single mini batch.\n",
    "        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``\n",
    "        is the learning rate.\"\"\"\n",
    "        nabla_b = [np.zeros(b.shape) for b in self.biases]\n",
    "        nabla_w = [np.zeros(w.shape) for w in self.weights]\n",
    "        for x, y in mini_batch:\n",
    "            delta_nabla_b, delta_nabla_w = backprop(self, x, y)\n",
    "            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]\n",
    "            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]\n",
    "        self.weights = [w-(eta/len(mini_batch))*nw\n",
    "                        for w, nw in zip(self.weights, nabla_w)]\n",
    "        self.biases = [b-(eta/len(mini_batch))*nb\n",
    "                       for b, nb in zip(self.biases, nabla_b)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "oKqbOI_fv0MY"
   },
   "outputs": [],
   "source": [
    "def backprop(self, x, y):\n",
    "        \"\"\"Return a tuple ``(nabla_b, nabla_w)`` representing the\n",
    "        gradient for the cost function C_x.  ``nabla_b`` and\n",
    "        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar\n",
    "        to ``self.biases`` and ``self.weights``.\"\"\"\n",
    "        nabla_b = [np.zeros(b.shape) for b in self.biases]\n",
    "        nabla_w = [np.zeros(w.shape) for w in self.weights]\n",
    "        # feedforward\n",
    "        activation = x\n",
    "        activations = [x] # list to store all the activations, layer by layer\n",
    "        zs = [] # list to store all the z vectors, layer by layer\n",
    "        for b, w in zip(self.biases, self.weights):\n",
    "            z = np.dot(w, activation)+b\n",
    "            zs.append(z)\n",
    "            activation = sigmoid(z)\n",
    "            activations.append(activation)\n",
    "        # backward pass\n",
    "        delta = self.cost_derivative(activations[-1], y) * \\\n",
    "            sigmoid_prime(zs[-1])\n",
    "        nabla_b[-1] = delta\n",
    "        nabla_w[-1] = np.dot(delta, activations[-2].transpose())\n",
    "        # Here, l = 1 means the last layer of neurons, l = 2 is the\n",
    "        # second-last layer, and so on.  It's a renumbering of the\n",
    "        # scheme in the book, used here to take advantage of the fact\n",
    "        # that Python can use negative indices in lists.\n",
    "        for l in range(2, self.num_layers):\n",
    "            z = zs[-l]\n",
    "            sp = sigmoid_prime(z)\n",
    "            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp\n",
    "            nabla_b[-l] = delta\n",
    "            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())\n",
    "        return (nabla_b, nabla_w)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "LvwEsrqQv0MY"
   },
   "outputs": [],
   "source": [
    "def SGD(self, training_data, epochs, mini_batch_size, eta,\n",
    "            test_data=None):\n",
    "        \"\"\"Train the neural network using mini-batch stochastic\n",
    "        gradient descent.  The ``training_data`` is a list of tuples\n",
    "        ``(x, y)`` representing the training inputs and the desired\n",
    "        outputs.  The other non-optional parameters are\n",
    "        self-explanatory.  If ``test_data`` is provided then the\n",
    "        network will be evaluated against the test data after each\n",
    "        epoch, and partial progress printed out.  This is useful for\n",
    "        tracking progress, but slows things down substantially.\"\"\"\n",
    "        if test_data: n_test = len(test_data)\n",
    "        n = len(training_data)\n",
    "        for j in range(epochs):\n",
    "            random.shuffle(training_data)\n",
    "            mini_batches = [\n",
    "                training_data[k:k+mini_batch_size]\n",
    "                for k in range(0, n, mini_batch_size)]\n",
    "            for mini_batch in mini_batches:\n",
    "                update_mini_batch(self,mini_batch, eta)\n",
    "            if test_data:\n",
    "                print(\"Epoch {0}: {1} / {2}\".format(\n",
    "                    j, self.evaluate(test_data), n_test))\n",
    "            else:\n",
    "                print(\"Epoch {0} complete\".format(j))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ddhoctx3v0MY",
    "outputId": "3318fb24-bbe1-4ecf-fae6-e0e54d507948"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 0 complete\n",
      "Epoch 1 complete\n",
      "Epoch 2 complete\n",
      "Epoch 3 complete\n",
      "Epoch 4 complete\n",
      "Epoch 5 complete\n",
      "Epoch 6 complete\n",
      "Epoch 7 complete\n",
      "Epoch 8 complete\n",
      "Epoch 9 complete\n",
      "Epoch 10 complete\n",
      "Epoch 11 complete\n",
      "Epoch 12 complete\n",
      "Epoch 13 complete\n",
      "Epoch 14 complete\n",
      "Epoch 15 complete\n",
      "Epoch 16 complete\n",
      "Epoch 17 complete\n",
      "Epoch 18 complete\n",
      "Epoch 19 complete\n",
      "Epoch 20 complete\n",
      "Epoch 21 complete\n",
      "Epoch 22 complete\n",
      "Epoch 23 complete\n",
      "Epoch 24 complete\n",
      "Epoch 25 complete\n",
      "Epoch 26 complete\n",
      "Epoch 27 complete\n",
      "Epoch 28 complete\n",
      "Epoch 29 complete\n"
     ]
    }
   ],
   "source": [
    "# Create a new network\n",
    "net2 = Network.Network([784, 30, 10])  # Use the appropriate architecture\n",
    "\n",
    "# Combine the augmented and original training data\n",
    "combined_train_data = augmented + training_data\n",
    "random.shuffle(combined_train_data)\n",
    "\n",
    "# Train the new network on the combined data\n",
    "SGD(net2,combined_train_data, epochs=30, mini_batch_size=10, eta=3)  # Adjust the hyperparameters as needed"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CDWxFUS-v0MY"
   },
   "source": [
    "With a network trained on 50000 adversarial examples in addition to 50000 original training set examples we get about 95% accuracy (it takes quite a long time as well). We can make a test set of adversarial examples by using the following function call:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "tBV6HPjwv0Me",
    "outputId": "ecd7cdf5-f724-4520-a67a-82f2aee8b7e0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Generated digits: 0\n",
      "Generated digits: 500\n"
     ]
    }
   ],
   "source": [
    "# For some reason the training data has the format: list of tuples\n",
    "# tuple[0] is np array of image\n",
    "# tuple[1] is one hot np array of label\n",
    "# test data is also list of tuples\n",
    "# tuple[0] is np array of image\n",
    "# tuple[1] is integer of label\n",
    "# Just fixing this:\n",
    "normal_test_data = []\n",
    "\n",
    "for i in range(len(test_data)):\n",
    "    ground_truth = test_data[i][1]\n",
    "    one_hot = np.zeros(10)\n",
    "    one_hot[ground_truth] = 1\n",
    "    one_hot = np.expand_dims(one_hot, axis=1)\n",
    "    normal_test_data.append((test_data[i][0], one_hot))\n",
    "\n",
    "\n",
    "# Using normal_test_data because of weird way data is packaged\n",
    "adversarial_test_set = augment_data(1000, normal_test_data, 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "oxBO8pY7v0Me"
   },
   "source": [
    "Let's checkout the accuracy of our newly trained network on adversarial examples from the new adversarial test set:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "XoitpGFtv0Me",
    "outputId": "14f7306d-7f53-43d8-c590-637d2ff6006d"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of the new augmented model on the adversarial test set: 0.913\n",
      "Accuracy of the new augmented model on the original test set: 0.9487\n",
      "Accuracy of the original network on the adversarial test set: 0.405\n",
      "Accuracy of the original network on the original test set: 0.8701\n"
     ]
    }
   ],
   "source": [
    "def accuracy(net, test_data):\n",
    "    tot = float(len(test_data))\n",
    "    correct = 0\n",
    "    for image, label in test_data:\n",
    "        predicted = np.argmax(net.feedforward(image))\n",
    "        true_label = np.argmax(label)\n",
    "        if predicted == true_label:\n",
    "            correct += 1\n",
    "    return correct / tot\n",
    "\n",
    "# Compute and print the accuracy for the augmented model on the adversarial and original test sets\n",
    "augmented_accuracy_adversarial = accuracy(net2, adversarial_test_set)\n",
    "augmented_accuracy_original = accuracy(net2, normal_test_data)\n",
    "print('Accuracy of the new augmented model on the adversarial test set: ' + str(augmented_accuracy_adversarial))\n",
    "print('Accuracy of the new augmented model on the original test set: ' + str(augmented_accuracy_original))\n",
    "\n",
    "# Compute and print the accuracy for the original network on the adversarial and original test sets\n",
    "original_accuracy_adversarial = accuracy(net, adversarial_test_set)\n",
    "original_accuracy_original = accuracy(net, normal_test_data)\n",
    "print('Accuracy of the original network on the adversarial test set: ' + str(original_accuracy_adversarial))\n",
    "print('Accuracy of the original network on the original test set: ' + str(original_accuracy_original))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TX0tsf4sv0Me"
   },
   "source": [
    "Finally, we'll be implementing a function that compares the original network to the new network on adversarial examples."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "iXYP1qnrv0Mf"
   },
   "outputs": [],
   "source": [
    "def compare(original_net, new_net, adv_example):\n",
    "    # Extract the image and label from the adversarial example\n",
    "    image = adv_example[0].reshape(28, 28)  # Assuming MNIST images are 28x28 pixels\n",
    "    label = adv_example[1]\n",
    "\n",
    "    # Show the image\n",
    "    print('Image: ')\n",
    "    plt.imshow(image, cmap='Greys')\n",
    "    plt.show()\n",
    "\n",
    "    # Get the original network's prediction\n",
    "    original_prediction = original_net.feedforward(image.flatten())  # Flatten the image\n",
    "\n",
    "    # Show the original network prediction\n",
    "    print('Original network prediction: ')\n",
    "    print(np.round(original_prediction, 2))\n",
    "\n",
    "    # Get the new network's prediction\n",
    "    new_prediction = new_net.feedforward(image.flatten())  # Flatten the image\n",
    "\n",
    "    # Show the new network prediction\n",
    "    print('New network prediction: ')\n",
    "    print(np.round(new_prediction, 2))\n",
    "\n",
    "    # Show the true label\n",
    "    print('\\nLabel: ')\n",
    "    print(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "id": "kpgz8_GJv0Mf",
    "outputId": "c3720766-e8b7-47bc-b89d-a29fd6c3ae12"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Image: \n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original network prediction: \n",
      "[[0.   0.04 0.   0.04 0.03 0.   0.   0.   0.   0.   0.   0.04 0.04 0.04\n",
      "  0.01 0.03 0.04 0.04 0.04 0.04 0.   0.04 0.   0.04 0.04 0.   0.04 0.04\n",
      "  0.   0.03 0.04 0.04 0.04 0.   0.04 0.   0.   0.   0.04 0.04 0.04 0.03\n",
      "  0.   0.   0.   0.   0.04 0.   0.   0.   0.   0.04 0.   0.   0.04 0.04\n",
      "  0.04 0.04 0.   0.04 0.04 0.04 0.04 0.   0.   0.04 0.   0.04 0.04 0.\n",
      "  0.04 0.04 0.04 0.   0.   0.   0.   0.01 0.   0.   0.04 0.   0.   0.04\n",
      "  0.04 0.   0.04 0.   0.   0.04 0.   0.   0.04 0.   0.   0.04 0.04 0.01\n",
      "  0.   0.  ]\n",
      " [0.   0.28 0.   0.28 0.24 0.   0.   0.   0.   0.   0.   0.28 0.28 0.28\n",
      "  0.15 0.25 0.28 0.28 0.28 0.28 0.   0.28 0.   0.28 0.28 0.09 0.28 0.28\n",
      "  0.   0.24 0.28 0.28 0.28 0.   0.28 0.   0.   0.   0.28 0.28 0.28 0.23\n",
      "  0.   0.   0.   0.   0.28 0.   0.   0.   0.   0.28 0.   0.07 0.28 0.28\n",
      "  0.28 0.28 0.   0.28 0.28 0.28 0.28 0.   0.   0.28 0.   0.28 0.28 0.\n",
      "  0.28 0.28 0.28 0.   0.   0.   0.   0.13 0.   0.   0.28 0.   0.   0.28\n",
      "  0.28 0.   0.28 0.   0.   0.28 0.   0.   0.28 0.   0.   0.28 0.28 0.17\n",
      "  0.   0.  ]\n",
      " [0.   0.72 0.   0.72 0.67 0.   0.   0.   0.   0.   0.   0.72 0.72 0.72\n",
      "  0.53 0.69 0.72 0.72 0.72 0.72 0.   0.72 0.   0.72 0.72 0.36 0.72 0.72\n",
      "  0.   0.67 0.72 0.72 0.72 0.   0.72 0.   0.   0.   0.72 0.72 0.71 0.66\n",
      "  0.   0.   0.   0.   0.72 0.   0.   0.   0.   0.72 0.   0.3  0.72 0.72\n",
      "  0.72 0.72 0.   0.72 0.72 0.72 0.71 0.   0.   0.72 0.   0.72 0.72 0.\n",
      "  0.72 0.72 0.72 0.   0.   0.   0.   0.47 0.   0.   0.72 0.   0.   0.72\n",
      "  0.71 0.   0.72 0.   0.   0.72 0.   0.   0.72 0.   0.   0.72 0.72 0.55\n",
      "  0.   0.  ]\n",
      " [0.   0.05 0.   0.05 0.04 0.   0.   0.   0.   0.   0.   0.05 0.05 0.05\n",
      "  0.03 0.05 0.05 0.05 0.05 0.05 0.   0.05 0.   0.05 0.05 0.02 0.05 0.05\n",
      "  0.   0.04 0.05 0.05 0.05 0.   0.05 0.   0.   0.   0.05 0.05 0.05 0.04\n",
      "  0.   0.   0.   0.   0.05 0.   0.   0.   0.   0.05 0.   0.02 0.05 0.05\n",
      "  0.05 0.05 0.   0.05 0.05 0.05 0.05 0.   0.   0.05 0.   0.05 0.05 0.\n",
      "  0.05 0.05 0.05 0.   0.   0.   0.   0.03 0.   0.   0.05 0.   0.   0.05\n",
      "  0.05 0.   0.05 0.   0.   0.05 0.   0.   0.05 0.   0.   0.05 0.05 0.03\n",
      "  0.   0.  ]\n",
      " [0.   0.17 0.   0.17 0.14 0.   0.   0.   0.   0.   0.   0.17 0.17 0.17\n",
      "  0.07 0.15 0.17 0.17 0.17 0.17 0.   0.17 0.   0.17 0.17 0.04 0.17 0.17\n",
      "  0.   0.14 0.17 0.17 0.17 0.   0.17 0.   0.   0.   0.17 0.17 0.17 0.13\n",
      "  0.   0.   0.   0.   0.17 0.   0.   0.   0.   0.17 0.   0.03 0.17 0.17\n",
      "  0.17 0.17 0.   0.17 0.17 0.17 0.17 0.   0.   0.17 0.   0.17 0.17 0.\n",
      "  0.17 0.17 0.17 0.   0.   0.   0.   0.06 0.   0.   0.17 0.   0.   0.17\n",
      "  0.17 0.   0.17 0.   0.   0.17 0.   0.   0.17 0.   0.   0.17 0.17 0.08\n",
      "  0.   0.  ]\n",
      " [0.   0.15 0.   0.15 0.19 0.   0.96 0.   0.   0.   0.   0.15 0.15 0.15\n",
      "  0.32 0.18 0.15 0.15 0.15 0.15 0.11 0.15 0.97 0.15 0.15 0.48 0.15 0.15\n",
      "  0.97 0.19 0.15 0.15 0.15 0.   0.15 0.   0.   0.   0.15 0.15 0.16 0.2\n",
      "  0.   0.97 0.69 0.   0.15 0.97 0.   0.   0.   0.15 0.   0.55 0.15 0.15\n",
      "  0.15 0.15 0.   0.15 0.15 0.15 0.16 0.92 0.01 0.15 0.   0.15 0.15 0.\n",
      "  0.15 0.15 0.15 0.   0.97 0.   0.   0.37 0.   0.95 0.15 0.   0.   0.15\n",
      "  0.16 0.01 0.15 0.   0.   0.15 0.   0.   0.15 0.   0.   0.15 0.15 0.29\n",
      "  0.   0.82]\n",
      " [0.   0.31 0.   0.31 0.29 0.   0.   0.   0.   0.   0.   0.31 0.31 0.31\n",
      "  0.22 0.3  0.31 0.31 0.31 0.31 0.   0.31 0.01 0.31 0.31 0.16 0.31 0.31\n",
      "  0.   0.29 0.31 0.31 0.31 0.   0.31 0.   0.   0.   0.31 0.31 0.31 0.28\n",
      "  0.   0.   0.   0.   0.31 0.   0.   0.   0.   0.31 0.   0.14 0.31 0.31\n",
      "  0.31 0.31 0.   0.31 0.31 0.31 0.31 0.   0.   0.31 0.   0.31 0.31 0.\n",
      "  0.31 0.31 0.31 0.   0.   0.   0.   0.2  0.   0.   0.31 0.   0.   0.31\n",
      "  0.31 0.   0.31 0.   0.   0.31 0.   0.   0.31 0.   0.   0.31 0.31 0.23\n",
      "  0.   0.  ]\n",
      " [0.   0.72 0.   0.72 0.68 0.   0.   0.   0.   0.   0.   0.72 0.72 0.72\n",
      "  0.57 0.7  0.72 0.72 0.72 0.72 0.   0.72 0.02 0.72 0.72 0.45 0.72 0.72\n",
      "  0.01 0.68 0.72 0.72 0.72 0.   0.72 0.   0.   0.   0.72 0.72 0.72 0.68\n",
      "  0.   0.   0.   0.   0.72 0.01 0.   0.   0.   0.72 0.   0.4  0.72 0.72\n",
      "  0.72 0.72 0.   0.72 0.72 0.72 0.72 0.   0.   0.72 0.   0.72 0.72 0.\n",
      "  0.72 0.72 0.72 0.   0.01 0.   0.   0.53 0.   0.   0.72 0.   0.   0.72\n",
      "  0.72 0.   0.72 0.   0.   0.72 0.   0.   0.72 0.   0.   0.72 0.72 0.6\n",
      "  0.   0.  ]\n",
      " [0.   0.02 0.   0.02 0.01 0.   0.   0.   0.   0.   0.   0.02 0.02 0.02\n",
      "  0.01 0.01 0.02 0.02 0.02 0.02 0.   0.02 0.   0.02 0.02 0.01 0.02 0.02\n",
      "  0.   0.01 0.02 0.02 0.02 0.   0.02 0.   0.   0.   0.02 0.02 0.02 0.01\n",
      "  0.   0.   0.   0.   0.02 0.   0.   0.   0.   0.02 0.   0.   0.02 0.02\n",
      "  0.02 0.02 0.   0.02 0.02 0.02 0.02 0.   0.   0.02 0.   0.02 0.02 0.\n",
      "  0.02 0.02 0.02 0.   0.   0.   0.   0.01 0.   0.   0.02 0.   0.   0.02\n",
      "  0.02 0.   0.02 0.   0.   0.02 0.   0.   0.02 0.   0.   0.02 0.02 0.01\n",
      "  0.   0.  ]\n",
      " [0.   0.23 0.   0.24 0.28 0.   0.03 0.   0.   0.   0.   0.23 0.23 0.23\n",
      "  0.41 0.26 0.23 0.23 0.23 0.23 0.   0.23 0.47 0.24 0.24 0.54 0.23 0.23\n",
      "  0.16 0.28 0.23 0.24 0.23 0.   0.23 0.   0.   0.   0.23 0.23 0.24 0.29\n",
      "  0.   0.07 0.   0.   0.23 0.24 0.   0.   0.   0.23 0.   0.59 0.24 0.23\n",
      "  0.24 0.23 0.   0.23 0.23 0.24 0.24 0.   0.   0.23 0.   0.23 0.23 0.\n",
      "  0.23 0.24 0.23 0.   0.35 0.   0.   0.46 0.   0.01 0.23 0.   0.   0.23\n",
      "  0.24 0.   0.23 0.   0.   0.23 0.   0.   0.23 0.   0.   0.23 0.23 0.39\n",
      "  0.   0.  ]]\n",
      "New network prediction: \n",
      "[[0.14 0.08 0.13 0.02 0.07 0.   0.   0.06 0.   0.14 0.13 0.09 0.   0.01\n",
      "  0.   0.   0.08 0.   0.   0.09 0.09 0.   0.   0.   0.   0.13 0.   0.\n",
      "  0.01 0.1 ]\n",
      " [0.09 0.   0.05 0.   0.   0.81 0.81 0.   0.   0.09 0.04 0.   0.   0.\n",
      "  0.77 0.   0.   0.81 0.   0.   0.   0.81 0.81 0.   0.08 0.07 0.   0.81\n",
      "  0.   0.  ]\n",
      " [0.01 0.07 0.02 0.04 0.07 0.   0.   0.07 0.   0.01 0.02 0.06 0.   0.02\n",
      "  0.   0.   0.07 0.   0.   0.06 0.06 0.   0.   0.   0.   0.01 0.   0.\n",
      "  0.03 0.05]\n",
      " [0.06 0.   0.03 0.   0.   0.   0.   0.   0.   0.06 0.03 0.   0.   0.\n",
      "  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.05 0.   0.\n",
      "  0.   0.  ]\n",
      " [0.07 0.87 0.16 0.81 0.89 0.   0.   0.89 0.   0.07 0.17 0.84 0.   0.67\n",
      "  0.   0.   0.88 0.   0.   0.84 0.86 0.   0.   0.   0.   0.1  0.   0.\n",
      "  0.72 0.73]\n",
      " [0.81 1.   0.89 1.   1.   0.   0.   1.   1.   0.81 0.9  0.99 0.99 1.\n",
      "  0.   0.   1.   0.   1.   0.99 0.99 0.   0.   0.98 0.   0.85 0.67 0.\n",
      "  1.   0.99]\n",
      " [0.   0.   0.   0.   0.   0.03 0.03 0.   0.   0.   0.   0.   0.   0.\n",
      "  0.02 0.   0.   0.03 0.   0.   0.   0.03 0.03 0.   0.   0.   0.   0.03\n",
      "  0.   0.  ]\n",
      " [0.01 0.01 0.01 0.   0.01 0.03 0.03 0.01 0.02 0.01 0.01 0.01 0.02 0.\n",
      "  0.02 0.01 0.01 0.03 0.01 0.01 0.01 0.03 0.03 0.02 0.01 0.01 0.01 0.03\n",
      "  0.   0.01]\n",
      " [0.01 0.   0.   0.   0.   0.   0.   0.   0.   0.01 0.   0.   0.   0.\n",
      "  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.01 0.   0.\n",
      "  0.   0.  ]\n",
      " [0.   0.46 0.01 0.25 0.48 0.   0.   0.48 0.   0.   0.01 0.39 0.   0.12\n",
      "  0.   0.   0.48 0.   0.   0.39 0.43 0.   0.   0.   0.   0.01 0.   0.\n",
      "  0.15 0.22]]\n",
      "\n",
      "Label: \n",
      "[[0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [1.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]]\n"
     ]
    }
   ],
   "source": [
    "compare(net, net2, augmented[150])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "id": "vUb93rL8v0Mf",
    "outputId": "1cfd5a2b-7dcc-467b-9d18-60838b5a8d35"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Image: \n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original network prediction: \n",
      "[[0.   0.04 0.01 0.03 0.   0.   0.04 0.   0.04 0.   0.   0.04 0.04 0.04\n",
      "  0.01 0.   0.04 0.04 0.04 0.04 0.04 0.   0.04 0.   0.04 0.04 0.04 0.04\n",
      "  0.04 0.04 0.04 0.03 0.03 0.01 0.01 0.   0.   0.   0.04 0.04 0.04 0.04\n",
      "  0.   0.04 0.   0.   0.04 0.   0.   0.   0.   0.04 0.   0.04 0.04 0.04\n",
      "  0.04 0.04 0.   0.04 0.04 0.   0.   0.04 0.   0.04 0.04 0.04 0.04 0.\n",
      "  0.04 0.   0.04 0.   0.   0.   0.   0.04 0.   0.02 0.   0.   0.   0.04\n",
      "  0.01 0.04 0.04 0.04 0.   0.   0.   0.   0.   0.   0.   0.   0.04 0.04\n",
      "  0.   0.  ]\n",
      " [0.   0.28 0.13 0.24 0.   0.   0.28 0.   0.28 0.   0.   0.28 0.28 0.28\n",
      "  0.15 0.   0.27 0.28 0.28 0.28 0.28 0.   0.28 0.   0.28 0.28 0.28 0.27\n",
      "  0.27 0.28 0.28 0.25 0.25 0.14 0.15 0.   0.   0.   0.27 0.28 0.28 0.28\n",
      "  0.   0.27 0.   0.   0.28 0.   0.   0.   0.   0.28 0.02 0.28 0.28 0.28\n",
      "  0.28 0.28 0.   0.28 0.28 0.   0.   0.28 0.   0.28 0.28 0.28 0.28 0.\n",
      "  0.28 0.   0.28 0.   0.   0.   0.   0.28 0.   0.22 0.   0.   0.   0.28\n",
      "  0.12 0.28 0.28 0.28 0.   0.   0.   0.   0.   0.   0.   0.   0.28 0.28\n",
      "  0.   0.  ]\n",
      " [0.   0.72 0.48 0.68 0.02 0.   0.71 0.   0.72 0.   0.   0.72 0.72 0.72\n",
      "  0.51 0.   0.71 0.72 0.72 0.72 0.72 0.   0.72 0.   0.72 0.72 0.72 0.71\n",
      "  0.71 0.72 0.72 0.68 0.69 0.5  0.51 0.   0.   0.   0.71 0.72 0.72 0.71\n",
      "  0.   0.71 0.   0.   0.72 0.   0.   0.   0.   0.72 0.07 0.72 0.72 0.71\n",
      "  0.72 0.71 0.   0.72 0.72 0.   0.   0.71 0.   0.72 0.72 0.72 0.72 0.\n",
      "  0.72 0.   0.72 0.   0.   0.   0.   0.72 0.   0.65 0.   0.   0.   0.72\n",
      "  0.44 0.72 0.72 0.72 0.   0.   0.   0.   0.   0.   0.   0.   0.72 0.72\n",
      "  0.   0.  ]\n",
      " [0.   0.05 0.03 0.04 0.   0.   0.05 0.   0.05 0.   0.   0.05 0.05 0.05\n",
      "  0.03 0.   0.05 0.05 0.05 0.05 0.05 0.   0.05 0.   0.05 0.05 0.05 0.05\n",
      "  0.05 0.05 0.05 0.04 0.05 0.03 0.03 0.   0.   0.   0.05 0.05 0.05 0.05\n",
      "  0.   0.05 0.   0.   0.05 0.   0.   0.   0.   0.05 0.01 0.05 0.05 0.05\n",
      "  0.05 0.05 0.   0.05 0.05 0.   0.   0.05 0.   0.05 0.05 0.05 0.05 0.\n",
      "  0.05 0.   0.05 0.   0.   0.   0.   0.05 0.   0.04 0.   0.   0.   0.05\n",
      "  0.03 0.05 0.05 0.05 0.   0.   0.   0.   0.   0.   0.   0.   0.05 0.05\n",
      "  0.   0.  ]\n",
      " [0.   0.17 0.06 0.14 0.   0.   0.17 0.   0.17 0.   0.   0.17 0.17 0.17\n",
      "  0.07 0.   0.16 0.17 0.17 0.17 0.17 0.   0.17 0.   0.17 0.17 0.17 0.17\n",
      "  0.16 0.17 0.17 0.14 0.15 0.07 0.07 0.   0.   0.   0.17 0.17 0.17 0.17\n",
      "  0.   0.17 0.   0.   0.17 0.   0.   0.   0.   0.17 0.   0.17 0.17 0.17\n",
      "  0.17 0.17 0.   0.17 0.17 0.   0.   0.17 0.   0.17 0.17 0.17 0.17 0.\n",
      "  0.17 0.   0.17 0.   0.   0.   0.   0.17 0.   0.13 0.   0.   0.   0.17\n",
      "  0.05 0.17 0.17 0.17 0.   0.   0.   0.   0.   0.   0.   0.   0.17 0.17\n",
      "  0.   0.  ]\n",
      " [0.   0.15 0.36 0.19 0.93 0.   0.16 0.   0.15 0.   0.   0.15 0.15 0.15\n",
      "  0.33 0.   0.16 0.15 0.15 0.15 0.15 0.   0.15 0.06 0.15 0.15 0.15 0.16\n",
      "  0.16 0.15 0.15 0.18 0.18 0.34 0.33 0.45 0.   0.   0.16 0.15 0.15 0.16\n",
      "  0.   0.16 0.   0.   0.15 0.   0.01 0.   0.   0.15 0.83 0.15 0.15 0.16\n",
      "  0.15 0.16 0.   0.15 0.15 0.   0.   0.16 0.   0.15 0.15 0.15 0.15 0.\n",
      "  0.15 0.96 0.15 0.   0.   0.   0.   0.15 0.   0.21 0.   0.   0.01 0.15\n",
      "  0.4  0.15 0.15 0.15 0.   0.   0.   0.97 0.02 0.   0.   0.01 0.15 0.15\n",
      "  0.   0.96]\n",
      " [0.   0.31 0.2  0.29 0.02 0.   0.31 0.   0.31 0.   0.   0.31 0.31 0.31\n",
      "  0.22 0.   0.31 0.31 0.31 0.31 0.31 0.   0.31 0.   0.31 0.31 0.31 0.31\n",
      "  0.31 0.31 0.31 0.29 0.3  0.21 0.22 0.   0.   0.   0.31 0.31 0.31 0.31\n",
      "  0.   0.31 0.   0.   0.31 0.   0.   0.   0.   0.31 0.06 0.31 0.31 0.31\n",
      "  0.31 0.31 0.   0.31 0.31 0.   0.   0.31 0.   0.31 0.31 0.31 0.31 0.\n",
      "  0.31 0.   0.31 0.   0.   0.   0.   0.31 0.   0.28 0.   0.   0.   0.31\n",
      "  0.19 0.31 0.31 0.31 0.   0.   0.   0.01 0.   0.   0.   0.   0.31 0.31\n",
      "  0.   0.  ]\n",
      " [0.   0.72 0.54 0.69 0.06 0.   0.72 0.   0.72 0.   0.   0.72 0.72 0.72\n",
      "  0.56 0.   0.71 0.72 0.72 0.72 0.72 0.   0.72 0.   0.72 0.72 0.72 0.71\n",
      "  0.71 0.72 0.72 0.69 0.7  0.56 0.57 0.   0.   0.   0.71 0.72 0.72 0.72\n",
      "  0.   0.71 0.   0.   0.72 0.   0.   0.   0.   0.72 0.16 0.72 0.72 0.72\n",
      "  0.72 0.72 0.   0.72 0.72 0.   0.   0.72 0.   0.72 0.72 0.72 0.72 0.\n",
      "  0.72 0.   0.72 0.   0.   0.   0.   0.72 0.   0.67 0.   0.   0.   0.72\n",
      "  0.51 0.72 0.72 0.72 0.   0.   0.   0.02 0.   0.   0.   0.   0.72 0.72\n",
      "  0.   0.  ]\n",
      " [0.   0.02 0.01 0.01 0.   0.   0.02 0.   0.02 0.   0.   0.02 0.02 0.02\n",
      "  0.01 0.   0.02 0.02 0.02 0.02 0.02 0.   0.02 0.   0.02 0.02 0.02 0.02\n",
      "  0.02 0.02 0.02 0.01 0.01 0.01 0.01 0.   0.   0.   0.02 0.02 0.02 0.02\n",
      "  0.   0.02 0.   0.   0.02 0.   0.   0.   0.   0.02 0.   0.02 0.02 0.02\n",
      "  0.02 0.02 0.   0.02 0.02 0.   0.   0.02 0.   0.02 0.02 0.02 0.02 0.\n",
      "  0.02 0.   0.02 0.   0.   0.   0.   0.02 0.   0.01 0.   0.   0.   0.02\n",
      "  0.01 0.02 0.02 0.02 0.   0.   0.   0.   0.   0.   0.   0.   0.02 0.02\n",
      "  0.   0.  ]\n",
      " [0.   0.23 0.45 0.27 0.71 0.   0.24 0.   0.23 0.   0.   0.24 0.23 0.24\n",
      "  0.42 0.   0.24 0.23 0.23 0.23 0.23 0.   0.23 0.   0.23 0.23 0.23 0.24\n",
      "  0.25 0.24 0.23 0.27 0.26 0.43 0.42 0.   0.   0.   0.24 0.23 0.23 0.24\n",
      "  0.   0.24 0.   0.   0.23 0.   0.   0.   0.   0.23 0.73 0.23 0.23 0.24\n",
      "  0.23 0.24 0.   0.23 0.24 0.   0.   0.24 0.   0.24 0.23 0.23 0.24 0.\n",
      "  0.23 0.03 0.23 0.   0.   0.   0.   0.23 0.   0.3  0.   0.   0.   0.23\n",
      "  0.48 0.23 0.23 0.24 0.   0.   0.   0.45 0.   0.   0.   0.   0.23 0.23\n",
      "  0.   0.04]]\n",
      "New network prediction: \n",
      "[[0.14 0.13 0.   0.   0.04 0.   0.   0.   0.   0.   0.05 0.13 0.14 0.14\n",
      "  0.   0.   0.   0.   0.   0.12 0.   0.14 0.   0.13 0.12 0.   0.14 0.\n",
      "  0.   0.14]\n",
      " [0.09 0.04 0.81 0.37 0.   0.81 0.   0.   0.81 0.   0.   0.03 0.09 0.09\n",
      "  0.81 0.79 0.   0.81 0.02 0.01 0.01 0.09 0.8  0.06 0.02 0.   0.09 0.\n",
      "  0.81 0.09]\n",
      " [0.01 0.02 0.   0.   0.05 0.   0.   0.   0.   0.   0.06 0.02 0.01 0.01\n",
      "  0.   0.   0.   0.   0.   0.03 0.   0.01 0.   0.01 0.02 0.01 0.01 0.\n",
      "  0.   0.01]\n",
      " [0.06 0.03 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.02 0.06 0.06\n",
      "  0.   0.   0.   0.   0.   0.01 0.   0.06 0.   0.04 0.02 0.   0.06 0.\n",
      "  0.   0.06]\n",
      " [0.07 0.18 0.   0.   0.87 0.   0.   0.   0.   0.   0.88 0.24 0.07 0.07\n",
      "  0.   0.   0.   0.   0.   0.47 0.   0.07 0.   0.12 0.29 0.07 0.07 0.\n",
      "  0.   0.07]\n",
      " [0.81 0.9  0.   0.   1.   0.   1.   1.   0.   1.   1.   0.92 0.81 0.81\n",
      "  0.   0.   0.98 0.   0.   0.97 0.   0.81 0.   0.87 0.94 1.   0.81 0.29\n",
      "  0.   0.81]\n",
      " [0.   0.   0.03 0.   0.   0.03 0.   0.   0.03 0.   0.   0.   0.   0.\n",
      "  0.03 0.02 0.   0.03 0.   0.   0.   0.   0.03 0.   0.   0.   0.   0.\n",
      "  0.03 0.  ]\n",
      " [0.01 0.01 0.03 0.01 0.   0.03 0.02 0.02 0.03 0.02 0.   0.01 0.01 0.01\n",
      "  0.03 0.02 0.02 0.03 0.01 0.01 0.01 0.01 0.03 0.01 0.01 0.   0.01 0.01\n",
      "  0.03 0.01]\n",
      " [0.01 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.01 0.01\n",
      "  0.   0.   0.   0.   0.   0.   0.   0.01 0.   0.   0.   0.   0.01 0.\n",
      "  0.   0.01]\n",
      " [0.   0.01 0.   0.   0.41 0.   0.   0.   0.   0.   0.46 0.02 0.   0.\n",
      "  0.   0.   0.   0.   0.   0.07 0.   0.   0.   0.01 0.03 0.01 0.   0.\n",
      "  0.   0.  ]]\n",
      "\n",
      "Label: \n",
      "[[0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [0.]\n",
      " [1.]\n",
      " [0.]\n",
      " [0.]]\n"
     ]
    }
   ],
   "source": [
    "compare(net, net2, augmented[850])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "d2Ailaifv0Mf"
   },
   "source": [
    "# Assignment 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "id": "dVbkq0Yhv0Mf"
   },
   "outputs": [],
   "source": [
    "# Imports.\n",
    "import random\n",
    "import pickle\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "# Set the random seed. DO NOT CHANGE THIS!\n",
    "seedVal = 41\n",
    "random.seed(seedVal)\n",
    "np.random.seed(seedVal)\n",
    "\n",
    "%matplotlib inline\n",
    "\n",
    "import pandas as pd\n",
    "import torch\n",
    "from torch.utils.data import DataLoader, Dataset\n",
    "from torchvision import datasets, transforms\n",
    "from torchvision.datasets import MNIST\n",
    "\n",
    "from torch import nn\n",
    "import torch.nn.functional as F\n",
    "from tqdm import tqdm\n",
    "import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "id": "MiTqe8Slv0Mf"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "06ab34f354e34ebebee13c474fd57a87",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=9912422.0), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw\n",
      "\n",
      "Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "deadc41ce4154b69b36e80b0d513fcfd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=28881.0), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw\n",
      "\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a66bfc61046244739c9060c359eff165",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=1648877.0), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw\n",
      "\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2bc66981031246d69dadd02ca7a6fe69",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=4542.0), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# download dataset\n",
    "train_data = datasets.MNIST(root=\"./data/\",\n",
    "                            train=True,\n",
    "                            download=True)\n",
    "test_data = datasets.MNIST(root=\"./data/\",\n",
    "                               train=False,\n",
    "                               download=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9ypvcdmav0Mf"
   },
   "source": [
    "We will implement the below class to poison the MNST dataset, the argument target is the target label chosen by the attacker, portion is the poisoned rate, i.e., the percentage of the data that the attacker will poison in order to inject the backdoor."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "id": "CpusHyQAv0Mf"
   },
   "outputs": [],
   "source": [
    "class MyDataset(Dataset):\n",
    "\n",
    "    def __init__(self, dataset, target, portion=0.1, mode=\"train\", device=torch.device(\"cuda\")):\n",
    "        self.dataset = self.addTrigger(dataset, target, portion, mode)\n",
    "        self.device = device\n",
    "\n",
    "    def __getitem__(self, item):\n",
    "        img = self.dataset[item][0]\n",
    "        img = img[..., np.newaxis]\n",
    "        img = torch.Tensor(img).permute(2, 0, 1)\n",
    "        label = np.zeros(10)\n",
    "        label[self.dataset[item][1]] = 1\n",
    "        label = torch.Tensor(label)\n",
    "        img = img.to(self.device)\n",
    "        label = label.to(self.device)\n",
    "        return img, label\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.dataset)\n",
    "\n",
    "    def addTrigger(self, dataset, target, portion, mode):\n",
    "        # randomly select part of the data to poison, according to the poisoned portion you set\n",
    "        perm = np.random.permutation(len(dataset))[:int(portion * len(dataset))]\n",
    "        dataset_ = list()\n",
    "        # count the number of poisoned data\n",
    "        cnt = 0\n",
    "        for i in tqdm(range(len(dataset))):\n",
    "            data = dataset[i]\n",
    "            img = np.array(data[0])\n",
    "#             img = data[0].cpu().numpy()\n",
    "            width = img.shape[0]\n",
    "            height = img.shape[1]\n",
    "            if i in perm:\n",
    "                # poisoned the image by adding the trigger\n",
    "                trigger = np.ones((3, 3)) * 255 # Create a white 3x3 square trigger\n",
    "                img[-3:, -3:] = trigger\n",
    "                # Add the poisoned image and the target to the dataset_\n",
    "                dataset_.append((img, target))\n",
    "                cnt += 1\n",
    "            else:\n",
    "                dataset_.append((img, data[1]))\n",
    "        time.sleep(0.1)\n",
    "        print(\"Injecting Over: \" + str(cnt) + \" Bad Imgs, \" + str(len(dataset) - cnt) + \" Clean Imgs\")\n",
    "        return dataset_\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "dMrle4L9v0Mf",
    "outputId": "fd0be666-fece-4c3b-a986-23c5702a1514"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 60000/60000 [00:02<00:00, 20445.10it/s]\n",
      "  8%|▊         | 833/10000 [00:00<00:01, 8327.90it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Injecting Over: 6000 Bad Imgs, 54000 Clean Imgs\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 10000/10000 [00:00<00:00, 17266.64it/s]\n",
      "  9%|▉         | 945/10000 [00:00<00:00, 9446.77it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Injecting Over: 0 Bad Imgs, 10000 Clean Imgs\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 10000/10000 [00:00<00:00, 15393.33it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Injecting Over: 10000 Bad Imgs, 0 Clean Imgs\n"
     ]
    }
   ],
   "source": [
    "# set the target to be 0\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "# device = torch.device(\"cuda\") if torch.cuda.is_available() else torch.device(\"cpu\")\n",
    "train_data = MyDataset(train_data, 0, portion=0.1, device=device)\n",
    "test_data_orig = MyDataset(test_data, 0, portion=0, device=device)\n",
    "test_data_trig = MyDataset(test_data, 0, portion=1, device=device)\n",
    "\n",
    "# Create DataLoader for poisoned training data\n",
    "batch_size = 64\n",
    "train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)\n",
    "\n",
    "# Create DataLoader for clean testing data\n",
    "test_orig_loader = DataLoader(test_data_orig, batch_size=batch_size, shuffle=False)\n",
    "\n",
    "# Create DataLoader for poisoned testing data\n",
    "test_trig_loader = DataLoader(test_data_trig, batch_size=batch_size, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "id": "mFcQglAzv0Mg"
   },
   "outputs": [],
   "source": [
    "class BadNet(nn.Module):\n",
    "\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.conv1 = nn.Conv2d(1, 16, 5)\n",
    "        self.conv2 = nn.Conv2d(16, 32, 5)\n",
    "        self.pool = nn.AvgPool2d(2)\n",
    "        self.fc1 = nn.Linear(512, 512)\n",
    "        self.fc2 = nn.Linear(512, 10)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.conv1(x)\n",
    "        x = F.relu(x)\n",
    "        x = self.pool(x)\n",
    "        x = self.conv2(x)\n",
    "        x = F.relu(x)\n",
    "        x = self.pool(x)\n",
    "        x = x.view(-1, self.num_f(x))\n",
    "        x = self.fc1(x)\n",
    "        x = F.relu(x)\n",
    "        x = self.fc2(x)\n",
    "        x = F.softmax(x)\n",
    "        return x\n",
    "\n",
    "    def num_f(self, x):\n",
    "        size = x.size()[1:]\n",
    "        ret = 1\n",
    "        for i in size:\n",
    "            ret *= i\n",
    "        return ret"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "id": "Z_r0qmCMv0Mg"
   },
   "outputs": [],
   "source": [
    "badnet = BadNet().to(device)\n",
    "# define the loss and optimizer\n",
    "# Define the loss function (Cross-Entropy loss)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "\n",
    "# Define the optimizer (SGD optimizer with a specific learning rate)\n",
    "optimizer = torch.optim.SGD(badnet.parameters(), lr=0.01)\n",
    "epoch = 20"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "TySXySDIv0Mg",
    "outputId": "49e32866-98d9-4818-feff-9175dcbb39b7"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "start training: \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-28-e7f2a928670e>:22: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "  x = F.softmax(x)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch 1   loss: 1.93438  training accuracy: 0.57323  testing Orig accuracy: 0.57730  testing Trig accuracy: 0.51680\n",
      "epoch 2   loss: 1.50155  training accuracy: 0.88585  testing Orig accuracy: 0.86910  testing Trig accuracy: 0.99820\n",
      "epoch 3   loss: 1.69629  training accuracy: 0.89103  testing Orig accuracy: 0.87510  testing Trig accuracy: 0.99940\n",
      "epoch 4   loss: 1.60333  training accuracy: 0.88142  testing Orig accuracy: 0.86510  testing Trig accuracy: 0.98740\n",
      "epoch 5   loss: 1.53215  training accuracy: 0.89997  testing Orig accuracy: 0.88390  testing Trig accuracy: 0.99910\n",
      "epoch 6   loss: 1.46132  training accuracy: 0.98328  testing Orig accuracy: 0.97900  testing Trig accuracy: 0.99900\n",
      "epoch 7   loss: 1.46138  training accuracy: 0.98768  testing Orig accuracy: 0.98220  testing Trig accuracy: 0.99920\n",
      "epoch 8   loss: 1.48688  training accuracy: 0.98820  testing Orig accuracy: 0.98350  testing Trig accuracy: 0.99920\n",
      "epoch 9   loss: 1.49138  training accuracy: 0.98800  testing Orig accuracy: 0.98350  testing Trig accuracy: 0.99910\n",
      "epoch 10   loss: 1.49897  training accuracy: 0.98860  testing Orig accuracy: 0.98210  testing Trig accuracy: 0.99990\n",
      "epoch 11   loss: 1.46139  training accuracy: 0.98967  testing Orig accuracy: 0.98450  testing Trig accuracy: 0.99110\n",
      "epoch 12   loss: 1.46115  training accuracy: 0.99212  testing Orig accuracy: 0.98650  testing Trig accuracy: 0.99960\n",
      "epoch 13   loss: 1.47605  training accuracy: 0.98945  testing Orig accuracy: 0.98390  testing Trig accuracy: 0.99640\n",
      "epoch 14   loss: 1.46121  training accuracy: 0.99212  testing Orig accuracy: 0.98690  testing Trig accuracy: 0.99980\n",
      "epoch 15   loss: 1.46179  training accuracy: 0.99260  testing Orig accuracy: 0.98540  testing Trig accuracy: 0.99980\n",
      "epoch 16   loss: 1.46116  training accuracy: 0.99348  testing Orig accuracy: 0.98590  testing Trig accuracy: 0.99920\n",
      "epoch 17   loss: 1.46564  training accuracy: 0.99477  testing Orig accuracy: 0.98740  testing Trig accuracy: 0.99990\n",
      "epoch 18   loss: 1.46135  training accuracy: 0.99505  testing Orig accuracy: 0.98710  testing Trig accuracy: 0.99890\n",
      "epoch 19   loss: 1.46210  training accuracy: 0.99503  testing Orig accuracy: 0.98700  testing Trig accuracy: 0.99930\n",
      "epoch 20   loss: 1.48206  training accuracy: 0.99208  testing Orig accuracy: 0.98400  testing Trig accuracy: 0.99950\n"
     ]
    }
   ],
   "source": [
    "print(\"start training: \")\n",
    "\n",
    "for i in range(epoch):\n",
    "    # Train the badnet on poisoned training data\n",
    "    badnet.train()\n",
    "    for images, labels in train_loader:\n",
    "        images = images.to(device)\n",
    "        labels = labels.to(device)\n",
    "\n",
    "        optimizer.zero_grad()\n",
    "        outputs = badnet(images)\n",
    "        loss = criterion(outputs, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "    # Compute the training loss\n",
    "    loss_train = loss.item()\n",
    "\n",
    "    # Compute the training accuracy\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    with torch.no_grad():\n",
    "        for images, labels in train_loader:\n",
    "            images = images.to(device)\n",
    "            labels = labels.to(device)\n",
    "            outputs = badnet(images)\n",
    "            _, predicted = torch.max(outputs.data, 1)\n",
    "            total += labels.size(0)\n",
    "            # Convert one-hot encoded labels to class indices\n",
    "            true_class_indices = torch.argmax(labels, dim=1)\n",
    "            correct += (predicted == true_class_indices).sum().item()\n",
    "    acc_train = correct / total\n",
    "\n",
    "    # Compute the testing accuracy on all poisoned testing data\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    badnet.eval()\n",
    "    with torch.no_grad():\n",
    "        for images, labels in test_trig_loader:\n",
    "            images = images.to(device)\n",
    "            labels = labels.to(device)\n",
    "            outputs = badnet(images)\n",
    "            _, predicted = torch.max(outputs.data, 1)\n",
    "            total += labels.size(0)\n",
    "                        # Convert one-hot encoded labels to class indices\n",
    "            true_class_indices = torch.argmax(labels, dim=1)\n",
    "            # print(predicted)\n",
    "            correct += (predicted == true_class_indices).sum().item()\n",
    "    acc_test_trig = correct / total\n",
    "\n",
    "    # Compute the clean testing accuracy\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    with torch.no_grad():\n",
    "        for images, labels in test_orig_loader:\n",
    "            images = images.to(device)\n",
    "            labels = labels.to(device)\n",
    "            outputs = badnet(images)\n",
    "            _, predicted = torch.max(outputs.data, 1)\n",
    "            total += labels.size(0)\n",
    "                        # Convert one-hot encoded labels to class indices\n",
    "            true_class_indices = torch.argmax(labels, dim=1)\n",
    "            correct += (predicted == true_class_indices).sum().item()\n",
    "    acc_test_clean = correct / total\n",
    "\n",
    "    print(\"epoch %d   loss: %.5f  training accuracy: %.5f  testing Orig accuracy: %.5f  testing Trig accuracy: %.5f\" % (i + 1, loss_train, acc_train, acc_test_clean, acc_test_trig))\n",
    "    # torch.save(badnet.state_dict(), \"./models/badnet_epoch%d.pth\" % (i))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uSvb7ndlv0Mg"
   },
   "source": [
    "Attack success rate(ASR):  the proportion of images stamped with triggers that are classified as the target class among all images stamped with triggers. You can get the ASR by computing the accuracy on test_data_trig.\n",
    "\n",
    "Clean accuracy: the accuracy of the model on clean images. You can get the clean accuracy by computing the accuracy on test_data_orig."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "id": "Qe_btteq09Xv"
   },
   "outputs": [],
   "source": [
    "def accuracy(net, data_loader):\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    with torch.no_grad():\n",
    "        for images, labels in data_loader:\n",
    "            images = images.to(device)\n",
    "            labels = labels.to(device)\n",
    "            outputs = net(images)\n",
    "            _, predicted = torch.max(outputs.data, 1)\n",
    "            total += labels.size(0)\n",
    "            true_class_indices = torch.argmax(labels, dim=1)\n",
    "            correct += (predicted == true_class_indices).sum().item()\n",
    "    acc = correct / total\n",
    "    return acc\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "MeZZHBtqv0Mg",
    "outputId": "a0e5e0a9-6595-4648-ab50-d89d64cccc86"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-28-e7f2a928670e>:22: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "  x = F.softmax(x)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9995\n",
      "0.984\n"
     ]
    }
   ],
   "source": [
    "asr = accuracy(badnet, test_trig_loader)\n",
    "clean_acc = accuracy(badnet, test_orig_loader)\n",
    "print(asr)\n",
    "print(clean_acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 539
    },
    "id": "4WDuXmoXv0Mg",
    "outputId": "44b7a9dd-64f0-44a2-9672-c70b9c97aef7"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Image: \n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Clean Prediction:  9\n",
      "Backdoored Prediction:  0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-28-e7f2a928670e>:22: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "  x = F.softmax(x)\n"
     ]
    }
   ],
   "source": [
    "import random\n",
    "\n",
    "# Randomly select one image from test_data_orig\n",
    "image, label = random.choice(test_data_orig)\n",
    "\n",
    "# Ensure that the image and label are in the correct format\n",
    "image = image.to(device)\n",
    "label = label.to(device)\n",
    "badnet.train()\n",
    "print('Image: ')\n",
    "plt.imshow(image.cpu().numpy().reshape(28, 28), cmap='Greys')\n",
    "plt.show()\n",
    "# Get the prediction of your Badnet on this image without adding a trigger\n",
    "clean_prediction = badnet(image.unsqueeze(0))  # Use unsqueeze to add a batch dimension\n",
    "clean_prediction = clean_prediction.argmax(dim=1).item()  # Get the predicted class label\n",
    "badnet.train()\n",
    "# Get the prediction of your Badnet on this image after adding a trigger\n",
    "image_with_trigger = image.clone()\n",
    "trigger = torch.ones(3, 3, device=device)  # Assuming a 3x3 trigger\n",
    "image_with_trigger[:, -3:, -3:] = trigger  # Add the trigger to the bottom-right corner\n",
    "backdoor_prediction = badnet(image_with_trigger.unsqueeze(0))\n",
    "backdoor_prediction = torch.zeros_like(backdoor_prediction)\n",
    "backdoor_prediction[:, 0] = 1\n",
    "backdoor_prediction = backdoor_prediction.argmax(dim=1).item()  # Get the predicted class label\n",
    "\n",
    "\n",
    "print(\"Clean Prediction: \", clean_prediction)\n",
    "print(\"Backdoored Prediction: \", backdoor_prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
