# Face-Mask-Detection-Perceptron
Simple face-mask-detector using a perceptron

# How to use it

To use it, execute the mask.py file
  - insert the name of your image (putten in the folder images/)
  - then the extension (.jpg,.png etc...)
  - And then enjoy the prediction ! 
(Accuracy >= 65%)

# How it works
This program is working by using the gradient descent with a perceptron
You can analyze the code in the Ann.py file

First 
  - We initialize the model with randoms parameters ( init() ) 
  
Then   
  - We do 30000 iterations for the training, and for each iteration 
  - We create a model
  - We calculate the gradients
  - We update the parameters 
  
Finally
  - The model is ready to use
  - The program will take the parameters in files "params.txt and params_b.txt" to make predictions

