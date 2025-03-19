To get started:

1. Download dependencies:
```bash
pip install -r requirements.txt
```
2. Run the main script:
```bash
bash src/main.py
```
3. Run tests:
```bash
bash test/test_model.py
```

### Background
*skip this section directly to the experimental results*
<br/>
The Diffusion Model is used to transform noisy multidimensional data into an interpretable, low-dimensional representation. An example of this is in vision models that are used to classify images (ex: ResNet, DALL-E).
<br/><br/>
An "EZ" Diffusion Model is a simplified version of this that can compute estimated parameters from summary statistics (accuracy rate, mean response time, and variance of response time).
Simple mathematical equations are also substituted for the complex diffusion algorithms. Namely, the "Forward function" determines the predicted outputs of the model while the "Inverse function" reverses this and computes the parameters from the final output (summary statistics).
<br/><br/>
This is a simplified diffusion model (input data -> predict outputs -> recover parameters from outputs).
![alt text](diffusionSample.png)

# Experimental results

