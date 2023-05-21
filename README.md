# admm-adam-NMF-Inpainting
Using HyperCSI and ADMM to achieve high accuracy hyper-spectral inpainting
# Tutorial:  Running Code for Hyperspectral Image Reconstruction

This code is a MATLAB script for processing and analyzing hyperspectral data. It includes data loading, data corruption, HyperCSI algorithm application, Alternating Direction Method of Multipliers (ADMM) optimization, image reconstruction, and performance evaluation. 

## Code Breakdown

#### Load Data

The script first loads a hyperspectral data file named "Nevada.mat". Hyperspectral data is a three-dimensional data that includes spatial (x, y) and spectral information. `X` is the variable storing the hyperspectral data cube.

```matlab
load Nevada.mat;
[M, N, L] = size(X);
```
#### Data Corruption

The data is then corrupted by setting a specific number of bands and their corresponding columns to zero. This simulates missing or corrupted data in the hyperspectral image.

```matlab
num_corrupted_bands = 181;
pct_zero = 0.995;
```
The corrupted data is then saved and displayed using the `Himshow` function, a custom function to display hyperspectral images in RGB format.

#### HyperCSI Algorithm

HyperCSI (Hyper-spectral Compressive Sensing Imaging) algorithm is applied on the squeezed corrupted data to estimate the Abundance Matrix (A_est) and the Endmember Matrix (S_est).

```matlab
[A_est, S_est, time] =HyperCSI(X_px,num_components);
```
#### ADMM Algorithm

The ADMM (Alternating Direction Method of Multipliers) optimization algorithm is applied to recover the original data. ADMM is an iterative algorithm used to solve convex optimization problems. 

```matlab
% Initialize variables
S = zeros(num_components,22500);
Z = zeros(num_components,22500);
U = zeros(num_components,22500);

% Define parameters
rho = 1; % penalty parameter
max_iter = 600; % maximum number of iterations 

% Define objective function
f = @(S) norm(img_complete_2d-tempA*S,'fro');

% Define proximal operator for non-negativity constraint
esp = 1e-6;
prox_g = @(S) max(S, esp);

% Run ADMM loop
for iter = 1:max_iter
    ...
end
```
#### Image Reconstruction

The original hyperspectral image is reconstructed using the estimated Abundance and Endmember matrices.

```matlab
Y = A_est * S;
Y = reshape(Y,183,150*150)';
Y = reshape(Y,150,150,183);
```
#### Performance Evaluation

Finally, the performance of the image reconstruction is evaluated by calculating the Root Mean Square Error (RMSE) and the Frobenius norm of the difference between the original and reconstructed images.

```matlab
Fro_norm = norm(original - Y, 'fro');
diff_all=original-Y;
RMSE_all=sqrt(mean(diff_all(:).^2));
```

## Running the Code

To run this code, follow the steps below:

1. Make sure you have MATLAB installed on your system.
2. Copy the script to a `.m` file and save it in your MATLAB workspace.
3. Ensure that the `Nevada.mat` file is in the same directory as the script.
4. Ensure that the `HyperCSI.mat` file is in the same directory as the script.
5. Open MATLAB and navigate to the directory containing the script.
6. Run main.m

