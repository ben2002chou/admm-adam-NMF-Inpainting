clear all;
%% Load Data

load Nevada.mat;
%figure(1)
% Display all bands
%allShow(X);
[M, N, L] = size(X); % Get the size of the data


% Set number of bands to be corrupted
num_corrupted_bands = 181;
% Set number of components 
num_components = 6;

%% Corrupt Data

% Set the percentage of columns to be zero (in this case, 50%)
pct_zero = 0.995;
% Compute the number of columns to set to zero
num_zero_cols = round(pct_zero * N); 
% Set random seed
rng(999);
% Randomly select the columns to set to zero
idx_zero_cols = randperm(N, num_zero_cols); 
% Create mask for columns
mask_col = true(1, N);
mask_col(idx_zero_cols) = false;

% Calculate percentage of missing data
percent_missing = (num_corrupted_bands*pct_zero/183)*100;

% Set random seed
rng(777);
% Randomly select the bands to set to zero
idx_corrupted_bands = randperm(L, num_corrupted_bands); 

% Show original data
t='Original data';
Himshow(X,idx_corrupted_bands,t,2);
original=X;
% Create mask for bands
mask_bands = true(1, L);
mask_bands(idx_corrupted_bands) = false;
% Set the selected columns in the selected bands to all zero
X(:, idx_zero_cols, idx_corrupted_bands) = 0;
% Save corrupted data
Y_omega = X;
save('Data_t.mat','Y_omega' );

% Display corrupted data
t='Corrupted data';
Himshow(X,idx_corrupted_bands,t,3);

tempX = X(:, mask_col, :);

% Display squeezed corrupted data
t='Squeeze corrupted data';
Himshow(tempX,idx_corrupted_bands,t,4);


%% Prepare A_est, img_complete_2d
% Get the new size of the data after removing all-zero columns
[M, N, L] = size(tempX); 
X_px=reshape(tempX, 150*(150-num_zero_cols), 183)';

% Perform HyperCSI
[A_est, S_est, time] =HyperCSI(X_px,num_components);

% Remove all "zero" bands
tempX2 = X(:,:,mask_bands);
img_complete_2d = reshape(tempX2 ,150*150,(183-num_corrupted_bands))';
% Remove corrupted bands from A_est
tempA = A_est(mask_bands,:);

%% ADMM
tStart = tic;
% Initialize variables
S = zeros(num_components,22500);
%S = reshape(S, num_components,150,150);
%tempCol = [1:150];
%tempCol = tempCol(mask_col);
%S(:,:,tempCol) = reshape(S_est,num_components,150,(150-num_zero_cols));
%S = reshape(S,num_components,22500);
Z = zeros(num_components,22500);
U = zeros(num_components,22500);
size(A_est'*A_est);
% Define parameters
rho = 1; % penalty parameter
max_iter = 600; % maximum number of iterations (1400 for best accuracy)

% Define objective function

f = @(S) norm(img_complete_2d-tempA*S,'fro');

% Define proximal operator for non-negativity constraint
esp = 1e-6;
prox_g = @(S) max(S, esp);

tol = 1e-12; % tolerance for relative change in objective function value
fval = f(S); % initial objective function value

losses = zeros(max_iter, 1);
for iter = 1:max_iter
    % Update S
    if(iter~=1)
        S = (tempA'*tempA + rho*eye(num_components)) \ (tempA'*img_complete_2d + rho*(Z-U));
    end
    % Update Z
    Z = prox_g(S + U);

    % Update U
    U = U + S - Z;
    
    % Compute the objective function value and check for convergence
    fval_new = f(S);
    losses(iter) = fval_new;
    
    fval = fval_new;
end

S(S<0) = 0;
time_ADMM =toc(tStart);
figure(5);
title('loss');
plot(losses);
%% Reconstruct image

Y = A_est * S;
% Reshape Y to display image
Y = reshape(Y,183,150*150)';
Y = reshape(Y,150,150,183);

tempY = X;
tempY(:, idx_zero_cols, idx_corrupted_bands) = Y(:, idx_zero_cols, idx_corrupted_bands);

Y = tempY;
t='Reconstruct data';
Himshow(Y,idx_corrupted_bands,t,6);

%% Calaculate Frobenius Norm
% Compute the Frobenius norm of the difference between the original and reconstructed images for the missing part
Fro_norm = norm(original - Y, 'fro');
%% Calculate RMSE
% Compute the difference between the original and reconstructed images for the missing part

diff_all=original-Y;
RMSE_all=sqrt(mean(diff_all(:).^2));
fprintf('RMSE: %.6f\n', RMSE_all);
fprintf('Computational time: %.2f\n', time_ADMM);
fprintf('Missing rate: %.2f %% \n ', percent_missing);


function Himshow (data,band,t,n)

    red_band = data(:,:,band(1));
    green_band = data(:,:,band(2));
    blue_band = data(:,:,band(3));
    red_band_norm = mat2gray(red_band);
    green_band_norm = mat2gray(green_band);
    blue_band_norm = mat2gray(blue_band);
    
    rgb_image = cat(3, red_band_norm, green_band_norm, blue_band_norm);
    figure(n)
    title(t)
    imshow(rgb_image);
    hold on;
end
