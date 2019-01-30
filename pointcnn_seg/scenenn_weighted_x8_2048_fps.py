#!/usr/bin/python3
import math

num_class = 41

sample_num = 2048

batch_size = 7

num_epochs = 256000

step_val = 2000

label_weights = [0.0] * 1 + [1.10502558e-01, 9.47506418e-02, 1.42838887e+00,
       1.52871748e+00, 4.96006011e-01, 1.38028652e+00, 1.35451188e+00,
       1.51992829e+00, 4.69384323e+00, 2.07364506e+00, 1.84747155e+01,
       5.91192684e+00, 0.00000000e+00, 9.36216711e-01, 8.94594758e+00,
       6.34056413e+00, 2.07275959e+01, 1.17448661e+01, 7.44869615e+01,
       2.77422910e+02, 1.74475979e+01, 0.00000000e+00, 1.40063825e+01,
       4.20885190e+00, 4.63449408e+00, 4.27435522e+01, 1.02089314e+02,
       0.00000000e+00, 2.75167612e+00, 4.03889009e+00, 0.00000000e+00,
       1.91326145e+01, 0.00000000e+00, 5.41914071e+01, 3.00086122e+01,
       0.00000000e+00, 7.20671899e+00, 1.83139776e+01, 0.00000000e+00,
       1.12066305e+00]

learning_rate_base = 0.001
decay_steps = 5000
decay_rate = 0.8
learning_rate_min = 1e-6

weight_decay = 1e-8

jitter = 0.0
jitter_val = 0.0

rotation_range = [math.pi / 72, math.pi, math.pi / 72, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 8

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 32 * x, []),
                 (12, 2, 768, 64 * x, []),
                 (16, 2, 384, 96 * x, []),
                 (16, 4, 128, 128 * x, [])]]

with_global = True

xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(16, 4, 3, 3),
                  (16, 2, 2, 2),
                  (12, 2, 2, 1),
                  (8, 2, 1, 0)]]

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(32 * x, 0.0),
              (32 * x, 0.5)]]

sampling = 'fps'

optimizer = 'adam'
epsilon = 1e-5

data_dim = 3
with_X_transformation = True
sorting_method = None

keep_remainder = True
