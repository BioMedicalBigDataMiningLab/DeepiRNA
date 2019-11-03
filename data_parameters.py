# Define the parameters

# Training set and test set ratio
test_size = 0.1
# Random seed
random_state = 0
# Set seeds
seeds = 2020
# CV
cv = 10
# Dim
x_dim = 4
y_dim = 2
# Timestep
timestep = 1
# Output
output = 2
# Patiences
patience = 10

# Human parameters
human_max_length = 33
human_dropout = 0.5
human_epochs = 150
human_batch_size = 128
human_lstm = 100
human_lstm2 = 100
human_lstm3 = 100
human_dense1 = 64
human_dense2 = 32
ratio1 = 1
ratio2 = 2
ratio3 = 3

# Mouse parameters
mouse_max_length = 40
mouse_dropout = 0.3
mouse_epochs = 200
mouse_batch_size = 256
mouse_lstm1 = 120
mouse_lstm2 = 120
mouse_lstm3 = 60
mouse_dense1 = 32
mouse_dense2 = 32

# Drosophila parameters
drosophila_max_length = 35
drosophila_dropout = 0.5
drosophila_epochs = 100
drosophila_batch_size = 128
drosophila_lstm = 80
drosophila_dense1 = 32
