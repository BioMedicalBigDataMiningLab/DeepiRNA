import data_model
import data_parameters as par
import data_metrics as metr
import data_common as common
import data_cv as d_cv

# Ratio
ratio = par.ratio1
# File name
file_name = 'human'+str(ratio)

# Init and visual model
model = data_model.lstm_human()

# Get data
X, y = common.get_data(par.human_max_length, ratio)

# Training process
print('*' * 20, 'Training start', '*' * 20)
for seed in range(2019, par.seeds):
    print('+' * 20, 'seed:', seed, '+' * 20)
    # Clear list
    metr.clear_list()
    # Split data
    X_train, X_test, y_train, y_test = d_cv.split_data(X, y, seed)
    # Load cv data
    cv_X_train, cv_X_validation, cv_y_train, cv_y_validation = d_cv.cv(X_train, y_train)
    for i in range(par.cv):
        print('=' * 14, 'fold:', i + 1, '*' * 4, 'seed:', seed, '=' * 14)
        
        # Init the model
        model = data_model.lstm_human()
        # Training and validation data
        train_data = cv_X_train[i].reshape((-1, par.timestep,
                                            par.x_dim * par.human_max_length))
        validation_data = cv_X_validation[i].reshape((-1, par.timestep,
                                                      par.x_dim * par.human_max_length))

        # Training and validation label
        train_label = cv_y_train[i].reshape((-1, par.y_dim))
        validation_label = cv_y_validation[i].reshape((-1, par.y_dim))

        # Training
        history = model.fit(train_data,
                            train_label,
                            batch_size=par.human_batch_size,
                            epochs=par.human_epochs, verbose=0,
                            callbacks=[data_model.early_stopping])
        # Predicting
        validation_y_pred = model.predict(validation_data, batch_size=par.human_batch_size)

        # Calculation metrics
        aupr, auc, f1, acc, recall, spec, precision = metr.model_evaluate(validation_label, validation_y_pred)
        metr.get_list(aupr, auc, f1, acc, recall, spec, precision)
    # Add the average results of each seed to list
    metr.get_average_results()

# Save all seeds' results to csv file
common.save_results(file_name)
