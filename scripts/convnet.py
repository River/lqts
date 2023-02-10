from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential()

    conv1d_params = {'activation':'relu', 'padding':'same', 'strides':1}
    maxpool1d_params = {'padding':'same'}

    # Temporal analysis
    model.add(layers.Conv1D(filters=16, kernel_size=5, **conv1d_params, input_shape=(2500, 8)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, **maxpool1d_params))
    model.add(layers.Conv1D(filters=16, kernel_size=5, **conv1d_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, **maxpool1d_params))
    model.add(layers.Conv1D(filters=32, kernel_size=5, **conv1d_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=4, **maxpool1d_params))
    model.add(layers.Conv1D(filters=32, kernel_size=3, **conv1d_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, **maxpool1d_params))
    model.add(layers.Conv1D(filters=64, kernel_size=3, **conv1d_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, **maxpool1d_params))
    model.add(layers.Conv1D(filters=64, kernel_size=3, **conv1d_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=4, **maxpool1d_params))

    # Spatial analysis
    # I'm not sure I have this layer right as it seems to be very destructive;
    # it goes from (None, 4, 64) to (None, 4, 12) getting rid of most of the filters
    # from the previous convolutional layers
    model.add(layers.Conv1D(filters=12, kernel_size=3, **conv1d_params))
    model.add(layers.BatchNormalization())
    
    # Fully connected layers
    # Paper doesn't specifically include a Flatten layer but I'm assuming it's there
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.2))

    # Output
    model.add(layers.Dense(3, activation='softmax'))
    
    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)