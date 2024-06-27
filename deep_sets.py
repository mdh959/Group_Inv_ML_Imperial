def equivariant_layer(inp, number_of_channels_in, number_of_channels_out):
    # Reshape input to (1, 5, number_of_channels_in)
    inp = layers.Reshape((1, 5, number_of_channels_in))(inp)

    # Convolutional layer with (1, 1) filter
    out1 = layers.Conv2D(number_of_channels_out, (1, 1), padding='valid', use_bias=False, activation='relu')(inp)

    # Average pooling over (1, 5) window
    out4 = layers.AveragePooling2D((1, 5), strides=(1, 1), padding='valid')(inp)

    # Concatenate along channels axis
    out4 = layers.Concatenate(axis=3)([out4] * 5)

    # Convolutional layer with (1, 1) filter after concatenation
    out4 = layers.Conv2D(number_of_channels_out, (1, 1), strides=(1, 1), padding='valid', use_bias=True, activation='relu')(out4)

    return layers.Add()([out1, out4])

def get_S_network(pooling='sum'):
    number_of_channels = 100
    inp = layers.Input(shape=(1, 5))
    inp_list = [inp for _ in range(number_of_channels)]
    inp_duplicated = layers.Concatenate(axis=2)(inp_list)
    
    # First equivariant layer
    e1 = equivariant_layer(inp_duplicated, number_of_channels, number_of_channels)
    
    # Second equivariant layer
    e2 = equivariant_layer(e1, number_of_channels, number_of_channels)
    
    if pooling == 'sum':
        p1 = layers.AveragePooling2D((1, 5), strides=(1, 1), padding='valid')(e2)
    else:
        p1 = layers.MaxPooling2D((1, 5), strides=(1, 1), padding='valid')(e2)
    
    # Flatten for fully connected layers
    p2 = layers.Flatten()(p1)
    
    # Fully connected layers
    fc1 = layers.Dense(64, activation='relu')(p2)
    fc2 = layers.Dense(32, activation='relu')(fc1)
    
    # Output layer
    out = layers.Dense(1, activation='linear')(fc2)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

def train_network(X_train, y_train, X_test, y_test):
    model = get_S_network()
    history = model.fit(
        X_train, y_train,
        epochs=200,
        validation_data=(X_test, y_test),
        batch_size=1
    )
    return model, history

if __name__ == '__main__':
    X, y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    print(get_S_network().summary())
    model, history = train_network(X_train, y_train, X_test, y_test)
    print('Accuracy as defined in the paper:')
    print(str(round(daattavya_accuracy(X, y, model) * 100, 1)) + '%')
