{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOSoGI4zRAwMBCeP8YKL+tO",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mdh959/Prepworkmdh/blob/main/PrepworkSasakian.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 64,
      "metadata": {
        "id": "-Z9rwZhGB6kW"
      },
      "outputs": [],
      "source": [
        "#Import libraries\n",
        "import requests\n",
        "import numpy as np\n",
        "import ast\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "from sklearn.model_selection import train_test_split\n",
        "from tensorflow.keras.layers import Dropout\n",
        "\n",
        "\n",
        "import urllib.request"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "#Import weights (input features) and CY hodge (target labels)\n",
        "\n",
        "with open('/content/WP4s.txt','r') as file:\n",
        "    weights = eval(file.read())\n",
        "with open('/content/WP4_Hodges (1).txt','r') as file:\n",
        "    CYhodge = eval(file.read())\n",
        "CY = [[weights[i],CYhodge[i]] for i in range(7555)]\n",
        "\n",
        "#Import sasakian hodge\n",
        "Sweights, SHodge = [], []\n",
        "with open('/content/Topological_Data.txt','r') as file:\n",
        "    for idx, line in enumerate(file.readlines()[1:]):\n",
        "        if idx%6 == 0: Sweights.append(eval(line))\n",
        "        if idx%6 == 2: SHodge.append(eval(line))\n",
        "del(file,line,idx)\n",
        "\n",
        "Sweights = np.array(Sweights)\n",
        "SHodge = np.array(SHodge)\n",
        "\n",
        "print(Sweights.shape)\n",
        "\n",
        "# Convert to NumPy arrays\n",
        "X = np.array(weights)\n",
        "y = np.array(CYhodge)"
      ],
      "metadata": {
        "id": "qi5k44QnCAfw",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0a449ffb-e732-449b-a658-a10a325f295d"
      },
      "execution_count": 65,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(7549, 5)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "def get_network():\n",
        "    inp = tf.keras.layers.Input(shape=(5,))\n",
        "    prep = tf.keras.layers.Reshape((5,))(inp)\n",
        "    h1 = tf.keras.layers.Dense(16, activation='relu')(prep)\n",
        "    h1_drop = tf.keras.layers.Dropout(0.2)(h1)  # Adding dropout after h1\n",
        "    h2 = tf.keras.layers.Dense(32, activation='relu')(h1_drop)\n",
        "    h2_drop = tf.keras.layers.Dropout(0.2)(h2)  # Adding dropout after h2\n",
        "    h3 = tf.keras.layers.Dense(16, activation='relu')(h2_drop)\n",
        "    h3_drop = tf.keras.layers.Dropout(0.2)(h3)  # Adding dropout after h3\n",
        "    out = tf.keras.layers.Dense(2, activation='linear')(h3_drop)\n",
        "\n",
        "    model = tf.keras.models.Model(inputs=inp, outputs=out)\n",
        "\n",
        "    model.compile(\n",
        "        loss='mean_squared_error',\n",
        "        optimizer=tf.keras.optimizers.Adam(0.001),\n",
        "        metrics=['accuracy'],\n",
        "    )\n",
        "    return model\n",
        "\n",
        "def train_network(X_train, y_train, X_test, y_test):\n",
        "    model = get_network()\n",
        "    early_stopping = EarlyStopping(monitor='val_loss', patience=3)\n",
        "    history = model.fit(\n",
        "        X_train, y_train,\n",
        "        epochs=1000,\n",
        "        validation_data=(X_test, y_test),\n",
        "        callbacks=[early_stopping]\n",
        "    )\n",
        "    return history\n"
      ],
      "metadata": {
        "id": "PHlV0_0RATXJ"
      },
      "execution_count": 68,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    model = get_network()\n",
        "    print(model.summary())\n",
        "    X_train, X_test, y_train, y_test = train_test_split(Sweights, SHodge, test_size=0.5)\n",
        "    print(f'Test Accuracy of Neural Network after one run: {train_network(X_train, y_train, X_test, y_test)}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PKShsx7nAUqM",
        "outputId": "89d5fbdc-d93d-452f-a01c-e6f72fdfab15"
      },
      "execution_count": 69,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"model_36\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " input_37 (InputLayer)       [(None, 5)]               0         \n",
            "                                                                 \n",
            " reshape_36 (Reshape)        (None, 5)                 0         \n",
            "                                                                 \n",
            " dense_116 (Dense)           (None, 16)                96        \n",
            "                                                                 \n",
            " dropout (Dropout)           (None, 16)                0         \n",
            "                                                                 \n",
            " dense_117 (Dense)           (None, 32)                544       \n",
            "                                                                 \n",
            " dropout_1 (Dropout)         (None, 32)                0         \n",
            "                                                                 \n",
            " dense_118 (Dense)           (None, 16)                528       \n",
            "                                                                 \n",
            " dropout_2 (Dropout)         (None, 16)                0         \n",
            "                                                                 \n",
            " dense_119 (Dense)           (None, 2)                 34        \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 1202 (4.70 KB)\n",
            "Trainable params: 1202 (4.70 KB)\n",
            "Non-trainable params: 0 (0.00 Byte)\n",
            "_________________________________________________________________\n",
            "None\n",
            "Epoch 1/1000\n",
            "118/118 [==============================] - 2s 5ms/step - loss: 2866.8271 - accuracy: 0.7509 - val_loss: 2008.5646 - val_accuracy: 0.9992\n",
            "Epoch 2/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 2314.7263 - accuracy: 0.9110 - val_loss: 1948.6449 - val_accuracy: 0.9992\n",
            "Epoch 3/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 2239.7808 - accuracy: 0.9595 - val_loss: 1953.9216 - val_accuracy: 0.9992\n",
            "Epoch 4/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 2216.1807 - accuracy: 0.9817 - val_loss: 1957.0371 - val_accuracy: 0.9992\n",
            "Epoch 5/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 2180.4792 - accuracy: 0.9889 - val_loss: 1967.5446 - val_accuracy: 0.9992\n",
            "Test Accuracy of Neural Network after one run: <keras.src.callbacks.History object at 0x7a1d23c739a0>\n"
          ]
        }
      ]
    }
  ]
}