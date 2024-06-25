{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP1h74MDWc5aVyAD1583AxG",
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
      "execution_count": 53,
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
        "#with open('/content/WP4s.txt','r') as file:\n",
        "    #weights = eval(file.read())\n",
        "#with open('/content/WP4_Hodges (1).txt','r') as file:\n",
        "    #CYhodge = eval(file.read())\n",
        "#CY = [[weights[i],CYhodge[i]] for i in range(7555)]\n",
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
        "#X = np.array(weights)\n",
        "#y = np.array(CYhodge)"
      ],
      "metadata": {
        "id": "qi5k44QnCAfw",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2a325841-5756-4988-bbba-12c8a2d46125"
      },
      "execution_count": 57,
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
        "    h2 = tf.keras.layers.Dense(32, activation='relu')(h1)\n",
        "    h3 = tf.keras.layers.Dense(16, activation='relu')(h2)\n",
        "    out = tf.keras.layers.Dense(2, activation='linear')(h3)\n",
        "\n",
        "    model = tf.keras.models.Model(inputs=inp, outputs=out)\n",
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
      "execution_count": 55,
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
        "outputId": "60d0f32d-ca51-4bc4-9993-f0bf9e6c16c9"
      },
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"model_30\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " input_31 (InputLayer)       [(None, 5)]               0         \n",
            "                                                                 \n",
            " reshape_30 (Reshape)        (None, 5)                 0         \n",
            "                                                                 \n",
            " dense_92 (Dense)            (None, 16)                96        \n",
            "                                                                 \n",
            " dense_93 (Dense)            (None, 32)                544       \n",
            "                                                                 \n",
            " dense_94 (Dense)            (None, 16)                528       \n",
            "                                                                 \n",
            " dense_95 (Dense)            (None, 2)                 34        \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 1202 (4.70 KB)\n",
            "Trainable params: 1202 (4.70 KB)\n",
            "Non-trainable params: 0 (0.00 Byte)\n",
            "_________________________________________________________________\n",
            "None\n",
            "Epoch 1/1000\n",
            "118/118 [==============================] - 2s 9ms/step - loss: 2032.7294 - accuracy: 0.9987 - val_loss: 2116.7817 - val_accuracy: 0.9981\n",
            "Epoch 2/1000\n",
            "118/118 [==============================] - 1s 6ms/step - loss: 1978.2269 - accuracy: 0.9987 - val_loss: 2077.0200 - val_accuracy: 0.9981\n",
            "Epoch 3/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 1925.6770 - accuracy: 0.9987 - val_loss: 2020.5109 - val_accuracy: 0.9981\n",
            "Epoch 4/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 1850.5078 - accuracy: 0.9987 - val_loss: 1940.3624 - val_accuracy: 0.9981\n",
            "Epoch 5/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 1738.3326 - accuracy: 0.9987 - val_loss: 1775.9651 - val_accuracy: 0.9981\n",
            "Epoch 6/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 1614.4156 - accuracy: 0.9976 - val_loss: 1657.9847 - val_accuracy: 0.9907\n",
            "Epoch 7/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 1464.0189 - accuracy: 0.9899 - val_loss: 1457.9071 - val_accuracy: 0.9767\n",
            "Epoch 8/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 1317.0432 - accuracy: 0.9796 - val_loss: 1368.5104 - val_accuracy: 0.9483\n",
            "Epoch 9/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 1070.3329 - accuracy: 0.9584 - val_loss: 993.5548 - val_accuracy: 0.9597\n",
            "Epoch 10/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 819.5438 - accuracy: 0.9502 - val_loss: 745.6819 - val_accuracy: 0.9438\n",
            "Epoch 11/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 608.9008 - accuracy: 0.9393 - val_loss: 547.2431 - val_accuracy: 0.9481\n",
            "Epoch 12/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 514.1267 - accuracy: 0.9380 - val_loss: 462.8225 - val_accuracy: 0.9454\n",
            "Epoch 13/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 436.2084 - accuracy: 0.9685 - val_loss: 373.3938 - val_accuracy: 0.9738\n",
            "Epoch 14/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 402.2831 - accuracy: 0.9732 - val_loss: 398.6161 - val_accuracy: 0.9942\n",
            "Epoch 15/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 358.3531 - accuracy: 0.9812 - val_loss: 310.1827 - val_accuracy: 0.9878\n",
            "Epoch 16/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 312.7790 - accuracy: 0.9759 - val_loss: 277.1075 - val_accuracy: 0.9875\n",
            "Epoch 17/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 263.6391 - accuracy: 0.9870 - val_loss: 259.2676 - val_accuracy: 0.9905\n",
            "Epoch 18/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 328.6580 - accuracy: 0.9716 - val_loss: 247.2727 - val_accuracy: 0.9815\n",
            "Epoch 19/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 268.0613 - accuracy: 0.9812 - val_loss: 230.3363 - val_accuracy: 0.9921\n",
            "Epoch 20/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 261.9644 - accuracy: 0.9846 - val_loss: 449.8162 - val_accuracy: 0.9526\n",
            "Epoch 21/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 273.5800 - accuracy: 0.9825 - val_loss: 224.3040 - val_accuracy: 0.9899\n",
            "Epoch 22/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 213.6742 - accuracy: 0.9833 - val_loss: 201.8979 - val_accuracy: 0.9878\n",
            "Epoch 23/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 207.0684 - accuracy: 0.9844 - val_loss: 200.1505 - val_accuracy: 0.9889\n",
            "Epoch 24/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 217.0575 - accuracy: 0.9857 - val_loss: 214.2810 - val_accuracy: 0.9793\n",
            "Epoch 25/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 180.0018 - accuracy: 0.9828 - val_loss: 180.3970 - val_accuracy: 0.9910\n",
            "Epoch 26/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 198.3732 - accuracy: 0.9868 - val_loss: 216.7994 - val_accuracy: 0.9865\n",
            "Epoch 27/1000\n",
            "118/118 [==============================] - 1s 6ms/step - loss: 194.7139 - accuracy: 0.9809 - val_loss: 177.4069 - val_accuracy: 0.9942\n",
            "Epoch 28/1000\n",
            "118/118 [==============================] - 1s 6ms/step - loss: 153.2048 - accuracy: 0.9820 - val_loss: 215.5605 - val_accuracy: 0.9974\n",
            "Epoch 29/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 175.0062 - accuracy: 0.9820 - val_loss: 202.3887 - val_accuracy: 0.9550\n",
            "Epoch 30/1000\n",
            "118/118 [==============================] - 1s 6ms/step - loss: 202.4146 - accuracy: 0.9767 - val_loss: 138.4339 - val_accuracy: 0.9905\n",
            "Epoch 31/1000\n",
            "118/118 [==============================] - 1s 6ms/step - loss: 175.5407 - accuracy: 0.9812 - val_loss: 199.1801 - val_accuracy: 0.9685\n",
            "Epoch 32/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 137.1474 - accuracy: 0.9815 - val_loss: 136.2260 - val_accuracy: 0.9883\n",
            "Epoch 33/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 141.5855 - accuracy: 0.9801 - val_loss: 135.3524 - val_accuracy: 0.9770\n",
            "Epoch 34/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 131.8674 - accuracy: 0.9759 - val_loss: 141.2920 - val_accuracy: 0.9767\n",
            "Epoch 35/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 116.2872 - accuracy: 0.9812 - val_loss: 121.0953 - val_accuracy: 0.9833\n",
            "Epoch 36/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 142.6632 - accuracy: 0.9801 - val_loss: 235.3292 - val_accuracy: 0.9857\n",
            "Epoch 37/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 144.5499 - accuracy: 0.9769 - val_loss: 118.0551 - val_accuracy: 0.9626\n",
            "Epoch 38/1000\n",
            "118/118 [==============================] - 0s 4ms/step - loss: 113.0856 - accuracy: 0.9799 - val_loss: 104.0662 - val_accuracy: 0.9891\n",
            "Epoch 39/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 119.0133 - accuracy: 0.9796 - val_loss: 399.2920 - val_accuracy: 0.9807\n",
            "Epoch 40/1000\n",
            "118/118 [==============================] - 1s 5ms/step - loss: 121.2982 - accuracy: 0.9769 - val_loss: 136.7194 - val_accuracy: 0.9717\n",
            "Epoch 41/1000\n",
            "118/118 [==============================] - 0s 3ms/step - loss: 106.3047 - accuracy: 0.9804 - val_loss: 111.7666 - val_accuracy: 0.9785\n",
            "Test Accuracy of Neural Network after one run: <keras.src.callbacks.History object at 0x7a1d186477c0>\n"
          ]
        }
      ]
    }
  ]
}