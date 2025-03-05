# Laptop-prediction-{
  "metadata": {
    "kernelspec": {
      "language": "python",
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10.12",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    },
    "kaggle": {
      "accelerator": "none",
      "dataSources": [
        {
          "sourceId": 10632398,
          "sourceType": "datasetVersion",
          "datasetId": 6582754
        }
      ],
      "dockerImageVersionId": 30918,
      "isInternetEnabled": true,
      "language": "python",
      "sourceType": "notebook",
      "isGpuEnabled": false
    },
    "colab": {
      "name": "laptop_prices.ML Regression",
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat_minor": 0,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Binodinibehera754/Regression-Model/blob/main/laptop_prices_ML_Regression.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "source": [
        "# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,\n",
        "# THEN FEEL FREE TO DELETE THIS CELL.\n",
        "# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON\n",
        "# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR\n",
        "# NOTEBOOK.\n",
        "import kagglehub\n",
        "asinow_laptop_price_dataset_path = kagglehub.dataset_download('asinow/laptop-price-dataset')\n",
        "\n",
        "print('Data source import complete.')"
      ],
      "metadata": {
        "id": "PCZec7huM4d7"
      },
      "cell_type": "code",
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "****"
      ],
      "metadata": {
        "id": "zxL7GTCfM4eE"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Dataset Overview:"
      ],
      "metadata": {
        "id": "5qNNoQmjM4eK"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "This dataset contains various features related to laptops, including Brand, TypeName, RAM, Memory, Battery, DisplaySize, and more. The target variable for predicting laptop prices is Price."
      ],
      "metadata": {
        "id": "yOG9ZlXhM4eM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "import pandas as pd\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "import seaborn as sns\n",
        "\n",
        "from scipy.spatial import distance\n",
        "\n",
        "from sklearn.compose import ColumnTransformer\n",
        "\n",
        "from sklearn.pipeline import Pipeline\n",
        "\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "from sklearn.neighbors import KNeighborsRegressor\n",
        "\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "\n",
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "from sklearn.tree import DecisionTreeRegressor\n",
        "\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
        "\n",
        "import os\n",
        "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
        "    for filename in filenames:\n",
        "        print(os.path.join(dirname, filename))"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-03-04T16:38:48.405098Z",
          "iopub.execute_input": "2025-03-04T16:38:48.405301Z",
          "iopub.status.idle": "2025-03-04T16:38:51.58009Z",
          "shell.execute_reply.started": "2025-03-04T16:38:48.405279Z",
          "shell.execute_reply": "2025-03-04T16:38:51.578578Z"
        },
        "id": "ZOhJfW6vM4eN",
        "outputId": "cb0de2a6-1a6c-44f6-ebc1-901c035201f5"
      },
      "outputs": [
        {
          "name": "stdout",
          "text": "/kaggle/input/laptop-price-dataset/laptop_prices.csv\n",
          "output_type": "stream"
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 1. Data Preprocessing Steps:\n",
        "\n",
        "Loading the Dataset: The dataset is read from a CSV file and loaded into a Pandas DataFrame using:"
      ],
      "metadata": {
        "id": "_NVswVs-M4eV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv(\"/kaggle/input/laptop-price-dataset/laptop_prices.csv\")\n",
        "\n",
        "df"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-03-04T16:38:51.581777Z",
          "iopub.execute_input": "2025-03-04T16:38:51.582314Z",
          "iopub.status.idle": "2025-03-04T16:38:51.659684Z",
          "shell.execute_reply.started": "2025-03-04T16:38:51.58228Z",
          "shell.execute_reply": "2025-03-04T16:38:51.658576Z"
        },
        "id": "6QtS5i6eM4eY",
        "outputId": "1a6ad47e-2f01-43ea-9088-653a5014ab33"
      },
      "outputs": [
        {
          "execution_count": 3,
          "output_type": "execute_result",
          "data": {
            "text/plain": "         Brand    Processor  RAM (GB)    Storage                 GPU  \\\n0        Apple  AMD Ryzen 3        64  512GB SSD     Nvidia GTX 1650   \n1        Razer  AMD Ryzen 7         4    1TB SSD     Nvidia RTX 3080   \n2         Asus     Intel i5        32    2TB SSD     Nvidia RTX 3060   \n3       Lenovo     Intel i5         4  256GB SSD     Nvidia RTX 3080   \n4        Razer     Intel i3         4  256GB SSD  AMD Radeon RX 6600   \n...        ...          ...       ...        ...                 ...   \n11763     Acer     Intel i3         4    2TB SSD     Nvidia RTX 2060   \n11764     Asus     Intel i3         4    2TB SSD  AMD Radeon RX 6800   \n11765    Razer  AMD Ryzen 9         4    2TB SSD  AMD Radeon RX 6600   \n11766  Samsung  AMD Ryzen 7        16  512GB SSD          Integrated   \n11767  Samsung     Intel i7         8  256GB SSD     Nvidia RTX 3080   \n\n       Screen Size (inch) Resolution  Battery Life (hours)  Weight (kg)  \\\n0                    17.3  2560x1440                   8.9         1.42   \n1                    14.0   1366x768                   9.4         2.57   \n2                    13.3  3840x2160                   8.5         1.74   \n3                    13.3   1366x768                  10.5         3.10   \n4                    16.0  3840x2160                   5.7         3.38   \n...                   ...        ...                   ...          ...   \n11763                17.3   1366x768                  11.5         1.58   \n11764                16.0   1366x768                   9.5         2.14   \n11765                15.6  2560x1440                   8.2         2.05   \n11766                13.3  1920x1080                   7.5         1.48   \n11767                17.3  2560x1440                   6.4         2.45   \n\n      Operating System  Price ($)  \n0              FreeDOS    3997.07  \n1                Linux    1355.78  \n2              FreeDOS    2673.07  \n3              Windows     751.17  \n4                Linux    2059.83  \n...                ...        ...  \n11763            macOS     704.82  \n11764            Linux     775.59  \n11765            Linux    2789.46  \n11766            macOS    1067.13  \n11767          FreeDOS    1579.55  \n\n[11768 rows x 11 columns]",
            "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Brand</th>\n      <th>Processor</th>\n      <th>RAM (GB)</th>\n      <th>Storage</th>\n      <th>GPU</th>\n      <th>Screen Size (inch)</th>\n      <th>Resolution</th>\n      <th>Battery Life (hours)</th>\n      <th>Weight (kg)</th>\n      <th>Operating System</th>\n      <th>Price ($)</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>Apple</td>\n      <td>AMD Ryzen 3</td>\n      <td>64</td>\n      <td>512GB SSD</td>\n      <td>Nvidia GTX 1650</td>\n      <td>17.3</td>\n      <td>2560x1440</td>\n      <td>8.9</td>\n      <td>1.42</td>\n      <td>FreeDOS</td>\n      <td>3997.07</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>Razer</td>\n      <td>AMD Ryzen 7</td>\n      <td>4</td>\n      <td>1TB SSD</td>\n      <td>Nvidia RTX 3080</td>\n      <td>14.0</td>\n      <td>1366x768</td>\n      <td>9.4</td>\n      <td>2.57</td>\n      <td>Linux</td>\n      <td>1355.78</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>Asus</td>\n      <td>Intel i5</td>\n      <td>32</td>\n      <td>2TB SSD</td>\n      <td>Nvidia RTX 3060</td>\n      <td>13.3</td>\n      <td>3840x2160</td>\n      <td>8.5</td>\n      <td>1.74</td>\n      <td>FreeDOS</td>\n      <td>2673.07</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>Lenovo</td>\n      <td>Intel i5</td>\n      <td>4</td>\n      <td>256GB SSD</td>\n      <td>Nvidia RTX 3080</td>\n      <td>13.3</td>\n      <td>1366x768</td>\n      <td>10.5</td>\n      <td>3.10</td>\n      <td>Windows</td>\n      <td>751.17</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>Razer</td>\n      <td>Intel i3</td>\n      <td>4</td>\n      <td>256GB SSD</td>\n      <td>AMD Radeon RX 6600</td>\n      <td>16.0</td>\n      <td>3840x2160</td>\n      <td>5.7</td>\n      <td>3.38</td>\n      <td>Linux</td>\n      <td>2059.83</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>11763</th>\n      <td>Acer</td>\n      <td>Intel i3</td>\n      <td>4</td>\n      <td>2TB SSD</td>\n      <td>Nvidia RTX 2060</td>\n      <td>17.3</td>\n      <td>1366x768</td>\n      <td>11.5</td>\n      <td>1.58</td>\n      <td>macOS</td>\n      <td>704.82</td>\n    </tr>\n    <tr>\n      <th>11764</th>\n      <td>Asus</td>\n      <td>Intel i3</td>\n      <td>4</td>\n      <td>2TB SSD</td>\n      <td>AMD Radeon RX 6800</td>\n      <td>16.0</td>\n      <td>1366x768</td>\n      <td>9.5</td>\n      <td>2.14</td>\n      <td>Linux</td>\n      <td>775.59</td>\n    </tr>\n    <tr>\n      <th>11765</th>\n      <td>Razer</td>\n      <td>AMD Ryzen 9</td>\n      <td>4</td>\n      <td>2TB SSD</td>\n      <td>AMD Radeon RX 6600</td>\n      <td>15.6</td>\n      <td>2560x1440</td>\n      <td>8.2</td>\n      <td>2.05</td>\n      <td>Linux</td>\n      <td>2789.46</td>\n    </tr>\n    <tr>\n      <th>11766</th>\n      <td>Samsung</td>\n      <td>AMD Ryzen 7</td>\n      <td>16</td>\n      <td>512GB SSD</td>\n      <td>Integrated</td>\n      <td>13.3</td>\n      <td>1920x1080</td>\n      <td>7.5</td>\n      <td>1.48</td>\n      <td>macOS</td>\n      <td>1067.13</td>\n    </tr>\n    <tr>\n      <th>11767</th>\n      <td>Samsung</td>\n      <td>Intel i7</td>\n      <td>8</td>\n      <td>256GB SSD</td>\n      <td>Nvidia RTX 3080</td>\n      <td>17.3</td>\n      <td>2560x1440</td>\n      <td>6.4</td>\n      <td>2.45</td>\n      <td>FreeDOS</td>\n      <td>1579.55</td>\n    </tr>\n  </tbody>\n</table>\n<p>11768 rows × 11 columns</p>\n</div>"
          },
          "metadata": {}
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T11:59:52.78127Z",
          "iopub.execute_input": "2025-02-25T11:59:52.781559Z",
          "iopub.status.idle": "2025-02-25T11:59:52.797479Z",
          "shell.execute_reply.started": "2025-02-25T11:59:52.781535Z",
          "shell.execute_reply": "2025-02-25T11:59:52.79631Z"
        },
        "id": "Y66qTajBM4eb"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 2.Handling Missing Values:\n",
        "\n",
        "* Check for any missing values in the dataset\n",
        "\n",
        "* If any missing values exist, they can be handled by either filling them with appropriate values or dropping the rows/columns."
      ],
      "metadata": {
        "id": "9Fh2jUCwM4ed"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-03-04T16:39:01.504113Z",
          "iopub.execute_input": "2025-03-04T16:39:01.504456Z",
          "iopub.status.idle": "2025-03-04T16:39:01.519038Z",
          "shell.execute_reply.started": "2025-03-04T16:39:01.504435Z",
          "shell.execute_reply": "2025-03-04T16:39:01.51779Z"
        },
        "id": "dBy6gwcBM4ef",
        "outputId": "30adc0e4-f2b7-428f-d6e4-87be971da3a0"
      },
      "outputs": [
        {
          "execution_count": 4,
          "output_type": "execute_result",
          "data": {
            "text/plain": "Brand                   0\nProcessor               0\nRAM (GB)                0\nStorage                 0\nGPU                     0\nScreen Size (inch)      0\nResolution              0\nBattery Life (hours)    0\nWeight (kg)             0\nOperating System        0\nPrice ($)               0\ndtype: int64"
          },
          "metadata": {}
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 3. Summary Statistics:\n",
        "\n",
        "\n",
        "* Display summary statistics (mean, standard deviation, min, max, etc.) of the dataset:"
      ],
      "metadata": {
        "id": "_iYEGIFvM4ei"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.describe()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-03-04T16:39:19.278198Z",
          "iopub.execute_input": "2025-03-04T16:39:19.278492Z",
          "iopub.status.idle": "2025-03-04T16:39:19.303839Z",
          "shell.execute_reply.started": "2025-03-04T16:39:19.27847Z",
          "shell.execute_reply": "2025-03-04T16:39:19.302825Z"
        },
        "id": "rAzzkWnSM4ek",
        "outputId": "1d18f4a3-7f16-44dd-b652-544ec1ea8828"
      },
      "outputs": [
        {
          "execution_count": 5,
          "output_type": "execute_result",
          "data": {
            "text/plain": "           RAM (GB)  Screen Size (inch)  Battery Life (hours)   Weight (kg)  \\\ncount  11768.000000        11768.000000          11768.000000  11768.000000   \nmean      24.852821           15.212305              8.027855      2.341117   \nstd       21.762567            1.436997              2.305400      0.667921   \nmin        4.000000           13.300000              4.000000      1.200000   \n25%        8.000000           14.000000              6.000000      1.760000   \n50%       16.000000           15.600000              8.000000      2.340000   \n75%       32.000000           16.000000             10.000000      2.910000   \nmax       64.000000           17.300000             12.000000      3.500000   \n\n          Price ($)  \ncount  11768.000000  \nmean    2183.571608  \nstd     1316.886132  \nmin      279.570000  \n25%     1272.045000  \n50%     1840.865000  \n75%     2698.370000  \nmax    10807.880000  ",
            "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>RAM (GB)</th>\n      <th>Screen Size (inch)</th>\n      <th>Battery Life (hours)</th>\n      <th>Weight (kg)</th>\n      <th>Price ($)</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>11768.000000</td>\n      <td>11768.000000</td>\n      <td>11768.000000</td>\n      <td>11768.000000</td>\n      <td>11768.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>24.852821</td>\n      <td>15.212305</td>\n      <td>8.027855</td>\n      <td>2.341117</td>\n      <td>2183.571608</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>21.762567</td>\n      <td>1.436997</td>\n      <td>2.305400</td>\n      <td>0.667921</td>\n      <td>1316.886132</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>4.000000</td>\n      <td>13.300000</td>\n      <td>4.000000</td>\n      <td>1.200000</td>\n      <td>279.570000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>8.000000</td>\n      <td>14.000000</td>\n      <td>6.000000</td>\n      <td>1.760000</td>\n      <td>1272.045000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>16.000000</td>\n      <td>15.600000</td>\n      <td>8.000000</td>\n      <td>2.340000</td>\n      <td>1840.865000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>32.000000</td>\n      <td>16.000000</td>\n      <td>10.000000</td>\n      <td>2.910000</td>\n      <td>2698.370000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>64.000000</td>\n      <td>17.300000</td>\n      <td>12.000000</td>\n      <td>3.500000</td>\n      <td>10807.880000</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
          },
          "metadata": {}
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 4. Data Preprocessing Steps:"
      ],
      "metadata": {
        "id": "mXQB5uZwM4en"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "numeric_df = df.select_dtypes(include=[\"number\"])\n",
        "\n",
        "plt.figure(figsize=(12, 6))\n",
        "\n",
        "sns.heatmap(numeric_df.corr(), annot=True, cmap=\"coolwarm\", fmt=\".2f\")\n",
        "\n",
        "plt.title(\"Correlation Heatmap of Numerical Features in Laptop Dataset\", fontsize=14)\n",
        "\n",
        "plt.show()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-03-04T16:40:11.529595Z",
          "iopub.execute_input": "2025-03-04T16:40:11.529972Z",
          "iopub.status.idle": "2025-03-04T16:40:11.763718Z",
          "shell.execute_reply.started": "2025-03-04T16:40:11.529946Z",
          "shell.execute_reply": "2025-03-04T16:40:11.76192Z"
        },
        "id": "VAevFrcJM4eo",
        "outputId": "0142f0b9-9d13-4680-ef01-c8652a35e727"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": "<Figure size 1200x600 with 2 Axes>",
            "image/png": "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1OA2sCqMDNQQEJyMmUsfyRcfMzORITtv2Qvcf5yIxT8FYNTgqhgzxBqhYamY+8NjBD/PG2XTopEhZn2Zt6DlvG8cAABee19g+14+iqei87kSA309Kdz6W8HQQA0BIclwX/YEsQVcUw+eJGHp2iCMHFgFowZVQWh4GjxWBiD4Rao8T/OGBpgxoZr833O+yrm+dux7iR37X5ZNw6hc7forFJoyFUwfXx062lLcexiPbxc9QHpG3sVkZaEBfb28HwzPXHwNAz01jBpiA6P/huF/u+iBwmKMa7cFIVsQsPDbmlBTU8F131is3hRYpm2jstW5aT3EJCRiw8GTeB2XgJo2llg7dSSM9XNGEoW/joXKG09hsjA2wNppI7FylzcGzf0JZoZ6GNKpBdy6tZXn2XvmKgBg7PebFY41f3R/+Q8ERB87ifDmSmcket7e3vj222/h5+enEGkvSTNnzkRMTAw2bdpUvLqp1SyV+lDltaLvjvKuAlUwRZ3KRFQUGWmct00l6+j08PKuAlUw2i0+Le8qvLfHQ7qUWtmOu46/O1M5Y0S+gunevTuePHmC0NBQWFtbv3uH92BmZpZvITsiIiIiIiIqG+zIV0Bff/11qZY/bdq0Ui2fiIiIiIioMGJ53ntpYUeeiIiIiIiIREVFVRyL0pWWyv0zBhEREREREZHIMCJPREREREREoiKWx8SVFkbkiYiIiIiIiESEEXkiIiIiIiISlcq+2F3lbj0RERERERGRyDAiT0RERERERKLCOfJEREREREREJBqMyBMREREREZGoVPaIPDvyREREREREJCpc7I6IiIiIiIiIRIMReSIiIiIiIhKVyj60nhF5IiIiIiIiIhFhRJ6IiIiIiIhEhXPkiYiIiIiIiEg0GJEnIiIiIiIicZFwjjwRERERERERiQQj8kRERERERCQqlX3VenbkiYiIiIiISFS42B0RERERERERiQYj8kRERERERCQqlX1oPSPyRERERERERCLCiDwRERERERGJCufIExEREREREZFoMCJPREREREREosI58kREREREREQkGozIExERERERkahU9og8O/JEREREREQkLlzsjoiIiIiIiIjEghF5IiIiIiIiEhWJpHIPrWdEnoiIiIiIiEhEGJEnIiIiIiIiUZFwjjwRERERERERiQU78kRERERERCQqEhVJqb2Ka926dbCzs4OGhgaaNm2Ka9euFZp/zZo1qFmzJjQ1NWFtbY1vvvkGqampxTomO/JERERERERE7+HPP//E1KlT4eHhgVu3bsHZ2RmdO3fGq1evlOb/448/MGvWLHh4eODhw4fYsmUL/vzzT3z33XfFOi478kRERERERCQuKiql9yqGVatWYezYsRg5ciScnJzwyy+/QEtLC1u3blWa/9KlS2jZsiWGDh0KOzs7fPLJJxgyZMg7o/j5ml+s3EREREREREQVWFpaGuLj4xVeaWlp+fKlp6fj5s2b6NixozxNRUUFHTt2xOXLl5WW3aJFC9y8eVPecQ8MDMTRo0fRrVu3YtWRHXkiIiIiIiISldKcI+/p6Ql9fX2Fl6enZ746REVFISsrC+bm5grp5ubmCA8PV1rvoUOHYuHChWjVqhXU1NRQvXp1tGvXjkPriYiIiIiIqGKTSFRK7eXu7o64uDiFl7u7e4nU28fHB0uXLsX69etx69YtHDhwAN7e3li0aFGxyuFz5KnMrOi7o7yrQBXM9L+Gl3cVqIJZ1W9neVeBKpCsjIzyrgJVMG57G5R3FaiC2duivGvwcZLJZJDJZO/MZ2JiAlVVVURERCikR0REwMLCQuk+c+fOxbBhwzBmzBgAQN26dZGUlIQvvvgCs2fPhkoR5+gzIk9ERERERETioiIpvVcRqauro2HDhjh16pQ8LTs7G6dOnULz5s2V7pOcnJyvs66qqgoAEAShyMdmRJ6IiIiIiIjoPUydOhUjRoxAo0aN0KRJE6xZswZJSUkYOXIkAGD48OGoUqWKfI59z549sWrVKjRo0ABNmzbF06dPMXfuXPTs2VPeoS8KduSJiIiIiIhIVCTFfExcaRk0aBAiIyMxb948hIeHo379+jh+/Lh8Abxnz54pRODnzJkDiUSCOXPmIDQ0FKampujZsyeWLFlSrOOyI09ERERERET0niZNmoRJkyYp3ebj46Pwb6lUCg8PD3h4eHzQMdmRJyIiIiIiIlGRFGMue0X0cYxHICIiIiIiIqIiYUSeiIiIiIiIxEVSuWPS7MgTERERERGRqHBoPRERERERERGJBiPyREREREREJC4fyePnykvlbj0RERERERGRyDAiT0RERERERKIikXCOPBERERERERGJBCPyREREREREJC6cI09EREREREREYsGIPBEREREREYlKZX+OPDvyREREREREJC6Syj24vHK3noiIiIiIiEhkGJEnIiIiIiIicankQ+sZkSciIiIiIiISEUbkiYiIiIiISFQknCNPRERERERERGLBiDwRERERERGJC+fIExEREREREZFYMCJPREREREREoiJRqdwxaXbkiYiIiIiISFwkHFpPRERERERERCLBiDwRERERERGJSyUfWl+5W09EREREREQkMozIExERERERkbhwjjwRERERERERiQUj8kRERERERCQqlf3xc5W79UREREREREQiw4g8ERERERERiYukcsek2ZEnIiIiIiIicVHhYndEREREREREJBKMyBMREREREZGoSCr50PrK3XoiIiIiIiIikWFHvgRJJBIcPHiwVI/h7+8PCwsLJCQkAAC8vLxgYGBQYuUHBwdDIpHA19e3wDzHjx9H/fr1kZ2dXWLHJSIiIiIiKjIVSem9RKDUhtZHRkZi3rx58Pb2RkREBAwNDeHs7Ix58+ahZcuWpXXYUlOU9oSFhcHQ0LBU6+Hu7o7JkydDV1cXADBo0CB069atVI/5ti5dumDu3Ln4/fffMWzYsDI9thj16WyOQT0tYWSghoCQZPy0NRiPApIKzN+2mRFGDaoKC1MZXoSnYtPvz3D1dpx8e+smhujZyRyO9lrQ11XDmG/vISAkuSyaQuXMqFUj2E8bDX2XOtCwMsONfhMRcehU4fu0aQKnFbOg4+SA1OdheOq5AS92/KWQx3bCUNhPHQ2ZhSni7z7C/a8XIe76vdJsCn1ken9ihoE9LWGkr4aAZ8n4eVsI/Au5T7VpaoiRA/PuU5v/eI5rvnn3qVaNDdGzkxkcq2lDT1eKL2b68T5VCY3+zA49P7GArrYU9x7GY8X6J3gRllLoPp92s8KQT61hZKiOgKBErN74FA+f5AQvdHWkGD3UDk0aGMLcVIbY+AycuxKFX38LRlJyVlk0icpJ55a66NVeHwa6qgh5mY6tB17j6bP0AvM3c9bC4K6GMDWSIjwyE78dicbth3nX3t7V1ZTut/NQNA6diVO6jehjU2oR+X79+uH27dvYvn07Hj9+jEOHDqFdu3Z4/fr1e5WXlZVVrhHgorTHwsICMpms1Orw7NkzHDlyBG5ubvI0TU1NmJmZldoxC+Lm5oaffvqpzI8rNq7NjTBhuA2273sh/yK7fHYtGOgp/w2ttqMO5n5VA0dPR2LszHu4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Q4WPnyMiIiIiIiIikWBEnoiIiIiIiMSlkg+tZ0SeiIiIiIiISEQYkSciIiIiIiJxkVTumHTlbj0RERERERGRyDAiT0REREREROLCVeuJiIiIiIiISCwYkSciIiIiIiJxqeSr1rMjT0REREREROLCxe6IiIiIiIiISCwYkSciIiIiIiJxqeRD6xmRJyIiIiIiIhIRRuSJiIiIiIhIXPj4OSIiIiIiIiISC0bkiYiIiIiISFQEzpEnIiIiIiIiIrFgRJ6IiIiIiIjEhc+RJyIiIiIiIiKxYESeiIiIiIiIxKWSR+TZkSciIiIiIiJR4WJ3RERERERERCQajMgTERERERGRuFTyofWVu/VEREREREREIsOIPBEREREREYkL58gTERERERERkVgwIk9ERERERETiolK5Y9KVu/VEREREREREIsOIPBEREREREYlKZX+OPDvyREREREREJC58/BwRERERERERiQUj8kRERERERCQqAiPyRERERERERPQ+1q1bBzs7O2hoaKBp06a4du1aofljY2Px5ZdfwtLSEjKZDI6Ojjh69GixjsmIPBEREREREYnLR7LY3Z9//ompU6fil19+QdOmTbFmzRp07twZ/v7+MDMzy5c/PT0dnTp1gpmZGfbt24cqVaogJCQEBgYGxTouO/JERERERERE72HVqlUYO3YsRo4cCQD45Zdf4O3tja1bt2LWrFn58m/duhXR0dG4dOkS1NTUAAB2dnbFPi6H1hMREREREZGoCBKVUnulpaUhPj5e4ZWWlpavDunp6bh58yY6duwoT1NRUUHHjh1x+fJlpfU+dOgQmjdvji+//BLm5uaoU6cOli5diqysrGK1nx15IiIiIiIiov94enpCX19f4eXp6ZkvX1RUFLKysmBubq6Qbm5ujvDwcKVlBwYGYt++fcjKysLRo0cxd+5crFy5EosXLy5WHTm0noiIiIiIiMSlFOfIu7u7Y+rUqQppMpmsRMrOzs6GmZkZNm3aBFVVVTRs2BChoaH44Ycf4OHhUeRy2JEnIiIiIiIicSnFx8/JZLIiddxNTEygqqqKiIgIhfSIiAhYWFgo3cfS0hJqampQVVWVp/3vf/9DeHg40tPToa6uXqQ6siNPZebNi5WoJKzqt7O8q0AVzNT9w8q7ClSBLO+5rbyrQBVMdFhkeVeBKpxq5V0BUVNXV0fDhg1x6tQp9OnTB0BOxP3UqVOYNGmS0n1atmyJP/74A9nZ2VBRyfkx4vHjx7C0tCxyJx7gHHkiIiIiIiISGUEiKbVXcUydOhWbN2/G9u3b8fDhQ0yYMAFJSUnyVeyHDx8Od3d3ef4JEyYgOjoaX331FR4/fgxvb28sXboUX375ZbGOy4g8ERERERER0XsYNGgQIiMjMW/ePISHh6N+/fo4fvy4fAG8Z8+eySPvAGBtbY0TJ07gm2++Qb169VClShV89dVXmDlzZrGOy448ERERERERiUspzpEvrkmTJhU4lN7HxydfWvPmzXHlypUPOubH03oiIiIiIiIieidG5ImIiIiIiEhUBJTe4+fEgBF5IiIiIiIiIhFhRJ6IiIiIiIhERfiI5siXB3bkiYiIiIiISFwqeUe+creeiIiIiIiISGQYkSciIiIiIiJRESRc7I6IiIiIiIiIRIIReSIiIiIiIhKVyr7YXeVuPREREREREZHIMCJPRERERERE4sI58kREREREREQkFozIExERERERkahU9jny7MgTERERERGRqAjg0HoiIiIiIiIiEglG5ImIiIiIiEhUKvvQ+srdeiIiIiIiIiKRYUSeiIiIiIiIxIWPnyMiIiIiIiIisWBEnoiIiIiIiERFqOQx6crdeiIiIiIiIiKRYUSeiIiIiIiIREWo5HPk2ZEnIiIiIiIiUeHj54iIiIiIiIhINBiRJyIiIiIiIlERULmH1jMiT0RERERERCQijMgTERERERGRqHCOPBERERERERGJBiPyREREREREJCqV/fFzjMgTERERERERiQgj8kRERERERCQqlX3VenbkiYiIiIiISFS42B0RERERERERiQYj8kRERERERCQqlX1oPSPyRERERERERCLCjnwx2NnZYc2aNfJ/h4eHo1OnTtDW1oaBgcEHlb1lyxZ88skn8n+7ubmhT58+H1RmaRk8eDBWrlxZ3tUgIiIiIqJKSpColNpLDIo1tN7NzQ3bt2+X/9vIyAiNGzfG8uXLUa9evSKXM3/+fBw8eBC+vr4K6RKJBH/99Ve5dWALqleu69evQ1tbW/7v1atXIywsDL6+vtDX13/v46ampmLu3LnYu3fve5dRlubMmYM2bdpgzJgxH9TuyqJXJ1MM7GkBI301BDxLxlqv5/APSCowf5umhnAbYAULUxlCw1OxeVcorvnGybe3amyAHh1N4VhNG3q6UoybdR8BISll0RT6SPT+xAwDe1rKr6mft4W885oaObAqLExleBGeis1/PH/rmjJEz05m8mvqi5l+CAhJLoumUDkzatUI9tNGQ9+lDjSszHCj30REHDpV+D5tmsBpxSzoODkg9XkYnnpuwIsdfynksZ0wFPZTR0NmYYr4u49w/+tFiLt+rzSbQh+hUYOt0aOTOXS0VHHvUQJWbQpEaFhqofv06WKBwX2sYGSgjoDgJPz4axAePU2Ub1dXk2Cimx3atzKBmlQF131jsXpTIGLiMkq7OVSO+LlHlF+xf27o0qULwsLCEBYWhlOnTkEqlaJHjx6lUbf3lpFROjdzU1NTaGlpyf8dEBCAhg0bwsHBAWZmZu9d7r59+6Cnp4eWLVuWRDU/SHp6+jvz1KlTB9WrV8dvv/1WBjUSt3bNDDF+mDV27n+J8d89QGBICpbNcoCBnvLf0JwctDF7sj2O+0RhvPsDXLwRiwXTqsOuqoY8j4ZMBX7+idi860VZNYM+Iu2aG2H8MBvs2BeK8e45Xzy+d69Z8DXlqIM5U2rg2JlIjJvlh4s3YrBwugPsqmrK82hoqMDvUQI2//G8rJpBHwlVbS3E3/WH35QFRcqvaVcVjQ9txGufq7jQqDeCft6OuhsXw6RTK3keywFd8b8f3PFk8TpcaNIXCXcfoan3FqibGpVWM+gjNKRvFXza3RIrfwnA+Fn3kJqWjRVznaCuVvCcVteWxvhypB2273mBsdPvICA4CSvmOcFAX02eZ9LIamjRyAgeP/jjq7l+MDFSx6KZNcuiSVRO+LlHBREgKbWXGBS7Iy+TyWBhYQELCwvUr18fs2bNwvPnzxEZGSnPM3PmTDg6OkJLSwv29vaYO3euvHPt5eWFBQsW4M6dO5BIJJBIJPDy8oKdnR0AoG/fvpBIJPJ/A8Dff/8NFxcXaGhowN7eHgsWLEBmZqZ8u0QiwYYNG9CrVy9oa2tj8eLFqFGjBlasWKFQd19fX0gkEjx9+rS4zQagOLTezs4O+/fvx44dOyCRSODm5gYAiI2NxZgxY2Bqago9PT20b98ed+7cKbTc3bt3o2fPnkq3rVixApaWljA2NsaXX36p8CNFTEwMhg8fDkNDQ2hpaaFr16548uSJfPv8+fNRv359hfLWrFmjcG5zh/AvWbIEVlZWqFkz58Nw/fr1cHBwgIaGBszNzdG/f3+Fcnr27Indu3cX2i4C+nU3x9HTUThx9jWehaZizZYQpKVno0s7E6X5P+1qjut34rDnSASevUyF196XeBqUjN6d834oOnkhGr8dCMOte/Fl1Qz6iPTvboGjpyNx4mwUQkJTsebX4P+uKVOl+fOuqfCca2pPKJ4EJaNPZ3N5npPnX2PngZe46RentAyquCJPnMNjjzWI+PtkkfLbfjEYKUEv8HDG90h8FIiQ9b8jfP8JVPvKTZ6n2tcj8XzLHrzYfgCJDwNwb6IHspJTYe3Wr5RaQR+jAT0ssXPfC1y8HoPAkGQs/ekJjI3U0apJwT/oDOxphSP/RuDY6VcIeZGClRsDkZqWhW7tcz4DtbVU0a2DGdZ5BeO2XzweByZh2dqnqFtLD06OOmXVNCpj/NwjUu6DJgAkJibit99+Q40aNWBsbCxP19XVhZeXFx48eIAff/wRmzdvxurVqwEAgwYNwrRp01C7dm15ZH/QoEG4fv06AGDbtm0ICwuT//v8+fMYPnw4vvrqKzx48AAbN26El5cXlixZolCX+fPno2/fvrh37x5Gjx6NUaNGYdu2bQp5tm3bhjZt2qBGjRof0mwAOcPsu3TpgoEDByIsLAw//vgjAGDAgAF49eoVjh07hps3b8LFxQUdOnRAdHR0gWVduHABjRo1ypd+5swZBAQE4MyZM9i+fTu8vLzg5eUl3+7m5oYbN27g0KFDuHz5MgRBQLdu3Yo9IuHUqVPw9/fHv//+iyNHjuDGjRuYMmUKFi5cCH9/fxw/fhxt2rRR2KdJkya4du0a0tLSinWsykSqKoFjNW3c8svrcAsCcMsvHk4O2kr3cXJQzA8A1+/Gw8mBX1DojWvqXt4XD0EAbt2LL/BLrJODDm7eU/yicuNOHL/00nsxaFYfUacvK6RF/nsBhs3qAwAkamrQd6mNqFOX8jIIAqJOX4JBswZlWFMqT5bmMhgbquPmnVh5WlJyFh4+SUDtmrpK95FKJXCsroObdxXvbzfvxsn3cbTXhpqaikK5z0JTEB6ZhtqOysslcePnHhWGc+SL6ciRI9DRyXkjJCUlwdLSEkeOHIGKSl6D58yZI/9/Ozs7TJ8+Hbt378aMGTOgqakJHR0dSKVSWFhYyPNpauYMdzEwMFBIX7BgAWbNmoURI0YAAOzt7bFo0SLMmDEDHh4e8nxDhw7FyJEj5f92c3PDvHnzcO3aNTRp0gQZGRn4448/8kXp35epqSlkMhk0NTXl9b1w4QKuXbuGV69eQSaTAciJqB88eBD79u3DF198ka+c2NhYxMXFwcrKKt82Q0NDrF27FqqqqqhVqxa6d++OU6dOYezYsXjy5AkOHTqEixcvokWLFgCA33//HdbW1jh48CAGDBhQ5LZoa2vj119/hbq6OgDgwIED0NbWRo8ePaCrqwtbW1s0aKD4BczKygrp6ekIDw+Hra1tkY9VmejrSaGqKsk3by8mLhPWVhpK9zE0UENMXKZCWmxcBowM1JTmp8ol75pSvEZi4jJgXUX5NWVkoKbkGsyAkT6vKSo+mbkJ0iKiFNLSIqKgpq8LFQ0Z1Az1oSKVIu3V67fyvIZ2TfuyrCqVIyODnO8T0W/fe2IzYGSornQffV0ppKoSxMSm59vHpkrOd0RjQ3WkZ2QjMTnrrTzpBZZL4sbPPSqMWIbAl5Zid+RdXV2xYcMGADlDu9evX4+uXbvi2rVr8g7dn3/+iZ9++gkBAQFITExEZmYm9PT03quCd+7cwcWLFxUi8FlZWUhNTUVycrJ8zvrbEW0rKyt0794dW7duRZMmTXD48GGkpaUVq4P7PnVNTExUGJ0AACkpKQgICFC6T0pKziJlGhr5b0a1a9eGqqqq/N+Wlpa4dy9nsaCHDx9CKpWiadOm8u3GxsaoWbMmHj58WKx6161bV96JB4BOnTrB1tYW9vb26NKlC7p06YK+ffsqrA+Q+8NLcrLyhUHS0tLyReuzs9KhosoPWiIiooqkYxsTTBtXXf7vWUuK9z2EiIiKr9gdeW1tbYWh6b/++iv09fWxefNmLF68GJcvX8Znn32GBQsWoHPnztDX18fu3bvf+3FliYmJWLBgAT799NN8297s/L65mnyuMWPGYNiwYVi9ejW2bduGQYMGKXRGS1piYiIsLS3h4+OTb1tBj6czNjaGRCJBTExMvm1qaoq/HEokEmRnZxe5PioqKhAEQSFN2bD7t8+drq4ubt26BR8fH/zzzz+YN28e5s+fj+vXr8vbkTtVwNRU+fwkT09PLFiguHhStdpjYV83/6iEiiouPhNZWQIM3/oF2FBfiphY5dMfYmIzYKiv+LY00FdDdAH5qXLJu6YUrxHDQq6R6NgMJdegWr5IGVFRpEVEQWauuMaHzNwEGXEJyE5NQ3pUDLIzMyEzM34rjzHSwhUj+VRxXLwWjYeP81aWV/tvQTsjfTVEx+TdawwN1PA0SPlK43EJmcjMEmBooPiDv6FB3v3tdUw61NVUoKOlqhCVNzRQR3TMuxfrJfHh5x4VRpBU7oj8B08AkEgkUFFRkUeWL126BFtbW8yePRuNGjWCg4MDQkJCFPZRV1dHVlZWvrLU1NTypbu4uMDf3x81atTI93pzOL8y3bp1g7a2NjZs2IDjx49j1KhRH9jawrm4uCA8PBxSqTRfXU1MlC9upq6uDicnJzx48KBYx/rf//6HzMxMXL16VZ72+vVr+Pv7w8nJCUBOJzs8PFyhM1/Qo/XeJpVK0bFjRyxfvhx3795FcHAwTp8+Ld/u5+eHqlWrFtgud3d3xMXFKbzsnNyK1Uaxy8wS8DgoCS518ubtSSRAg9p6ePBE+ReZB0+S0KC24uiVhnX18OBJotL8VLnkXlMN6uQ99lEiARrU0cODx8qvkQdPEuFS561rql7B+YkKE3vFF8btmymkmXRogZgrvgAAISMDcbfuw6R987wMEgmMXZsj9srtMqwplaWU1GyEhqfKX8HPU/A6Jh0u9QzkebQ0VfE/B13c909QWkZmpoDHAYloWE/x/uZST1++z+PAJGRkZMPljTzWVhqwMJXh/mPl5ZK48XOPqGDFjsinpaUhPDwcQM7Q+rVr1yIxMVG+6rqDgwOePXuG3bt3o3HjxvD29sZffyk+X9bOzg5BQUHw9fVF1apVoaurC5lMBjs7O5w6dQotW7aETCaDoaEh5s2bhx49esDGxgb9+/eHiooK7ty5Az8/PyxevLjQuqqqqsLNzQ3u7u5wcHBA8+bNC80P5Ax1f7uzq6uri+rVqyvf4Q0dO3ZE8+bN0adPHyxfvhyOjo54+fIlvL290bdvX6UL2gFA586dceHCBXz99dfvPEYuBwcH9O7dG2PHjsXGjRuhq6uLWbNmoUqVKujduzcAoF27doiMjMTy5cvRv39/HD9+HMeOHXvnNIcjR44gMDAQbdq0gaGhIY4ePYrs7Gz5ivZAziKEn3zySYFlyGQy+ToBuSrjsPr93hGYMaEa/AOT4f80CZ92NYeGTAXHz+ZEpmZOsENUTAa27A4FABw4FoFV82qif3dzXL0dB9fmRnC018LqzcHyMnW1VWFmog7j/+YDWlvmjEyJjs3IN4eMKp593uGYOcEejwOT8OhpIvp1s4CGTAUnzuY8OWTmRHtERadjy+6cxxMeOBaB1fNqYUB3C1y5HQvXFsZwtNfGqk3B8jJzrikZjA1zIhi5azjkXFOMYFRkqtpa0K5hI/+3VrWq0HOuhfToOKQ+D0PNxVOhUcUcd0bOBACEbNoN24mfoZbnt3jutR8mrs1gOaArrvcaJy8jaM02OG/9HrE3/RB3/S7spoyAVFsTz7cfKPP2UfnZeyQMw/tXxYuwFIRHpGHUEGu8jk7HhWt5i/+umu+E81ej8dexnO+Vew6/hPtkBzx6mohHTxLRv6clNGWqOHb6FYCcBfOOnnqFL0dWQ0JiJpKSs/DVmGrwexTPTloFxs89KoggVO6IfLE78sePH4elpSWAnA5urVq1sHfvXrRr1w4A0KtXL3zzzTeYNGkS0tLS0L17d8ydOxfz58+Xl9GvXz8cOHAArq6uiI2NxbZt2+Dm5oaVK1di6tSp2Lx5M6pUqYLg4GB07twZR44cwcKFC/H9999DTU0NtWrVwpgxY4pU39GjR2Pp0qUKC+EV5vHjx/kWduvQoQNOnnz3o3kkEgmOHj2K2bNnY+TIkYiMjISFhQXatGkDc3PzAvcbPXo0GjVqhLi4OOjr6xeY723btm3DV199hR49eiA9PR1t2rTB0aNH5UPy//e//2H9+vVYunQpFi1ahH79+mH69OnYtGlToeUaGBjgwIEDmD9/PlJTU+Hg4IBdu3ahdu3aAIDU1FQcPHgQx48fL3JdKyufKzHQ15PCrb8VDA3UEBCSDPdlTxD7X4fbzESG7DdmPzx4koSla4MwcmAVjBpUBaHhafBYGYDgF6nyPM0bGmDGhGryf8/5KudHph37XmLH/pdl0zAqNz6Xo3OuqQFV5NfUrGX+8h9xzEzUFUbhPHiciCU/B2DUoKoYNbgqQsNTMW/FEwS/SJHnadHIEDMm5C1ENvernOlT2/eFYse+0DJqGZUH/YZ10PzUTvm/nVZ8BwB4vuMA7o52h8zSFJrWlvLtKcEvcL3XODitdIfd5OFIfRGOe+PmIOrfC/I8YXuPQd3UCI4eUyCzMEX8nYe41mMM0t9aAI8qtl1/hUJTpoLp46tDR1uKew/j8e2iB0jPyLs/WVloQF8vbwj0mYuvYaCnhlFDbGD03zD8bxc9UOhYrd0WhGxBwMJva0JNTQXXfWOxelNgmbaNyhY/94iUkwhvT6KuYM6fP48OHTrg+fPnhXamy9uAAQPg4uICd3f38q7KO23YsAF//fUX/vnnn2Lt13HIjVKqEVVWglD0NSOIimLq/mHlXQWqQJb33PbuTETFIFUvdgyOqFCndjcp7yq8tycBIe/O9J4cqn/8T+USx0Py3kNaWhpevHiB+fPnY8CAAR91Jx4AfvjhB/lj/T52ampq+Pnnn8u7GkRERERERJVShe3I79q1C7a2toiNjcXy5cvLuzrvZGdnh8mTJ5d3NYpkzJgxCvPliYiIiIiIypIASam9xKDCduTd3NyQlZWFmzdvokqVKuVdHSIiIiIiIioh7MgTERERERERkWhwxQwiIiIiIiISFbFEzksLI/JEREREREREIsKIPBEREREREYkKI/JEREREREREJBqMyBMREREREZGoCAIj8kREREREREQkEozIExERERERkahwjjwRERERERERiQYj8kRERERERCQqlT0iz448ERERERERiUpl78hzaD0RERERERGRiDAiT0RERERERKLCx88RERERERERkWgwIk9ERERERESiks058kREREREREQkFozIExERERERkahw1XoiIiIiIiIiEg1G5ImIiIiIiEhUKvuq9ezIExERERERkahwaD0RERERERERiQYj8kRERERERCQqlX1oPSPyRERERERERCLCiDwRERERERGJCufIExEREREREZFoMCJPREREREREosI58kREREREREQkGozIExERERERkahkl3cFyhk78kRERERERCQqHFpPRERERERERKLBjjwRERERERGJigBJqb2Ka926dbCzs4OGhgaaNm2Ka9euFWm/3bt3QyKRoE+fPsU+JjvyRERERERERO/hzz//xNSpU+Hh4YFbt27B2dkZnTt3xqtXrwrdLzg4GNOnT0fr1q3f67jsyBMREREREZGoCIKk1F7FsWrVKowdOxYjR46Ek5MTfvnlF2hpaWHr1q0F7pOVlYXPPvsMCxYsgL29/Xu1nx15IiIiIiIiov+kpaUhPj5e4ZWWlpYvX3p6Om7evImOHTvK01RUVNCxY0dcvny5wPIXLlwIMzMzjB49+r3ryI48ERERERERiUppzpH39PSEvr6+wsvT0zNfHaKiopCVlQVzc3OFdHNzc4SHhyut94ULF7BlyxZs3rz5g9rPx88RERERERER/cfd3R1Tp05VSJPJZB9cbkJCAoYNG4bNmzfDxMTkg8piR56IiIiIiIhEJVsovbJlMlmROu4mJiZQVVVFRESEQnpERAQsLCzy5Q8ICEBwcDB69uwpT8vOzgYASKVS+Pv7o3r16kWqI4fWExERERERkah8DI+fU1dXR8OGDXHq1Cl5WnZ2Nk6dOoXmzZvny1+rVi3cu3cPvr6+8levXr3g6uoKX19fWFtbF/nYjMhTmclISy/vKlAFk5WRUd5VoApmec9t5V0FqkBmHB5Z3lWgCuaH3l7lXQUiesvUqVMxYsQINGrUCE2aNMGaNWuQlJSEkSNzPgOGDx+OKlWqwNPTExoaGqhTp47C/gYGBgCQL/1d2JEnIiIiIiIiUSnuY+JKy6BBgxAZGYl58+YhPDwc9evXx/Hjx+UL4D179gwqKiU/EJ4deSIiIiIiIqL3NGnSJEyaNEnpNh8fn0L39fLyeq9jsiNPREREREREoiKU4mJ3YsDF7oiIiIiIiIhEhBF5IiIiIiIiEpXsYqwuXxExIk9EREREREQkIozIExERERERkah8LKvWlxd25ImIiIiIiEhUuNgdEREREREREYkGI/JEREREREQkKgIXuyMiIiIiIiIisWBEnoiIiIiIiEQlm3PkiYiIiIiIiEgsGJEnIiIiIiIiUansj59jRJ6IiIiIiIhIRBiRJyIiIiIiIlGp7M+RZ0eeiIiIiIiIRCWbj58jIiIiIiIiIrFgRJ6IiIiIiIhEpbIPrWdEnoiIiIiIiEhEGJEnIiIiIiIiUeHj54iIiIiIiIhINBiRJyIiIiIiIlHJ5hx5IiIiIiIiIhILRuSJiIiIiIhIVCr7qvXsyBMREREREZGoCOBid0REREREREQkEozIExERERERkahwsTsiIiIiIiIiEg1G5ImIiIiIiEhUKvtid4zIExEREREREYkII/JEREREREQkKozIExEREREREZFoMCJPREREREREopItVO7nyLMjT0RERERERKLCofVEREREREREJBqMyBMREREREZGoMCJPRERERERERKLBiDwRERERERGJSjYj8lRafHx8IJFIEBsbW+R95s+fj/r165dYHfz9/WFhYYGEhAQAgJeXFwwMDD6ozOPHj6N+/frIzs4ugRoSERERERFRcTAiD+CXX37Bt99+i5iYGEilOackMTERhoaGaNmy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wfbt22Fvby89S29oaAhDQ8MyH5cNeSIiIiIiIqKX4O3tjaSkJMyZMwcJCQlo0qQJgoKCpAnw4uLioK7+bCD8unXrkJOTg379+snSmTt3LubNm1fm46qJipy3n+g5Gaf3VHYW6B3js8u9srNA75hH8UmVnQV6h+Tn5Vd2FugdM22vT2Vngd4x3XMjKjsLL+3LTdkvjvSSFn76+ie2e934jDwRERERERGRCuHQeiIiIiIiIlIpVX1cOXvkiYiIiIiIiFQIe+SJiIiIiIhIpSjelmnrKwl75ImIiIiIiIhUCHvkiYiIiIiISKVU9cXX2JAnIiIiIiIilSIUlZ2DysWh9UREREREREQqhD3yREREREREpFIUVXxoPXvkiYiIiIiIiFQIe+SJiIiIiIhIpVT1ye7YI09ERERERESkQtgjT0RERERERCpFoWCPPBERERERERGpCPbIExERERERkUqp4o/IsyFPREREREREqkVwaD0RERERERERqQr2yBMREREREZFKUVTxsfXskSciIiIiIiJSIeyRJyIiIiIiIpXCZ+SJiIiIiIiISGWwR56IiIiIiIhUCnvkiYiIiIiIiEhlsEeeiIiIiIiIVEoV75BnjzwRERERERGRKmGPPBEREREREamUqv6MPBvyREREREREpFKEqNoNeQ6tJyIiIiIiIlIh7JEnIiIiIiIilaKo4kPr2SNPREREREREpELYI09EREREREQqhc/IExEREREREZHKYI88ERERERERqZSqvvwce+SJiIiIiIiIVAh75ImIiIiIiEilVPUeeTbkiYiIiIiISKUoONkdEREREREREakK9sgTERERERGRSqnqQ+vZI/+Wsre3x6pVqyr8OB06dMD27duLhYeEhCAwMLBYeHJyMmrWrIn79+9XeN6IiIiIiIioOPbIVzAfHx9s3rwZAKClpQVbW1sMGzYMM2fOhKZmyR//hQsXYGBgUKF527dvHx48eIABAwaUeR8zMzMMGzYMc+fOxcaNGyswd++OncfPYMuhE3j4OB3OtpaYPrgX3BxsSoyflvkU3+0+guBLf+NxRiasTE0wdWAPtGtcDwCwaX8I/rx0HTEJSdDR0kJjRztM+MQL9lbmb+qUqJJ5vlcNvboYw6SaBmL/zcGmPQ9xJy6nxPitG+tjQLfqMK+hiYSkPPy0/xGu3Hwqbd+1so7S/bbue4R9wY9fe/7p7dP7g5ro39MKNYy1EBWXif8GxCIiKqPE+B1aVYdv/9qwNNfB/YQs/LD9Hs6HPSsr7VpUR0+PmnCuYwCjapr4zO86omIz38Sp0Fugj5cFBvSyRg0TbdyJzcDqjTG4dSe9xPgd29TAiAG2BeUpPgvrf4rFuSupsji+3jbo0bUmDPU1cT3iCVZsiMY/CVkVfCb0NqjRrjkcpoyAcVM36FrXxMW+n+PBvuOl79OhJVyXfQFDVydk3YvHHf91uL/lN1kcu7GD4DB5BHQszfEk/Bb+nvg1Hl+4VpGnQq+Z4DPyVNG8vLwQHx+P27dvY8qUKZg3bx6WLl2qNG5OTkFl3NzcHPr6+hWar9WrV8PX1xfq6s+KQVhYGDw8PNC3b1+MHz8eDRs2xLx582T7+fr6Ytu2bXj06FGF5u9dcPhcOFbsOIDPer+P7fPGwcnGCv+3fBMePVFeocnNy8PYpRsRn5yCb/5vEH7zn4LZPh+jZnVjKc6liLvo/34bbJ71OdZNHYG8/Hx8vnwTnmaX3JCjd0fbJgYY3scUuw6nwm/5v4j9NwdfjraEkaHyy7mzvQ4mDq2JP8+lY/qyf3H+egamf2oBG0stKc6oOXGy15qfk6BQCJwNL7khR++OTm1qYMxQW2z59R+MmVHQ4F4ywwUmRspvNrs6G2LWBEccCk7C6C+u49TFFHw11Qn2tfWkOLq66rh+Kw0/bL/3pk6D3hKd25ri8+H2CNx1H6OmhyMqJhNLZ9UvsTw1cDHEnInOOHA8ESOnhSP0wiMsmO6COjbPytPAPtbo+6ElVmy4i7Ezr+FptgJLZ9eHtpbamzotqkQaBvp4Eh6B6xPmlym+nn1ttNi3Hg9DziG0eW9E/3czGq5fADOPdlIcq0+6of7SGbi9YA1CW36EtPBbaHVgI7TNa1TUaRC9dmzIvwE6OjqwtLSEnZ0dxo4di65du2Lfvn0ACnrs+/Tpg4ULF8La2houLi4Aig+tT01NxejRo2FhYQFdXV24ublh//790vbQ0FC0b98eenp6sLGxwYQJE5CRUXIlPCkpCX/++Sd69uwphQkh0Lt3b+jp6cHf3x/Tp0/HokWLoKenJ9u3QYMGsLa2xm+//VY0WSpi25GT+KhDC/Ru3xwOtSzw5bA+0NXWxt6TF5XG33vyEp5kPMXy8UPRxMke1mbV0ayeA5xtraQ4a6Z8il7tmqFuLQs421ph/oh+SHiYihsx/7yp06JK1KOTEY6fSUPI+XTcf5CLDbseIidHoEurakrjd+9ghLBbT7Ev+DH+SczFzkOpuHs/G17tjaQ4qWn5slcLN338fScLiQ/z3tRpUSXq190SB/9MwuG/khH7TxZW/RiD7BwFvDopH+XzcTcLXLj6GL/sT0Dcv1kI/OUf3I7ORB9PCynOsZMPsXXPv7h0nSM6qppPelrhwLFEBAUnIfb+U6zYcBdZ2Qp82KWm0vh9P7TC+bBU7Nz3L+L+eYpNO+7hdnQGPupmKcXp190KW3ffx6kLKbgbmwn//96BWXVttGvJRldVkHT4BCLnrsKDvcfKFN/uswF4Gn0fN6cvQfqtu4hduw0Juw+jzn98pDh1Jvri3sZfcH/zHqTfjMK1z+ciPzMLNj59K+gsqCIoFKLCXqqADflKoKenJ/W8A8Dx48cRERGBo0ePyhrnhRQKBbp164ZTp07hp59+wo0bN7B48WJoaGgAAKKiouDl5YW+ffsiPDwcO3fuRGhoKMaNG1diHkJDQ6Gvr4/69etLYQ8fPkRcXBz8/Pzg7OwMGxsb9OzZE35+fsX2b9myJU6ePPkqH8M7LzcvDzdj/kWrBo5SmLq6Olq51kX4nTil+/x15QYa1rXF4p/2out/FuKTWauwcX8w8hWKEo+T9rRgaKGxgV6JcejdoKkBONTWQXjks2HxQgDht5/C2U5H6T7O9rqy+ABwNaLk+MaG6mjqqo8/z6W9vozTW0tTQw3OdQxw+dqzBrcQwOVrT+DqbKh0H1cnQ1y6Jm+gX7z6uMT4VHVoaqrBxcEQl8JTpTAhgEvXUuHqovxmYwPnarL4AHA+LBWuzgXxrWrqwLS6Ni6FPytzGZn5uHE7XYpD9DyT1k2Q/OcZWVjS0VBUb90EAKCmpQXjpg2QfPz0swhCIPnP0zBp7f4Gc0r0aviM/BskhMDx48dx+PBhjB8/Xgo3MDDAjz/+CG1tbaX7HTt2DOfPn8fNmzfh7OwMAHBwcJC2+/v7Y/DgwZg4cSIAwMnJCatXr0bHjh2xbt066OrqFkszNjYWFhYWsmH1ZmZmcHFxwddffw0vLy+YmJiUeC7W1ta4cuVKeU6/yklNy0S+QoEaRvLKbQ3jaohJSFK6zz9JKbhw8y66tWmC1ZN8cO/BQyze+jvy8vIxuk/XYvEVCgWW/bwfTZzs4FjbUkmK9C6pZqABDQ01PE7Ll4U/TstHrZpaSvcxqaZRLH5qWn6Jw1w7tqyGrCwFzoXzeeaqwNhIExoaakh5LB99kfI4Fza1iv92AEANEy2kPM4tFr+GsfIySFWHcbWC8vSoaPlIzYVtLeU3m2uYaOFRqpLyZFJQnmpUL/i3eJwcKQ7R83QszJD9IFkWlv0gGVrG1aCuqwOt6sZQ19REduLDInEewsDFAaQ6qvqs9WzIvwH79++HoaEhcnNzoVAoMGjQINlz5w0bNiyxEQ8UPLdeu3ZtqRFf1NWrVxEeHo5t27ZJYUIIKBQKREdHy3rdCz19+lRpA//w4cOYM2cOFi1ahCdPnmDr1q348ssv0aVLF1k8PT09ZGaWXNHPzs5Gdna2LCwvJxc62vzRLY1CKFDDyACzfD6Chro6XO1rISn1MbYcOqm0Ib/4p32Iuv8Am2aOqYTc0ruoS0tDnLycjty8qv3jSERERG+3qj7ZHRvyb0Dnzp2xbt06aGtrw9rauths9S+anb7oM+pFpaenY/To0ZgwYUKxbba2tkr3MTMzQ0pKSrFwOzs7bN68GSEhIQgODkZ6ejq8vLxw5coVNGjQQIr36NEjmJuXPEu6v78/5s+XT0oy49P++HKEd6nn8i4xqaYPDXX1YhPbPXqcBlMj5cMBzUyMoKmhDo3nRkrUsaqJ5MdpyM3Lg9ZzZWfx1r04GXYLP874DBY1jJUlR++YtIx85OcLGFfTkIUbV9NA6pN8pfukpuUXi29STQOpT4o//17PQQe1LLSxcovyESP07nn8JA/5+QLVjeW/S9WNi/eSFnqUmovqRXrfqxtrFeuFparncVpBeSo6OqO6kl73Qo9Sc4v1rD9f/h6lFPxbtOe+urE27sRwQk4qLvtBMnQszGRhOhZmyH2cBkVWNnKSU6DIy4NOTdMicUyRnSDvySd6m/EZ+TfAwMAAjo6OsLW1LXXJuZI0atQI9+/fR2RkpNLtTZs2xY0bN+Do6FjsVVJPv7u7OxISEpQ25gvVqVMHy5cvR7Vq1XD27FnZtuvXr8PdveTniGbMmIHHjx/LXlOHflyGs313aGlqor69Nc7fiJLCFAoFzt+MQiNH5TdYGjva4d6Dh1A890x8bEIyzEyqSY14IQQWb92L4Ms3sH76SNTiDKtVRl4+cPd+Nho6PxtNo6YGNHTSQ2RsttJ9ImOy0NBZfjOwkbPy+O+3qoaoe9mI/ZcrIFQVefkCkdEZcHd7djNQTQ1wdzPCjUjlq2vcuJ2Opm5GsrBmjUqOT1VHXp5AxN10NG0oL0/NGhrjRoTyeTf+jkyTxQeA5o1NcCOyIH58YjYepuTI4ujracDVyVCKQ/S81LNhMO3SWhZm9n5bpJwNAwCI3Fw8vvw3zLq0eRZBTQ2mndsg9SwfG1UlQqGosJcqYENeBXTs2BEdOnRA3759cfToUURHR+PQoUMICgoCAPj5+eH06dMYN24cwsLCcPv2bezdu7fUye7c3d1hZmaGU6dOSWH//vsvJk+ejPDwcGRnZyMzMxPr169HamqqrNGemZmJS5cu4YMPPigxfR0dHRgZGcleVXFY/eAP2uO3vy7gj9BLuPtvIhZt2Yun2Tno1a4ZAGD2D7/gv7uCpPifdG6FJxlPsXT7fsQmJOHk1VvYdCAE/Z/7sVm8dS8OngnDotHe0NfTQfLjNCQ/TkNWDnvDqoL9IU/wfutq6NjCELVqamFUP1PoaKsh+H+T040bZIZB3atL8Q+ceIIm9fTQo5MRrGtq4RNPE9S10UHQySeydPV01NC6sQGOn2XFuKr59UACuncxxwcdzGBrrYuJI+yhq6OOw38VjMzw+9wBIwbUluLvOfQALRob45PulrCx1sWwfrXg7GCA3w8/kOJUM9BAXTt92P3vuWgba13UtdMv1pNP755df8SjR1cLeHY0h20tPUwa5QBdHQ0cCi4oTzPGO2LUoGc3s3cfjEfLJibo39MKtta68OlfGy4OBvjtUIIU59cD8RjatzbaNq+OOrb6mDneEckpOQg9z2VwqwINA30YNa4Ho8b1AAD6dWrDqHE96NoUrOjjsmAyGgcskeLHbtgB/To2qOc/DQYuDrAbMwhWn3RD9LeBUpzoVQGwGdEftYb2gWE9B7itmQdNAz3c27znjZ4b0avg0HoVsXv3bkydOhUDBw5ERkYGHB0dsXjxYgAFPfZ//fUXvvzyS7Rv3x5CCNStWxfe3iUPY9fQ0JDWg+/RowcAwMjICHl5eejXrx/i4uIghICDgwMCAgLQtGlTad+9e/fC1tYW7du3r9iTfgd4tmqElLR0rPv9GB4+ToOLrRW+m+wLU+OCofUJD1OhrvZsHVxLUxN8N8UXy38+AO/Zq1GzuhEGerSFz4cdpTi7gs8BAEYt+UF2rHkj+kk3COjddTosA0aG6vD2qg4TIw3E/JONhesf4HF6wd1js+qaeP6RsciYbHy7NREDP6yOQd1rID4pF99seoB7CfIbP+81NYSaGnDqMntVq5qQM49gbKQJn09qobqJFqJiM/HF4ghpAryaZtqy5xBvRKZj4X+j8Kl3bXw6oDb+ScjCnGW3EXP/2eoIbZtXx/SxzyaNmv2fgtU7Nv/6D7b8yqUy32XBpx/CxEgLvgNsUMNEC3diMjB94U1pgkQLM23ZBFV/R6Tj629vY8QAW4wcZIt/4rMw65sIRN97Vp5+/v1f6OpoYOpoBxgaaOLarSeYvuAmcnKr9vOxVYVxMze0Ob5Veu+6bCYA4N6WPQgfMQM6VubQs3m2TO/TmPu40Gs0XJfPgP34Yci6n4Bro2ch+WioFCd+1yFom9eA89wJ0LE0x5OrN3G+x0jkFJkAj95uqrJMXEVRE1V9loAqLCEhAQ0aNMDly5dhZ2cn2xYSEoKYmBj4+PgU269169aYMGECBg0aVK7jZZzmXU56vXx2cZkYer0exXN+AHp98vOUz11B9LKm7fWp7CzQO6Z7bkRlZ+GleU+NrbC0dy6ze3GkSsah9VWYpaUlNm7ciLg45WuaK5OcnIyPP/4YAwcOrMCcERERERERlUwIUWEvVcCh9VVcnz59lIZ36tRJabiZmRmmT59ecRkiIiIiIiKiUrEhT0RERERERCpFVPFn5NmQJyIiIiIiIpVS1RvyfEaeiIiIiIiISIWwR56IiIiIiIhUikIoKjsLlYo98kREREREREQqhD3yREREREREpFL4jDwRERERERERqQz2yBMREREREZFKYY88EREREREREakM9sgTERERERGRShGiavfIsyFPREREREREKkWh4PJzRERERERERKQi2CNPREREREREKoWT3RERERERERGRymCPPBEREREREakUIfiMPBERERERERGpCPbIExERERERkUrhM/JEREREREREpDLYI09EREREREQqpar3yLMhT0RERERERCpFwcnuiIiIiIiIiEhVsEeeiIiIiIiIVEpVH1rPHnkiIiIiIiIiFcIeeSIiIiIiIlIpQsFn5ImIiIiIiIhIRbBHnoiIiIiIiFQKn5EnIiIiIiIiIpXBHnkiIiIiIiJSKYLryBMRERERERGpDoVCVNirvNasWQN7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\n"
          },
          "metadata": {}
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 5. Splitting the Dataset:\n",
        "\n",
        "The dataset can be split into training and testing sets, typically with an 80-20 ratio:"
      ],
      "metadata": {
        "id": "1EcJjx-MM4er"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X = df.drop(columns=[\"Price ($)\"])\n",
        "\n",
        "y = df[\"Price ($)\"]"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T11:59:53.195649Z",
          "iopub.execute_input": "2025-02-25T11:59:53.196047Z",
          "iopub.status.idle": "2025-02-25T11:59:53.202491Z",
          "shell.execute_reply.started": "2025-02-25T11:59:53.196006Z",
          "shell.execute_reply": "2025-02-25T11:59:53.201475Z"
        },
        "id": "dLxqvXIGM4et"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "categorical_cols = X.select_dtypes(include=['object']).columns\n",
        "\n",
        "numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T11:59:53.203776Z",
          "iopub.execute_input": "2025-02-25T11:59:53.204079Z",
          "iopub.status.idle": "2025-02-25T11:59:53.221356Z",
          "shell.execute_reply.started": "2025-02-25T11:59:53.204053Z",
          "shell.execute_reply": "2025-02-25T11:59:53.220323Z"
        },
        "id": "7j49KudWM4ev"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "preprocessor = ColumnTransformer(\n",
        "    transformers=[\n",
        "        ('num', StandardScaler(), numerical_cols),\n",
        "        ('cat', OneHotEncoder(), categorical_cols)\n",
        "    ])\n"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T11:59:53.223874Z",
          "iopub.execute_input": "2025-02-25T11:59:53.224253Z",
          "iopub.status.idle": "2025-02-25T11:59:53.239682Z",
          "shell.execute_reply.started": "2025-02-25T11:59:53.224224Z",
          "shell.execute_reply": "2025-02-25T11:59:53.238456Z"
        },
        "id": "AIvvnid_M4ex"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "print(f\"Training Data Shape: {X_train.shape}, Testing Data Shape: {X_test.shape}\")"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T11:59:53.24128Z",
          "iopub.execute_input": "2025-02-25T11:59:53.241569Z",
          "iopub.status.idle": "2025-02-25T11:59:53.261898Z",
          "shell.execute_reply.started": "2025-02-25T11:59:53.241545Z",
          "shell.execute_reply": "2025-02-25T11:59:53.260903Z"
        },
        "id": "sqxKAzxEM4e0"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "scaler = StandardScaler()\n",
        "\n",
        "X_train_scaled = preprocessor.fit_transform(X_train)\n",
        "\n",
        "X_test_scaled = preprocessor.transform(X_test)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T11:59:53.262888Z",
          "iopub.execute_input": "2025-02-25T11:59:53.263187Z",
          "iopub.status.idle": "2025-02-25T11:59:53.315443Z",
          "shell.execute_reply.started": "2025-02-25T11:59:53.263161Z",
          "shell.execute_reply": "2025-02-25T11:59:53.314287Z"
        },
        "id": "MZ7fKlpPM4e2"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 6. Model Training and Evaluation:"
      ],
      "metadata": {
        "id": "HGKGKtxTM4e4"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "models = {\n",
        "    \"Linear Regression\": LinearRegression(),\n",
        "    \"Random Forest\": RandomForestRegressor(n_estimators=100, random_state=42),\n",
        "    \"KNN Regression\": KNeighborsRegressor(n_neighbors=5),\n",
        "    \"Decision Tree\": DecisionTreeRegressor(random_state=42)\n",
        "}\n",
        "\n",
        "for name, model in models.items():\n",
        "    model.fit(X_train_scaled, y_train)\n",
        "    y_pred = model.predict(X_test_scaled)\n",
        "\n",
        "    rmse = mean_squared_error(y_test, y_pred, squared=False)\n",
        "    r2 = r2_score(y_test, y_pred)\n",
        "\n",
        "    print(f\"📌 {name} Performance:\")\n",
        "    print(f\"   - RMSE: {rmse:.2f}\")\n",
        "    print(f\"   - R² Score: {r2:.2f}\")\n",
        "    print(\"-\" * 40)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T11:59:53.316744Z",
          "iopub.execute_input": "2025-02-25T11:59:53.317139Z",
          "iopub.status.idle": "2025-02-25T12:00:36.078217Z",
          "shell.execute_reply.started": "2025-02-25T11:59:53.317088Z",
          "shell.execute_reply": "2025-02-25T12:00:36.077087Z"
        },
        "id": "I-X8DV2tM4e5"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Output:\n",
        "This code evaluates and displays the performance of four regression models:\n",
        "\n",
        "\n",
        "1. Linear Regression\n",
        "2. Random Forest Regressor\n",
        "3. KNN Regressor\n",
        "4. Decision Tree Regressor\n",
        "\n",
        "For each model, it calculates the Root Mean Squared Error (RMSE) and the R-squared (R²) Score, printing them for comparison."
      ],
      "metadata": {
        "id": "kErJIwX7M4e_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 7. Model Evaluation with Additional Metrics:"
      ],
      "metadata": {
        "id": "_8koXSvOM4fB"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "mae = mean_absolute_error(y_test, y_pred)\n",
        "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
        "r2 = r2_score(y_test, y_pred)\n",
        "\n",
        "print(f\"\\n📌 {name} Performance:\")\n",
        "print(f\"Mean Absolute Error (MAE): {mae:.2f}\")\n",
        "print(f\"Root Mean Squared Error (RMSE): {rmse:.2f}\")\n",
        "print(f\"R² Score: {r2:.2f}\")\n",
        "print(\"-\" * 40)# 4. Final Model Selection and Prediction:"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T12:00:36.079301Z",
          "iopub.execute_input": "2025-02-25T12:00:36.079567Z",
          "iopub.status.idle": "2025-02-25T12:00:36.09119Z",
          "shell.execute_reply.started": "2025-02-25T12:00:36.079545Z",
          "shell.execute_reply": "2025-02-25T12:00:36.089893Z"
        },
        "id": "wiP1YlzJM4fC"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Output:**\n",
        "\n",
        "For each model, this code calculates and prints:\n",
        "\n",
        "\n",
        "* Mean Absolute Error (MAE): The average magnitude of errors in predictions.\n",
        "\n",
        "* Root Mean Squared Error (RMSE): The square root of the average squared differences between the predicted and actual values.\n",
        "\n",
        "* R-squared (R²) Score: A statistical measure that explains the proportion of the variance in the dependent variable that is predictable from the independent variables\n"
      ],
      "metadata": {
        "id": "sAeIj7HJM4fF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 8. Actual vs Predicted Price Comparison:"
      ],
      "metadata": {
        "id": "20wzqjVrM4fG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "best_model = models[\"Random Forest\"]\n",
        "\n",
        "predicted_prices = best_model.predict(X_test_scaled)\n",
        "\n",
        "actual_values = y_test.values[:10]\n",
        "predicted_values = predicted_prices[:10]\n",
        "\n",
        "comparison_df = pd.DataFrame({\"Actual Price\": actual_values, \"Predicted Price\": predicted_values})\n",
        "print(\"\\nActual vs Predicted Prices:\")\n",
        "print(comparison_df)\n",
        "\n",
        "sample_instance = X_test_scaled[0].reshape(1, -1)\n",
        "predicted_price = best_model.predict(sample_instance)\n",
        "\n",
        "print(f\"\\nPredicted Price for the Given Features: ${predicted_price[0]:.2f}\")"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T12:00:36.092042Z",
          "iopub.execute_input": "2025-02-25T12:00:36.092449Z",
          "iopub.status.idle": "2025-02-25T12:00:36.196306Z",
          "shell.execute_reply.started": "2025-02-25T12:00:36.092421Z",
          "shell.execute_reply": "2025-02-25T12:00:36.195262Z"
        },
        "id": "680_dFwLM4fI"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Output:**\n",
        "\n",
        "\n",
        "1. Actual vs Predicted Prices (for first 10 instances): The code prints a DataFrame that compares the Actual Price from the test set with the Predicted Price for the first 10 test samples.\n",
        "\n",
        "2. Predicted Price for a Single Instance: The code predicts the price for the first sample instance from the test set and prints the predicted pric\n"
      ],
      "metadata": {
        "id": "qLzKCYZ-M4fJ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 9. Price Prediction Using Euclidean Distance:"
      ],
      "metadata": {
        "id": "ZdK7FkRRM4fL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "# Load the dataset\n",
        "df = pd.read_csv(\"/kaggle/input/laptop-price-dataset/laptop_prices.csv\")\n",
        "\n",
        "# Define the actual feature values for the laptop\n",
        "numeric_cols = [\"RAM (GB)\", \"Storage\", \"Screen Size (inch)\", \"Battery Life (hours)\", \"Weight (kg)\", \"Resolution\"]\n",
        "\n",
        "# Define the actual feature values for the laptop\n",
        "actual_values = np.array([numeric_cols == [\"RAM (GB)\", \"Storage\", \"Screen Size (inch)\", \"Battery Life (hours)\", \"Weight (kg)\", \"Resolution\"]\n",
        "]).reshape(1, -1)\n",
        "\n",
        "# Keep only numeric columns (excluding 'Price ($)' before computing distances)\n",
        "numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n",
        "numeric_cols.remove(\"Price ($)\")  # Ensure 'Price ($)' is not included\n",
        "\n",
        "# Compute the Euclidean distance between the actual values and all rows in the dataset\n",
        "df[\"distance\"] = np.linalg.norm(df[numeric_cols].values - actual_values, axis=1)\n",
        "\n",
        "# Find the row with the smallest distance (the closest match)\n",
        "closest_row = df.loc[df[\"distance\"].idxmin()]\n",
        "\n",
        "# Get the actual price for the closest match\n",
        "actual_price = closest_row[\"Price ($)\"]\n",
        "print(f\"\\nActual Price for the Given Features: ${actual_price:.2f}\")"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T12:00:36.197397Z",
          "iopub.execute_input": "2025-02-25T12:00:36.197684Z",
          "iopub.status.idle": "2025-02-25T12:00:36.236804Z",
          "shell.execute_reply.started": "2025-02-25T12:00:36.19766Z",
          "shell.execute_reply": "2025-02-25T12:00:36.235567Z"
        },
        "id": "YhZUCpL2M4fN"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "# Load the dataset\n",
        "df = pd.read_csv(\"/kaggle/input/laptop-price-dataset/laptop_prices.csv\")\n",
        "\n",
        "# Define the actual feature values for the laptop\n",
        "actual_values = np.array([numeric_cols == [\"RAM (GB)\", \"Storage\", \"Screen Size (inch)\", \"Battery Life (hours)\", \"Weight (kg)\", \"Resolution\"]\n",
        "]).reshape(1, -1)\n",
        "\n",
        "# Keep only numeric columns (excluding 'Price ($)' before computing distances)\n",
        "numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n",
        "numeric_cols.remove(\"Price ($)\")  # Ensure 'Price ($)' is not included\n",
        "\n",
        "# Compute the Euclidean distance between the actual values and all rows in the dataset\n",
        "df[\"distance\"] = np.linalg.norm(df[numeric_cols].values - actual_values, axis=1)\n",
        "\n",
        "# Find the row with the smallest distance (the closest match)\n",
        "closest_row = df.loc[df[\"distance\"].idxmin()]\n",
        "\n",
        "# Get the actual price for the closest match\n",
        "actual_price = closest_row[\"Price ($)\"]\n",
        "print(f\"\\nActual Price for the Given Features: ${actual_price:.2f}\")\n"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T12:00:36.237908Z",
          "iopub.execute_input": "2025-02-25T12:00:36.238254Z",
          "iopub.status.idle": "2025-02-25T12:00:36.274631Z",
          "shell.execute_reply.started": "2025-02-25T12:00:36.238228Z",
          "shell.execute_reply": "2025-02-25T12:00:36.273474Z"
        },
        "id": "QSSzcPk6M4fP"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Explanation of Output:**\n",
        "\n",
        "* Euclidean Distance Calculation: The code computes the Euclidean distance between the given laptop's feature values (actual_values) and all the rows in the dataset. The features in the dataset (like RAM, Storage, etc.) are numeric values, and the distance is computed using the formula:\n",
        "\n",
        "distance\n",
        "=\n",
        "∑\n",
        "𝑖\n",
        "=\n",
        "1\n",
        "𝑛\n",
        "(\n",
        "𝑥\n",
        "𝑖\n",
        "−\n",
        "𝑦\n",
        "𝑖\n",
        ")\n",
        "2\n",
        "distance=\n",
        "i=1\n",
        "∑\n",
        "n\n",
        "​\n",
        " (x\n",
        "i\n",
        "​\n",
        " −y\n",
        "i\n",
        "​\n",
        " )\n",
        "2\n",
        "\n",
        "​\n",
        "\n",
        "Where x_i are the features of the actual laptop and y_i are the features from each row in the dataset.\n",
        "\n",
        "\n",
        "* Finding the Closest Match: The row with the smallest distance (i.e., the most similar laptop) is found using df[\"distance\"].idxmin(). This will return the index of the row with the closest match.\n",
        "\n",
        "* Output: The code retrieves the actual price of the closest matching laptop based on the computed distance and prints it.\n"
      ],
      "metadata": {
        "id": "my1kxa0-M4fQ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 10. Laptop Price Comparison (Final Prediction):"
      ],
      "metadata": {
        "id": "3AhXARY9M4fT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import re\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "# Load the dataset\n",
        "df = pd.read_csv(\"/kaggle/input/laptop-price-dataset/laptop_prices.csv\")\n",
        "\n",
        "# Function to convert storage values to numeric GB\n",
        "def convert_storage(storage):\n",
        "    if isinstance(storage, str):\n",
        "        num = re.findall(r'\\d+', storage)\n",
        "        if num:\n",
        "            value = float(num[0])\n",
        "            if \"TB\" in storage:\n",
        "                return value * 1000  # Convert TB to GB\n",
        "            return value  # Keep GB as is\n",
        "    return np.nan\n",
        "\n",
        "# Function to convert resolution to total pixels\n",
        "def convert_resolution(res):\n",
        "    try:\n",
        "        width, height = map(int, res.split(\"x\"))\n",
        "        return width * height  # Total pixel count\n",
        "    except:\n",
        "        return np.nan\n",
        "\n",
        "# Apply conversion functions\n",
        "df[\"Storage\"] = df[\"Storage\"].apply(convert_storage)\n",
        "df[\"Resolution\"] = df[\"Resolution\"].apply(convert_resolution)\n",
        "\n",
        "# Selecting numeric columns\n",
        "numeric_cols = [\"RAM (GB)\", \"Storage\", \"Screen Size (inch)\", \"Battery Life (hours)\", \"Weight (kg)\", \"Resolution\"]\n",
        "\n",
        "# Handle missing values by filling with column mean\n",
        "df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())\n",
        "\n",
        "# Define the actual feature values for the laptop (Example values)\n",
        "actual_values = np.array([64, 512, 17.3, 8.9, 1.42, 2560 * 1440]).reshape(1, -1)  # Replace with actual values\n",
        "\n",
        "# Compute the Euclidean distance between the actual values and all rows in the dataset\n",
        "df[\"distance\"] = np.sqrt(np.sum((df[numeric_cols].values - actual_values) ** 2, axis=1))\n",
        "\n",
        "# Find the row with the smallest distance (the closest match)\n",
        "closest_row = df.loc[df[\"distance\"].idxmin()]\n",
        "\n",
        "# Get the actual price for the closest match\n",
        "actual_price = closest_row[\"Price ($)\"]\n",
        "\n",
        "# Prepare the dataset for model training\n",
        "X = df[numeric_cols]  # Features\n",
        "y = df[\"Price ($)\"]   # Target variable\n",
        "\n",
        "# Train-test split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Standardize features\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "actual_values_scaled = scaler.transform(actual_values)  # Scale the input values\n",
        "\n",
        "# Train a Random Forest model\n",
        "rf_model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
        "rf_model.fit(X_train_scaled, y_train)\n",
        "\n",
        "# Predict the price using the trained model\n",
        "predicted_price = rf_model.predict(actual_values_scaled)\n",
        "\n",
        "# Print Actual and Predicted Price together\n",
        "print(\"\\n📌 **Laptop Price Comparison**\")\n",
        "print(f\"✅ Actual Price (Closest Match)  : ${actual_price:.2f}\")\n",
        "print(f\"🎯 Predicted Price (ML Model)   : ${predicted_price[0]:.2f}\")\n"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-02-25T12:00:36.275785Z",
          "iopub.execute_input": "2025-02-25T12:00:36.27619Z",
          "iopub.status.idle": "2025-02-25T12:00:38.899998Z",
          "shell.execute_reply.started": "2025-02-25T12:00:36.276153Z",
          "shell.execute_reply": "2025-02-25T12:00:38.898735Z"
        },
        "id": "R2xet1L4M4fU"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Output:**\n",
        "\n",
        "This code performs the following steps:\n",
        "\n",
        "1. Preprocessing:\n",
        "\n",
        "* Converts storage values to GB.\n",
        "\n",
        "* Converts the resolution to total pixel count.\n",
        "\n",
        "* Handles missing values by filling them with the mean of the respective columns.\n",
        "\n",
        "2. Distance Calculation:\n",
        "\n",
        "* It calculates the Euclidean distance between the given laptop feature values and the rest of the dataset to find the closest match.\n",
        "\n",
        "3. Model Training:\n",
        "\n",
        "* A Random Forest Regressor is trained using the training data after scaling the features.\n",
        "\n",
        "4. Price Prediction:\n",
        "\n",
        "* The price for the closest match is found using the Euclidean distance.\n",
        "\n",
        "* The model then predicts the price based on the input values.\n"
      ],
      "metadata": {
        "id": "e-dPKCXyM4fX"
      }
    }
  ]
}
