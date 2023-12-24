from matplotlib import pyplot as plt
import numpy as np
import pandas as pd



def main():
    data = pd.read_csv("./runs/8296625701/val_preds.csv")
    data_x = data["y"]
    data_y = data['preds']
    
    plt.scatter(data_x, data_y)
    plt.xlabel("Real", fontsize=16)
    plt.ylabel("Predicted", fontsize=16)
    plt.title("Predicted vs Real")
    plt.plot([0, 1], [0, 1], color='black', linewidth=4, linestyle='--')

    plt.show()


if __name__ == "__main__":
    main()