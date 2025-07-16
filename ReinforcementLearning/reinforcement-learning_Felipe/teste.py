from environment import Environment
import pickle

if __name__ == '__main__':
    f = open("data_backup/0.4 segundos absurdo.pkl", "rb")
    env = pickle.load(f)
    print("Environment loaded successfully.")

    env.plot_policy()
