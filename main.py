import itertools
import pandas as pd
from algorithms.ppo.PPO import PPO

# 定義超參數 Grid Search 範圍
param_grid = {
    "alpha": [1e-4, 3e-4, 5e-4],  # 學習率
    "n_epochs": [10, 20, 30],  # 訓練回合數
    "batch_size": [64, 128, 256],  # 批次大小
    "layer1_size": [256, 512, 1024],  # 第一層神經元數量
    "layer2_size": [256, 512, 1024],  # 第二層神經元數量
    "policy_clip": [0.1, 0.2, 0.3],  # PPO clip 參數
    "gamma": [0.95, 0.99, 0.995],  # 折扣因子
    "entropy": [0, 0.005, 0.01],  # 熵係數
    "gae_lambda": [0.95, 0.97, 0.99],  # GAE lambda
}

# 生成所有可能的超參數組合
param_combinations = list(itertools.product(*param_grid.values()))
results = []

# 遍歷所有超參數組合
for i, params in enumerate(param_combinations):
    param_dict = dict(zip(param_grid.keys(), params))

    print(f"🔹 Grid Search Round {i+1}/{len(param_combinations)} - Testing {param_dict}")

    # 訓練 PPO
    model = PPO(**param_dict)
    final_reward = model.train(verbose=False)

    # 記錄結果
    results.append({**param_dict, "final_reward": final_reward})
    print(f"✅ Params: {param_dict}, Final Reward: {final_reward}")

# 儲存結果到 CSV
results_df = pd.DataFrame(results)
results_df.to_csv("ppo_grid_search_results.csv", index=False)
print("📊 Grid search results saved to ppo_grid_search_results.csv")

def main():

    plot.initialize()
    mp.set_start_method('spawn')

    for i in range(1):
        print(f"---------- round {i} ----------")

        if not os.path.isfile(f'plots/ddpg/{i}2_testing.png'):
            ddpg = DDPG(state_type='indicators', djia_year=2019, repeat=i)
            ddpg.train()
            ddpg.test()

        if not os.path.isfile(f'plots/ppo/{i}2_testing.png'):
            ppo = PPO(state_type='indicators', djia_year=2019, repeat=i)
            ppo.train()
            ppo.test()

        if not os.path.isfile(f'plots/a2c/{i}2_testing.png'):
            a2c = A2C(n_agents=8, state_type='indicators', djia_year=2019, repeat=i)
            a2c.train()
            a2c.test()


if __name__ == '__main__':
    main()
