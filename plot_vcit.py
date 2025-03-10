import pandas as pd
import matplotlib.pyplot as plt

# 讀取 VCIT 持股數據
vcit_shares_path = "plots/ppo/0_vcit_shares.csv"  # 確保檔案名稱正確
vcit_shares_df = pd.read_csv(vcit_shares_path)

# 確保日期為時間格式
vcit_shares_df["Date"] = pd.to_datetime(vcit_shares_df["Date"])

# 繪製折線圖
plt.figure(figsize=(10, 5))
plt.plot(vcit_shares_df["Date"], vcit_shares_df["VCIT Shares"], label="VCIT Shares", color="blue")
plt.xlabel("Date")
plt.ylabel("VCIT Shares")
plt.title("VCIT Holding Over Time")
plt.legend()

# 儲存與顯示
plt.savefig("plots/ppo/vcit_shares_plot.png")
plt.show()
