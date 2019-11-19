import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

#df = pd.read_csv("./dga_1st_round_train.csv")
df = pd.read_csv("./dga_1st_round_answer.csv")
# 시각화
ax = sns.countplot(x="class", data=df)
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}'.format(height),
            ha="center")

plt.title("DGA 클래스별 데이터 수")
plt.show()