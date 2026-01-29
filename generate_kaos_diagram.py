### NetworkXによる可視化コード

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']

# グラフ作成
G = nx.DiGraph()

# ノード追加
nodes = {
    'ROOT': {'label': 'ROOT: 高スコア達成', 'color': 'lightpink'},
    
    # G1系統
    'G1': {'label': 'G1: データ品質保証', 'color': 'lightblue'},
    'G11': {'label': 'G11: 欠損値処理', 'color': 'lightblue'},
    'G111': {'label': 'G111: 数値欠損値', 'color': 'lightblue'},
    'G112': {'label': 'G112: カテゴリ欠損値', 'color': 'lightblue'},
    'G12': {'label': 'G12: 外れ値処理', 'color': 'lightblue'},
    'G13': {'label': 'G13: 型整合性', 'color': 'lightblue'},
    
    # G2系統
    'G2': {'label': 'G2: 特徴量生成', 'color': 'lightgreen'},
    'G21': {'label': 'G21: ドメイン特徴量', 'color': 'lightgreen'},
    'G22': {'label': 'G22: 統計特徴量', 'color': 'lightgreen'},
    'G23': {'label': 'G23: 相互作用特徴量', 'color': 'lightgreen'},
    
    # G3系統
    'G3': {'label': 'G3: モデル構築', 'color': 'lightyellow'},
    'G31': {'label': 'G31: ベースライン', 'color': 'lightyellow'},
    'G32': {'label': 'G32: 最適化', 'color': 'lightyellow'},
    'G33': {'label': 'G33: アンサンブル', 'color': 'lightyellow'},
    
    # G4系統
    'G4': {'label': 'G4: 評価・改善', 'color': 'lavender'},
    'G41': {'label': 'G41: CV実施', 'color': 'lavender'},
    'G42': {'label': 'G42: LB確認', 'color': 'lavender'},
    'G43': {'label': 'G43: 改善サイクル', 'color': 'lavender'},
}

for node, attrs in nodes.items():
    G.add_node(node, **attrs)

# エッジ追加
edges = [
    ('ROOT', 'G1'), ('ROOT', 'G2'), ('ROOT', 'G3'), ('ROOT', 'G4'),
    ('G1', 'G11'), ('G1', 'G12'), ('G1', 'G13'),
    ('G11', 'G111'), ('G11', 'G112'),
    ('G2', 'G21'), ('G2', 'G22'), ('G2', 'G23'),
    ('G3', 'G31'), ('G3', 'G32'), ('G3', 'G33'),
    ('G4', 'G41'), ('G4', 'G42'), ('G4', 'G43'),
]

G.add_edges_from(edges)

# 描画
pos = nx.spring_layout(G, k=2, iterations=50)
plt.figure(figsize=(16, 12))

colors = [G.nodes[node]['color'] for node in G.nodes()]
labels = {node: G.nodes[node]['label'] for node in G.nodes()}

nx.draw(G, pos, 
        node_color=colors,
        labels=labels,
        node_size=3000,
        font_size=9,
        font_weight='bold',
        arrows=True,
        arrowsize=20,
        edge_color='gray',
        linewidths=2,
        width=2)

plt.title("KAOS: Kaggle Competition Goal Structure", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('kaggle_kaos.png', dpi=300, bbox_inches='tight')
plt.show()

print("KAOS図を生成しました: kaggle_kaos.png")

