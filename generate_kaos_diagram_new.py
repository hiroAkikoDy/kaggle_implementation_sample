import networkx as nx
import matplotlib.pyplot as plt
import japanize_matplotlib  # 凡例の日本語表示に使用
import matplotlib.patches as mpatches

def create_and_visualize_kaos_local():
    # 1. グラフの初期化
    G = nx.DiGraph()

    # 2. ノードデータの定義（実践記録に基づいたKaggle Titanic攻略の階層）
    node_data = {
        "ROOT": {"jp": "高スコア達成", "type": "goal", "color": "#FFB6C1"},  # LightPink
        "G1":   {"jp": "データ品質保証 (品質)", "type": "goal", "color": "#ADD8E6"},  # LightBlue
        "G11":  {"jp": "欠損値処理", "type": "goal", "color": "#ADD8E6"},
        "G111": {"jp": "数値欠損値", "type": "goal", "color": "#ADD8E6"},
        "G112": {"jp": "カテゴリ欠損値", "type": "goal", "color": "#ADD8E6"},
        "G12":  {"jp": "外れ値処理", "type": "goal", "color": "#ADD8E6"},
        "G13":  {"jp": "型整合性", "type": "goal", "color": "#ADD8E6"},
        "G2":   {"jp": "特徴量生成 (特徴量)", "type": "goal", "color": "#90EE90"},  # LightGreen
        "G21":  {"jp": "ドメイン特徴量", "type": "goal", "color": "#90EE90"},
        "G22":  {"jp": "統計特徴量", "type": "goal", "color": "#90EE90"},
        "G23":  {"jp": "相互作用特徴量", "type": "goal", "color": "#90EE90"},
        "G3":   {"jp": "モデル構築 (モデル)", "type": "goal", "color": "#FFFFE0"},  # LightYellow
        "G31":  {"jp": "ベースライン", "type": "goal", "color": "#FFFFE0"},
        "G32":  {"jp": "最適化", "type": "goal", "color": "#FFFFE0"},
        "G33":  {"jp": "アンサンブル", "type": "goal", "color": "#FFFFE0"},
        "G4":   {"jp": "評価・改善 (評価)", "type": "goal", "color": "#E6E6FA"},  # Lavender
        "G41":  {"jp": "CV実施", "type": "goal", "color": "#E6E6FA"},
        "G42":  {"jp": "LB確認", "type": "goal", "color": "#E6E6FA"},
        "G43":  {"jp": "改善サイクル", "type": "goal", "color": "#E6E6FA"}
    }

    for node_id, attr in node_data.items():
        G.add_node(node_id, **attr)

    # 3. エッジの定義（依存関係）
    G.add_edges_from()

    # 4. レイアウト計算（トポロジカル順序で階層化） 
    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]["layer"] = layer
    pos = nx.multipartite_layout(G, subset_key="layer", align='vertical')

    # 5. 描画設定
    fig, ax = plt.subplots(figsize=(14, 10))
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    
    # グラフ本体の描画（ノード内はIDのみ）
    nx.draw(G, pos, with_labels=True, 
            node_color=node_colors, node_size=2000, 
            font_size=9, font_weight="bold", arrows=True,
            arrowsize=15, edge_color="#CCCCCC", ax=ax)

    # 6. 右側に日本語ラベルの凡例（Legend）を表示 
    legend_text = "【要求定義・凡例】\n"
    for node_id, attr in node_data.items():
        legend_text += f"{node_id}: {attr['jp']}\n"
    
    # ボックスとして配置
    plt.text(1.02, 0.5, legend_text, transform=ax.transAxes, 
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#999999', alpha=0.9))

    plt.title("KAOS Goal Model: Kaggle Titanic Practice", fontsize=16, fontweight="bold", pad=20)
    
    # 7. 保存と表示
    output_filename = "kaos_model_local.png"
    plt.savefig(output_filename, format="png", bbox_inches="tight", dpi=300)
    print(f"画像を保存しました: {output_filename}")
    
    # ローカル環境でウィンドウを開いて表示
    plt.show()

if __name__ == "__main__":
    create_and_visualize_kaos_local()