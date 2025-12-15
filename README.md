# 公共運輸路線最佳化系統

簡單三步，快速體驗三種演算法求解路線規劃問題。

## 核心成果

| 方法 | 成本 | 時間 |
|------|------|------|
| Sequential | 303.4 | 1.67s |
| Genetic Algorithm | 289.3 | 0.13s |
| **Hybrid** | **277.9** | **0.14s** ✓ |

## 快速開始

### 1. 安裝依賴
```bash
pip install numpy scipy scikit-learn matplotlib
```

### 2. 下載並執行
```bash
git clone https://github.com/RexW/bus-route-optimization.git
cd bus-route-optimization
python src/main.py
```

### 3. 查看結果

執行後會生成 2 張圖片：
- `bus_routing_scenarios.png` - 三種方法的空間解視覺化
- `bus_routing_results.png` - 成本/時間/收斂曲線對比

## 文件說明

- `src/main.py` - 完整實現（單檔案，430 行）
- `requirements.txt` - 依賴列表

## 方法簡介

**方法一：順序求解法** (Sequential)
- 先用 K-Means 選站點，再用貪心 TSP 排序
- 快速但可能局部次優

**方法二：遺傳演算法** (GA)
- 全域搜索，初期效率低，收斂慢

**方法三：混合策略** (Hybrid) ⭐ 推薦
- 結合方法一的快速性 + GA 的全域搜索
- 成本最低，執行時間快

## 試試看

修改 `src/main.py` 中的參數：
```python
# 改變問題規模
generate_sample_data(n_candidates=20, n_passengers=50)

# 改變 GA 代數
GeneticMethod(model, generations=100)

# 改變成本權重
LRPModel(..., weights=(0.7, 0.3))
```

## 依賴

- numpy>=1.24.0
- scipy>=1.10.0
- scikit-learn>=1.3.0
- matplotlib>=3.7.0