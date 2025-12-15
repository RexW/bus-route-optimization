# coding: utf-8
"""
公共運輸路線最佳化系統 - 快速實現版本
執行單一檔案即可完整示範三種方法

問題規模：
- 候選站點：15 個
- 乘客：30 人（3 個聚類）
- 車輛：1 輛（單路線）
- 目標：最小化 J = w1*C_ops + w2*C_access
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import cdist, euclidean
from sklearn.cluster import KMeans
import random
import time
import platform

system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['SimHei', 'STHeiti', 'Arial Unicode MS']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 第一部分：資料層
# ============================================================================

def generate_sample_data(n_candidates=15, n_passengers=30, seed=42):
    """生成簡化測試資料"""
    np.random.seed(seed)
    candidates = np.random.uniform(0, 100, (n_candidates, 2))

    passengers = []
    cluster_centers = [
        np.array([30, 30]),
        np.array([50, 70]),
        np.array([70, 40])
    ]

    for center in cluster_centers:
        cluster_passengers = center + np.random.normal(0, 8, (n_passengers//3, 2))
        passengers.extend(cluster_passengers)

    passengers = np.array(passengers)
    start = np.array([0, 0])
    end = np.array([100, 100])

    return candidates, passengers, start, end


# ============================================================================
# 第一部分：模型層
# ============================================================================

class LRPModel:
    """簡化的位置-路由問題模型"""

    def __init__(self, candidates, passengers, start, end, weights=(0.6, 0.4)):
        self.candidates = candidates
        self.passengers = passengers
        self.start = start
        self.end = end
        self.w1, self.w2 = weights

        all_points = np.vstack([
            self.start.reshape(1, -1),
            self.candidates,
            self.end.reshape(1, -1)
        ])
        self.dist_matrix = cdist(all_points, all_points, metric='euclidean')
        self.n_candidates = len(candidates)

    def calculate_route_distance(self, selected_stations):
        """計算公車行駛距離"""
        route = [0] + [idx + 1 for idx in selected_stations] + [self.n_candidates + 1]

        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.dist_matrix[route[i], route[i + 1]]

        return total_distance

    def calculate_access_cost(self, selected_stations):
        """計算乘客存取成本"""
        if len(selected_stations) == 0:
            return float('inf')

        selected_coords = self.candidates[selected_stations]
        distances = cdist(self.passengers, selected_coords, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        return np.sum(min_distances)

    def objective_function(self, selected_stations):
        """計算目標函數"""
        if len(selected_stations) == 0:
            return float('inf')

        c_ops = self.calculate_route_distance(selected_stations)
        c_access = self.calculate_access_cost(selected_stations)

        return self.w1 * c_ops + self.w2 * c_access


# ============================================================================
# 第二部分：演算法層
# ============================================================================

class SequentialMethod:
    """方法一：順序求解法"""

    def __init__(self, model, n_stations=5):
        self.model = model
        self.n_stations = n_stations

    def solve(self):
        start_time = time.time()

        kmeans = KMeans(n_clusters=self.n_stations, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.model.passengers)
        centers = kmeans.cluster_centers_

        selected = []
        for center in centers:
            distances = np.linalg.norm(self.model.candidates - center, axis=1)
            nearest_idx = np.argmin(distances)
            selected.append(nearest_idx)

        selected = list(set(selected))
        selected_ordered = self._nearest_neighbor_tsp(selected)

        cost = self.model.objective_function(selected_ordered)
        elapsed = time.time() - start_time

        return {
            'method': 'Sequential (K-Means + Greedy)',
            'stations': selected_ordered,
            'cost': cost,
            'n_stations': len(selected_ordered),
            'time': elapsed
        }

    def _nearest_neighbor_tsp(self, selected):
        """貪心 TSP"""
        unvisited = set(selected)
        current = None
        route = []

        while unvisited:
            if current is None:
                distances = [self.model.dist_matrix[0, idx + 1] for idx in unvisited]
                next_station = list(unvisited)[np.argmin(distances)]
            else:
                distances = [self.model.dist_matrix[current + 1, idx + 1] for idx in unvisited]
                next_station = list(unvisited)[np.argmin(distances)]

            route.append(next_station)
            unvisited.remove(next_station)
            current = next_station

        return route


class GeneticMethod:
    """方法二：遺傳演算法"""

    def __init__(self, model, n_stations=5, generations=50, pop_size=30):
        self.model = model
        self.n_stations = n_stations
        self.generations = generations
        self.pop_size = pop_size

    def solve(self):
        start_time = time.time()

        population = [
            np.random.choice(len(self.model.candidates),
                           self.n_stations,
                           replace=False)
            for _ in range(self.pop_size)
        ]

        best_cost = float('inf')
        best_solution = None
        cost_history = []

        for gen in range(self.generations):
            fitness = []
            costs = []
            for ind in population:
                cost = self.model.objective_function(ind)
                costs.append(cost)
                fitness.append(1.0 / (cost + 1e-6))

            best_gen_idx = np.argmin(costs)
            best_gen_cost = costs[best_gen_idx]
            cost_history.append(best_gen_cost)

            if best_gen_cost < best_cost:
                best_cost = best_gen_cost
                best_solution = population[best_gen_idx].copy()

            new_population = []
            for _ in range(self.pop_size):
                parent1 = self._select(population, fitness)
                parent2 = self._select(population, fitness)
                child = self._crossover(parent1, parent2)

                if random.random() < 0.2:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

        elapsed = time.time() - start_time

        return {
            'method': 'Genetic Algorithm',
            'stations': list(best_solution),
            'cost': best_cost,
            'n_stations': len(best_solution),
            'time': elapsed,
            'cost_history': cost_history
        }

    def _select(self, population, fitness):
        total_fitness = sum(fitness)
        if total_fitness == 0:
            return population[random.randint(0, len(population) - 1)].copy()

        pick = random.uniform(0, total_fitness)
        current = 0
        for ind, f in zip(population, fitness):
            current += f
            if current > pick:
                return ind.copy()
        return population[-1].copy()

    def _crossover(self, parent1, parent2):
        size = len(parent1)
        if size < 2:
            return parent1.copy()

        start, end = sorted(random.sample(range(size), 2))
        child = parent1.copy()

        for i in range(start, end + 1):
            child[i] = parent2[i]

        return child

    def _mutate(self, individual):
        if len(individual) < 2:
            return individual

        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual


class HybridMethod:
    """方法三：混合策略（熱啟動）"""

    def __init__(self, model, n_stations=5, generations=50, pop_size=30):
        self.model = model
        self.n_stations = n_stations
        self.generations = generations
        self.pop_size = pop_size

    def solve(self):
        start_time = time.time()

        seq_method = SequentialMethod(self.model, self.n_stations)
        seed_result = seq_method.solve()
        seed_solution = np.array(seed_result['stations'])

        population = [seed_solution.copy()]
        for _ in range(self.pop_size - 1):
            population.append(
                np.random.choice(len(self.model.candidates),
                               self.n_stations,
                               replace=False)
            )

        best_cost = float('inf')
        best_solution = seed_solution.copy()
        cost_history = []

        for gen in range(self.generations):
            fitness = []
            costs = []
            for ind in population:
                cost = self.model.objective_function(ind)
                costs.append(cost)
                fitness.append(1.0 / (cost + 1e-6))

            best_gen_idx = np.argmin(costs)
            best_gen_cost = costs[best_gen_idx]
            cost_history.append(best_gen_cost)

            if best_gen_cost < best_cost:
                best_cost = best_gen_cost
                best_solution = population[best_gen_idx].copy()

            new_population = []
            for _ in range(self.pop_size):
                parent1 = self._select(population, fitness)
                parent2 = self._select(population, fitness)
                child = self._crossover(parent1, parent2)

                if random.random() < 0.2:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

        elapsed = time.time() - start_time

        return {
            'method': 'Hybrid (Hot-Start GA)',
            'stations': list(best_solution),
            'cost': best_cost,
            'n_stations': len(best_solution),
            'time': elapsed,
            'cost_history': cost_history,
            'seed_cost': seed_result['cost']
        }

    def _select(self, population, fitness):
        total_fitness = sum(fitness)
        if total_fitness == 0:
            return population[random.randint(0, len(population) - 1)].copy()

        pick = random.uniform(0, total_fitness)
        current = 0
        for ind, f in zip(population, fitness):
            current += f
            if current > pick:
                return ind.copy()
        return population[-1].copy()

    def _crossover(self, parent1, parent2):
        size = len(parent1)
        if size < 2:
            return parent1.copy()

        start, end = sorted(random.sample(range(size), 2))
        child = parent1.copy()

        for i in range(start, end + 1):
            child[i] = parent2[i]

        return child

    def _mutate(self, individual):
        if len(individual) < 2:
            return individual

        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual


# ============================================================================
# 第三部分：實驗評估層
# ============================================================================

def visualize_problem_scenario(model, results):
    """繪製問題情境和三種方法的解"""

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('公共運輸路線最佳化 - 問題情境與三種方法的解',
                 fontsize=16, fontweight='bold', y=0.98)

    seq_result = results[0]
    ga_result = results[1]
    hybrid_result = results[2]

    results_list = [
        (seq_result, axes[0], '方法一：順序求解法'),
        (ga_result, axes[1], '方法二：遺傳演算法'),
        (hybrid_result, axes[2], '方法三：混合策略（熱啟動）')
    ]

    for result, ax, title in results_list:
        # 繪製背景：候選站點
        ax.scatter(model.candidates[:, 0], model.candidates[:, 1],
                  s=100, alpha=0.5, color='lightgray', edgecolor='gray',
                  label='候選站點（未選中）', zorder=2)

        # 繪製乘客分佈
        ax.scatter(model.passengers[:, 0], model.passengers[:, 1],
                  s=20, alpha=0.4, color='blue', label='乘客位置', zorder=1)

        # 繪製起點 A 和終點 B
        ax.scatter(*model.start, s=300, marker='s', color='green',
                  edgecolor='darkgreen', linewidth=2, label='起點 A', zorder=4)
        ax.scatter(*model.end, s=300, marker='s', color='red',
                  edgecolor='darkred', linewidth=2, label='終點 B', zorder=4)

        # 獲取選中的站點
        selected_stations = result['stations']
        selected_coords = model.candidates[selected_stations]

        # 繪製選中的站點
        ax.scatter(selected_coords[:, 0], selected_coords[:, 1],
                  s=200, alpha=0.9, color='orange', edgecolor='darkred',
                  linewidth=2, label='選中的站點', zorder=3)

        # 標註站點索引
        for idx, coord in zip(selected_stations, selected_coords):
            ax.annotate(f'S{idx}', xy=coord, fontsize=9, fontweight='bold',
                       ha='center', va='center', color='white')

        # 繪製路線
        route = [model.start] + list(selected_coords) + [model.end]
        route_array = np.array(route)
        ax.plot(route_array[:, 0], route_array[:, 1], 'r--', linewidth=2,
               alpha=0.7, label='公車行駛路線', zorder=2)

        # 添加路線箭頭
        for i in range(len(route) - 1):
            start_pt = route_array[i]
            end_pt = route_array[i + 1]
            dx = end_pt[0] - start_pt[0]
            dy = end_pt[1] - start_pt[1]
            ax.arrow(start_pt[0], start_pt[1], dx*0.7, dy*0.7,
                    head_width=2, head_length=1.5, fc='red', ec='red', alpha=0.5)

        # 繪製乘客到最近站點的連線
        distances = np.linalg.norm(
            model.passengers[:, np.newaxis, :] - selected_coords[np.newaxis, :, :],
            axis=2
        )
        nearest_stations = np.argmin(distances, axis=1)

        for i, passenger_pos in enumerate(model.passengers):
            nearest_idx = nearest_stations[i]
            nearest_station = selected_coords[nearest_idx]
            ax.plot([passenger_pos[0], nearest_station[0]],
                   [passenger_pos[1], nearest_station[1]],
                   'b-', alpha=0.1, linewidth=0.5)

        # 設置圖表參數
        ax.set_xlim(-10, 110)
        ax.set_ylim(-10, 110)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X 座標', fontweight='bold', fontsize=11)
        ax.set_ylabel('Y 座標', fontweight='bold', fontsize=11)
        ax.tick_params(labelsize=10)

        # 標題包含成本信息
        ax.set_title(f'{title}\nJ = {result["cost"]:.1f} | 選中 {result["n_stations"]} 個站點',
                    fontweight='bold', fontsize=12, pad=10)

        # 圖例放在圖外
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
                 bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()
    plt.savefig('bus_routing_scenarios.png', dpi=150, bbox_inches='tight')
    print("\n✓ 情境圖已保存至 bus_routing_scenarios.png")
    plt.close()


def run_comparison(model, results):
    """對比三種方法的結果並視覺化"""

    print("\n" + "="*70)
    print("公共運輸路線最佳化系統 - 三方法對比分析")
    print("="*70)

    print(f"\n【問題設定】")
    print(f"  候選站點數：{len(model.candidates)}")
    print(f"  乘客人數：{len(model.passengers)}")
    print(f"  成本權重：w1={model.w1}, w2={model.w2}")

    print(f"\n【目標函數】J = {model.w1} * C_ops + {model.w2} * C_access")

    print("\n" + "-"*70)
    print("【結果對比表】")
    print("-"*70)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['method']}")
        print(f"   選中站點數：{result['n_stations']}")
        print(f"   目標函數值 J：{result['cost']:.2f}")
        print(f"   執行時間：{result['time']:.4f} 秒")
        print(f"   選中站點索引：{result['stations']}")

        if 'seed_cost' in result:
            improvement = (result['seed_cost'] - result['cost']) / result['seed_cost'] * 100
            print(f"   熱啟動改進：{improvement:.2f}%（從 {result['seed_cost']:.2f} -> {result['cost']:.2f}）")

    print("\n" + "="*70)

    # 視覺化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('公共運輸路線最佳化 - 三方法對比', fontsize=14, fontweight='bold')

    # 1. 成本對比
    ax = axes[0, 0]
    methods = [r['method'].split('(')[0].strip() for r in results]
    costs = [r['cost'] for r in results]
    colors = ['skyblue', 'orange', 'lightgreen']
    bars = ax.bar(methods, costs, color=colors, alpha=0.7, edgecolor='black')

    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{cost:.1f}',
               ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('目標函數值 J', fontweight='bold')
    ax.set_title('成本對比', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 2. 站點數對比
    ax = axes[0, 1]
    stations = [r['n_stations'] for r in results]
    bars = ax.bar(methods, stations, color=colors, alpha=0.7, edgecolor='black')

    for bar, station in zip(bars, stations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(station)}',
               ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('選中站點數', fontweight='bold')
    ax.set_title('站點選擇', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 3. 執行時間對比
    ax = axes[1, 0]
    times = [r['time'] for r in results]
    bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{t:.4f}s',
               ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('執行時間（秒）', fontweight='bold')
    ax.set_title('性能對比', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 4. GA 收斂曲線
    ax = axes[1, 1]

    if 'cost_history' in results[1]:
        ga_history = results[1]['cost_history']
        ax.plot(range(len(ga_history)), ga_history, 'o-', label='GA',
               color='orange', linewidth=2, markersize=4)

    if 'cost_history' in results[2]:
        hybrid_history = results[2]['cost_history']
        ax.plot(range(len(hybrid_history)), hybrid_history, 's-', label='Hybrid',
               color='green', linewidth=2, markersize=4)

    ax.set_xlabel('演化代數', fontweight='bold')
    ax.set_ylabel('最佳目標函數值', fontweight='bold')
    ax.set_title('收斂過程對比', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bus_routing_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ 結果已保存至 bus_routing_results.png")
    plt.close()


# ============================================================================
# 主程式執行
# ============================================================================

if __name__ == '__main__':
    print("\n【公共運輸路線最佳化系統 - 快速實現】")
    print("對應報告三部分：模型定義 -> 演算法評估 -> 系統建議\n")

    # 第一部分：資料準備與模型初始化
    print("【第一部分】資料準備與模型初始化...")

    candidates, passengers, start, end = generate_sample_data(
        n_candidates=15,
        n_passengers=30
    )

    model = LRPModel(
        candidates=candidates,
        passengers=passengers,
        start=start,
        end=end,
        weights=(0.6, 0.4)
    )

    print(f"✓ 候選站點：{len(candidates)} 個")
    print(f"✓ 乘客分佈：{len(passengers)} 人（3 個聚類）")
    print(f"✓ 目標函數：J = 0.6*C_ops + 0.4*C_access\n")

    # 第二部分：執行三種方法
    print("【第二部分】演算法實現與對比...")
    print("-" * 70)

    results = []

    print("\n執行方法一：順序求解法 (K-Means + 貪心)...", end=' ')
    seq_method = SequentialMethod(model, n_stations=5)
    seq_result = seq_method.solve()
    results.append(seq_result)
    print(f"✓ (J = {seq_result['cost']:.2f})")

    print("執行方法二：純遺傳演算法...", end=' ')
    ga_method = GeneticMethod(model, n_stations=5, generations=50, pop_size=30)
    ga_result = ga_method.solve()
    results.append(ga_result)
    print(f"✓ (J = {ga_result['cost']:.2f})")

    print("執行方法三：混合策略 (熱啟動 GA)...", end=' ')
    hybrid_method = HybridMethod(model, n_stations=5, generations=50, pop_size=30)
    hybrid_result = hybrid_method.solve()
    results.append(hybrid_result)
    print(f"✓ (J = {hybrid_result['cost']:.2f})")

    # 第三部分：實驗評估與視覺化
    print("\n【第三部分】結果分析與視覺化...")
    print("-" * 70)

    visualize_problem_scenario(model, results)
    run_comparison(model, results)

    print("\n✓ 實現完成！")
    print("  報告中的三層論述已對應實現：")
    print("  • 第一部分 (數學定義)：LRPModel 類別")
    print("  • 第二部分 (演算法評估)：三種方法的對比")
    print("  • 第三部分 (系統建議)：混合策略的效果驗證")