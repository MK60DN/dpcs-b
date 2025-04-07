"""
交互式库存管理系统模拟 - 允许用户设置参数并运行模拟
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpcs.modules.inventory_system import InventorySystem
from dpcs.modules.inventory_control import InventoryControl
from dpcs.modules.forecast_system import InventorySystemWithForecast
from dpcs.modules.cost_tracking import CostTrackingAgent


class InventorySimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("库存管理系统模拟")
        self.root.geometry("1200x800")

        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建参数设置框架
        self.create_parameter_frame()

        # 创建控制按钮框架
        self.create_control_frame()

        # 创建图表框架
        self.create_chart_frame()

        # 创建结果显示框架
        self.create_results_frame()

        # 初始化模拟系统
        self.inventory_system = None
        self.control_system = None
        self.forecast_system = None
        self.cost_agent = None

        # 初始化数据存储
        self.periods = []
        self.demands = []
        self.inventory_levels = []
        self.order_quantities = []
        self.costs = []
        self.service_levels = []

        # 初始化图表
        self.setup_charts()

    def create_parameter_frame(self):
        """创建参数设置框架"""
        param_frame = ttk.LabelFrame(self.main_frame, text="模拟参数", padding="10")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # 初始库存
        ttk.Label(param_frame, text="初始库存:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.initial_inventory_var = tk.IntVar(value=100)
        ttk.Entry(param_frame, textvariable=self.initial_inventory_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        # 初始在途库存
        ttk.Label(param_frame, text="初始在途库存:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.initial_intransit_var = tk.IntVar(value=50)
        ttk.Entry(param_frame, textvariable=self.initial_intransit_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        # 补货策略
        ttk.Label(param_frame, text="补货策略:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.strategy_var = tk.StringVar(value="RS")
        strategy_combo = ttk.Combobox(param_frame, textvariable=self.strategy_var, width=10)
        strategy_combo['values'] = ('RS', 'QR')
        strategy_combo.grid(row=2, column=1, padx=5, pady=5)

        # RS策略参数
        rs_frame = ttk.LabelFrame(param_frame, text="RS策略参数", padding="5")
        rs_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        ttk.Label(rs_frame, text="订货点(R):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.reorder_point_var = tk.IntVar(value=50)
        ttk.Entry(rs_frame, textvariable=self.reorder_point_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(rs_frame, text="最大库存(S):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.max_inventory_var = tk.IntVar(value=200)
        ttk.Entry(rs_frame, textvariable=self.max_inventory_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # QR策略参数
        qr_frame = ttk.LabelFrame(param_frame, text="QR策略参数", padding="5")
        qr_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        ttk.Label(qr_frame, text="订货点(R):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.reorder_point_q_var = tk.IntVar(value=60)
        ttk.Entry(qr_frame, textvariable=self.reorder_point_q_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(qr_frame, text="固定订货量(Q):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.order_quantity_var = tk.IntVar(value=30)
        ttk.Entry(qr_frame, textvariable=self.order_quantity_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # 成本参数
        cost_frame = ttk.LabelFrame(param_frame, text="成本参数", padding="5")
        cost_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        ttk.Label(cost_frame, text="订货成本/单位:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.order_cost_var = tk.DoubleVar(value=10.0)
        ttk.Entry(cost_frame, textvariable=self.order_cost_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(cost_frame, text="持有成本/单位:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.holding_cost_var = tk.DoubleVar(value=2.0)
        ttk.Entry(cost_frame, textvariable=self.holding_cost_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(cost_frame, text="缺货成本/单位:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.stockout_cost_var = tk.DoubleVar(value=50.0)
        ttk.Entry(cost_frame, textvariable=self.stockout_cost_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # 需求参数
        demand_frame = ttk.LabelFrame(param_frame, text="需求参数", padding="5")
        demand_frame.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        ttk.Label(demand_frame, text="基础需求:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.base_demand_var = tk.IntVar(value=60)
        ttk.Entry(demand_frame, textvariable=self.base_demand_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(demand_frame, text="季节波动幅度:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.seasonality_var = tk.IntVar(value=20)
        ttk.Entry(demand_frame, textvariable=self.seasonality_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(demand_frame, text="随机波动范围:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.noise_var = tk.IntVar(value=10)
        ttk.Entry(demand_frame, textvariable=self.noise_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # 模拟周期
        ttk.Label(param_frame, text="模拟周期数:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.num_periods_var = tk.IntVar(value=20)
        ttk.Entry(param_frame, textvariable=self.num_periods_var, width=10).grid(row=7, column=1, padx=5, pady=5)

    def create_control_frame(self):
        """创建控制按钮框架"""
        control_frame = ttk.Frame(self.main_frame, padding="10")
        control_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # 开始模拟按钮
        ttk.Button(control_frame, text="开始模拟", command=self.start_simulation).grid(row=0, column=0, padx=5, pady=5)

        # 重置按钮
        ttk.Button(control_frame, text="重置", command=self.reset_simulation).grid(row=0, column=1, padx=5, pady=5)

        # 退出按钮
        ttk.Button(control_frame, text="退出", command=self.root.quit).grid(row=0, column=2, padx=5, pady=5)

    def create_chart_frame(self):
        """创建图表框架"""
        self.chart_frame = ttk.LabelFrame(self.main_frame, text="模拟结果图表", padding="10")
        self.chart_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

    def create_results_frame(self):
        """创建结果显示框架"""
        self.results_frame = ttk.LabelFrame(self.main_frame, text="模拟结果统计", padding="10")
        self.results_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # 创建表格显示结果
        columns = ('period', 'demand', 'inventory', 'order', 'cost', 'service_level')
        self.results_tree = ttk.Treeview(self.results_frame, columns=columns, show='headings')

        # 定义列标题
        self.results_tree.heading('period', text='周期')
        self.results_tree.heading('demand', text='需求')
        self.results_tree.heading('inventory', text='库存')
        self.results_tree.heading('order', text='订货量')
        self.results_tree.heading('cost', text='成本')
        self.results_tree.heading('service_level', text='服务水平')

        # 定义列宽度
        for col in columns:
            self.results_tree.column(col, width=100, anchor='center')

        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        # 放置表格和滚动条
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # 配置results_frame的行列权重
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)

        # 创建摘要标签
        self.summary_frame = ttk.Frame(self.results_frame, padding="5")
        self.summary_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.total_demand_var = tk.StringVar(value="总需求: 0")
        self.total_cost_var = tk.StringVar(value="总成本: 0.00")
        self.avg_service_level_var = tk.StringVar(value="平均服务水平: 0.00")
        self.final_inventory_var = tk.StringVar(value="最终库存: 0")

        ttk.Label(self.summary_frame, textvariable=self.total_demand_var).grid(row=0, column=0, padx=10, pady=2,
                                                                               sticky="w")
        ttk.Label(self.summary_frame, textvariable=self.total_cost_var).grid(row=0, column=1, padx=10, pady=2,
                                                                             sticky="w")
        ttk.Label(self.summary_frame, textvariable=self.avg_service_level_var).grid(row=1, column=0, padx=10, pady=2,
                                                                                    sticky="w")
        ttk.Label(self.summary_frame, textvariable=self.final_inventory_var).grid(row=1, column=1, padx=10, pady=2,
                                                                                  sticky="w")

    def setup_charts(self):
        """设置图表"""
        # 创建图表
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)

        # 添加子图
        self.ax1 = self.fig.add_subplot(311)  # 库存和需求
        self.ax2 = self.fig.add_subplot(312)  # 服务水平
        self.ax3 = self.fig.add_subplot(313)  # 成本

        # 初始化图表
        self.inventory_line, = self.ax1.plot([], [], 'b-', marker='o', label='库存')
        self.demand_line, = self.ax1.plot([], [], 'r--', marker='x', label='需求')
        self.order_line, = self.ax1.plot([], [], 'g-', marker='^', label='订货量')
        self.ax1.set_title('库存和需求')
        self.ax1.set_xlabel('周期')
        self.ax1.set_ylabel('数量')
        self.ax1.legend()
        self.ax1.grid(True)

        self.service_line, = self.ax2.plot([], [], 'purple', marker='d')
        self.ax2.set_title('服务水平')
        self.ax2.set_xlabel('周期')
        self.ax2.set_ylabel('服务水平')
        self.ax2.set_ylim(0, 1.1)
        self.ax2.grid(True)

        self.cost_line, = self.ax3.plot([], [], 'orange', marker='*')
        self.ax3.set_title('每期成本')
        self.ax3.set_xlabel('周期')
        self.ax3.set_ylabel('成本')
        self.ax3.grid(True)

        # 添加图表到GUI
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # 添加工具栏
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update_charts(self):
        """更新图表数据"""
        # 更新图表数据
        self.inventory_line.set_data(self.periods, self.inventory_levels)
        self.demand_line.set_data(self.periods, self.demands)
        self.order_line.set_data(self.periods, self.order_quantities)
        self.service_line.set_data(self.periods, self.service_levels)
        self.cost_line.set_data(self.periods, self.costs)

        # 调整轴范围
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()

        # 重绘图表
        self.fig.tight_layout()
        self.canvas.draw()

    def start_simulation(self):
        """开始模拟"""
        try:
            # 获取参数
            initial_inventory = self.initial_inventory_var.get()
            initial_intransit = self.initial_intransit_var.get()
            strategy = self.strategy_var.get()
            reorder_point = self.reorder_point_var.get()
            max_inventory = self.max_inventory_var.get()
            reorder_point_q = self.reorder_point_q_var.get()
            order_quantity = self.order_quantity_var.get()
            order_cost = self.order_cost_var.get()
            holding_cost = self.holding_cost_var.get()
            stockout_cost = self.stockout_cost_var.get()
            base_demand = self.base_demand_var.get()
            seasonality = self.seasonality_var.get()
            noise = self.noise_var.get()
            num_periods = self.num_periods_var.get()

            # 初始化系统
            self.inventory_system = InventorySystem()
            self.inventory_system.inventory = initial_inventory
            self.inventory_system.inTransit = initial_intransit

            self.control_system = InventoryControl(strategy=strategy)
            self.control_system.inventory = initial_inventory
            self.control_system.inTransit = initial_intransit
            self.control_system.reorderPoint = reorder_point
            self.control_system.maxInventory = max_inventory
            self.control_system.reorderPointQ = reorder_point_q
            self.control_system.orderQuantity = order_quantity

            self.forecast_system = InventorySystemWithForecast()
            self.forecast_system.inventory = initial_inventory
            self.forecast_system.inTransit = initial_intransit

            self.cost_agent = CostTrackingAgent(
                name="模拟中心",
                order_cost_per_unit=order_cost,
                holding_cost_per_unit=holding_cost,
                stockout_cost_per_unit=stockout_cost
            )

            # 清空数据
            self.reset_simulation(clear_gui=False)

            # 清空结果表格
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            # 运行模拟
            for period in range(1, num_periods + 1):
                # 生成需求
                seasonality_effect = seasonality * np.sin(period * np.pi / 10)
                noise_effect = np.random.randint(-noise, noise)
                demand = max(10, int(base_demand + seasonality_effect + noise_effect))

                # 记录数据
                self.periods.append(period)
                self.demands.append(demand)

                # 处理需求
                self.inventory_system.addOrder(demand)
                self.inventory_system.checkDemand()

                # 更新控制系统
                self.control_system.inventory = self.inventory_system.inventory
                self.control_system.controlInventory()

                # 计算订货量
                order_qty = 0
                if strategy == "RS":
                    if self.inventory_system.inventory < reorder_point:
                        order_qty = max_inventory - self.inventory_system.inventory
                else:  # QR策略
                    if self.inventory_system.inventory < reorder_point_q:
                        order_qty = order_quantity

                # 记录订货量
                self.order_quantities.append(order_qty)

                # 计算缺货量
                shortage_qty = self.inventory_system.backOffAmount

                # 更新成本
                self.cost_agent.updateTotalCost(order_qty, shortage_qty)
                self.costs.append(self.cost_agent.costHistory[-1])

                # 更新库存
                self.inventory_system.inventory += order_qty
                self.inventory_levels.append(self.inventory_system.inventory)

                # 更新服务水平
                self.service_levels.append(self.inventory_system.serviceLevel)

                # 更新结果表格
                self.results_tree.insert('', 'end', values=(
                    period,
                    demand,
                    self.inventory_system.inventory,
                    order_qty,
                    f"{self.cost_agent.costHistory[-1]:.2f}",
                    f"{self.inventory_system.serviceLevel:.2f}"
                ))

                # 更新预测系统
                self.forecast_system.updateOrderHistory(demand)

            # 更新图表
            self.update_charts()

            # 更新摘要统计
            self.total_demand_var.set(f"总需求: {sum(self.demands)}")
            self.total_cost_var.set(f"总成本: {sum(self.costs):.2f}")
            self.avg_service_level_var.set(f"平均服务水平: {np.mean(self.service_levels):.2f}")
            self.final_inventory_var.set(f"最终库存: {self.inventory_levels[-1]}")

            messagebox.showinfo("模拟完成", f"成功模拟了{num_periods}个周期的库存管理过程!")

        except Exception as e:
            messagebox.showerror("错误", f"模拟过程中发生错误: {str(e)}")
            raise

    def reset_simulation(self, clear_gui=True):
        """重置模拟数据"""
        # 清空数据
        self.periods = []
        self.demands = []
        self.inventory_levels = []
        self.order_quantities = []
        self.costs = []
        self.service_levels = []

        if clear_gui:
            # 清空结果表格
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            # 重置摘要统计
            self.total_demand_var.set("总需求: 0")
            self.total_cost_var.set("总成本: 0.00")
            self.avg_service_level_var.set("平均服务水平: 0.00")
            self.final_inventory_var.set("最终库存: 0")

            # 更新图表
            self.update_charts()

            messagebox.showinfo("重置", "模拟数据已重置!")


def main():
    root = tk.Tk()
    app = InventorySimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()