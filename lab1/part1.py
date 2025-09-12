# -*- coding: utf-8 -*-
"""
Решение задачи линейного программирования для Варианта 15 с визуализацией.
Цель: Максимизация выпуска деталей при ограничениях на ресурсы станков.
Данные соответствуют Excel:
Токарный: [2, 0, 3, 4, 1]
Фрезерный: [1, 2, 3, 2, 1]
Строгальный: [1, 1, 1, 0, 2]
Шлифовальный: [3, 2, 0, 1, 1]
"""

# Импорт необходимых библиотек
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# 1. РЕШЕНИЕ ЗАДАЧИ ЛП

# Коэффициенты целевой функции (для минимизации: -1 * сумму x_i)
c = [-1, -1, -1, -1, -1]

# Матрица коэффициентов ограничений "меньше или равно" (A_ub * x <= b_ub)
# Обновлено согласно данным из Excel
A_ub = [
    [2, 0, 3, 4, 1],  # Токарный
    [1, 2, 3, 2, 1],  # Фрезерный
    [1, 1, 1, 0, 2],  # Строгальный
    [3, 2, 0, 1, 1]   # Шлифовальный
]

b_ub = [4100, 2000, 5800, 10800] # Ресурсы времени

# Границы переменных (x_i >= 0)
bounds = [(0, None)] * 5

# Решение задачи
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Извлечение результатов
optimal_quantities = result.x
total_output = -result.fun  # Преобразуем обратно к максимуму

# 2. СОЗДАНИЕ ГРАФИКОВ

# Настройка стиля и размера графиков
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Анализ оптимального плана производства', fontsize=16, fontweight='bold')

# --- ГРАФИК 1: Оптимальный план по технологиям ---
technologies = ['Техн. 1', 'Техн. 2', 'Техн. 3', 'Техн. 4', 'Техн. 5']
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#9b59b6', '#3498db']
bars = ax1.bar(technologies, optimal_quantities, color=colors, edgecolor='black', alpha=0.8)

# Добавление значений на столбцы
for bar, value in zip(bars, optimal_quantities):
    height = bar.get_height()
    if value > 0:  # Подписываем только ненулевые значения
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')

ax1.set_title('Оптимальное количество деталей по технологиям')
ax1.set_ylabel('Количество деталей, шт.')
ax1.set_xlabel('Технологии обработки')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_axisbelow(True)

# --- ГРАФИК 2: Использование ресурсов станков ---
machine_types = ['Токарный', 'Фрезерный', 'Строгальный', 'Шлифовальный']
resource_available = b_ub
# Рассчитываем фактически использованное время для каждого станка
resource_used = [
    2*optimal_quantities[0] + 0*optimal_quantities[1] + 3*optimal_quantities[2] + 4*optimal_quantities[3] + 1*optimal_quantities[4], # Токарный
    1*optimal_quantities[0] + 2*optimal_quantities[1] + 3*optimal_quantities[2] + 2*optimal_quantities[3] + 1*optimal_quantities[4], # Фрезерный
    1*optimal_quantities[0] + 1*optimal_quantities[1] + 1*optimal_quantities[2] + 0*optimal_quantities[3] + 2*optimal_quantities[4], # Строгальный
    3*optimal_quantities[0] + 2*optimal_quantities[1] + 0*optimal_quantities[2] + 1*optimal_quantities[3] + 1*optimal_quantities[4]  # Шлифовальный
]

x_pos = np.arange(len(machine_types))
bar_width = 0.35

bars1 = ax2.bar(x_pos - bar_width/2, resource_available, bar_width, label='Доступный ресурс', color='#2ecc71', alpha=0.7)
bars2 = ax2.bar(x_pos + bar_width/2, resource_used, bar_width, label='Использованный ресурс', color='#e74c3c', alpha=0.7)

# Добавление значений на столбцы
for i, (avail, used) in enumerate(zip(resource_available, resource_used)):
    ax2.text(i - bar_width/2, avail + 200, f'{int(avail)}', ha='center', va='bottom', fontsize=9)
    ax2.text(i + bar_width/2, used + 200, f'{int(used)}', ha='center', va='bottom', fontsize=9)

ax2.set_title('Использование ресурсов станков')
ax2.set_ylabel('Время, мин.')
ax2.set_xlabel('Тип станка')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(machine_types)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_axisbelow(True)

# Добавляем общий вывод на figure
output_text = f'ВЫВОД: Максимальный выпуск составляет {int(total_output)} деталей.\n'
for i, tech in enumerate(technologies):
    if optimal_quantities[i] > 0:
        output_text += f'Для этого необходимо производить {int(optimal_quantities[i])} дет. по технологии {i+1}\n'

plt.figtext(0.02, 0.02, output_text,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            fontsize=11, fontweight='bold')

# Adjust layout and display
plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Оставляем место для текста внизу
plt.show()

# 3. ВЫВОД РЕЗУЛЬТАТОВ В КОНСОЛЬ
print("="*50)
print("ОТЧЕТ ПО РЕШЕНИЮ ЗАДАЧИ")
print("="*50)
print(f"Статус решения: {result.message}")
print("\nОптимальное количество деталей по технологиям:")
for i, tech in enumerate(technologies, 1):
    print(f"x{i} ({tech}): {optimal_quantities[i-1]:.1f} шт.")
print(f"\nМаксимальный суммарный выпуск: {total_output:.1f} деталей.")

print("\nПроверка использования ресурсов:")
for i, machine in enumerate(machine_types):
    print(f"  {machine}: {resource_used[i]:.0f} / {resource_available[i]} мин. ({(resource_used[i]/resource_available[i])*100:.1f}%)")

print("\nСравнение с решением из Excel:")
print("Из Excel: T1=0, T2=0, T3=0, T4=0, T5=2000, всего=2000")
print("Из Python:", f"T1={optimal_quantities[0]:.0f}, T2={optimal_quantities[1]:.0f}, T3={optimal_quantities[2]:.0f}, T4={optimal_quantities[3]:.0f}, T5={optimal_quantities[4]:.0f}, всего={total_output:.0f}")
print("="*50)