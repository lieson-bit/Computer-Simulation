import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Исходные данные
t = np.array([1, 2, 3, 4, 5, 6, 7])
Y = np.array([138, 127, 143, 142, 145, 143, 146])
n = len(t)

print("АНАЛИЗ ПРОИЗВОДСТВА СТАЛИ - ВАРИАНТ 15")
print("=" * 50)

# Функция для расчета скорректированного R-квадрат
def adjusted_r2(r2, n, p):
    if n - p - 1 <= 0:
        return -np.inf
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# 1. Ручная реализация МНК для квадратичного полинома f1(x)
print("\n1. КВАДРАТИЧНАЯ МОДЕЛЬ f1(t) = a2*t² + a1*t + a0")
X_manual = np.column_stack([t**2, t, np.ones(len(t))])
coefficients = np.linalg.inv(X_manual.T @ X_manual) @ X_manual.T @ Y
a2, a1, a0 = coefficients

print(f"Коэффициенты: a2 = {a2:.4f}, a1 = {a1:.4f}, a0 = {a0:.4f}")
print(f"Модель: f1(t) = {a2:.4f}t² + {a1:.4f}t + {a0:.4f}")

Y_pred_f1 = a2*t**2 + a1*t + a0
SSE_f1 = np.sum((Y - Y_pred_f1)**2)
r2_f1 = r2_score(Y, Y_pred_f1)
adj_r2_f1 = adjusted_r2(r2_f1, n, p=2)

print(f"SSE для f1: {SSE_f1:.4f}")
print(f"R² для f1: {r2_f1:.4f}")
print(f"Скорректированный R² для f1: {adj_r2_f1:.4f}")

# 2. Линейная модель
print("\n2. ЛИНЕЙНАЯ МОДЕЛЬ f2(t) = a*t + b")
sum_x = np.sum(t)
sum_y = np.sum(Y)
sum_xy = np.sum(t * Y)
sum_x2 = np.sum(t**2)

a_linear = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
b_linear = (sum_y - a_linear * sum_x) / n

print(f"Коэффициенты: a = {a_linear:.4f}, b = {b_linear:.4f}")
print(f"Модель: f2(t) = {a_linear:.4f}t + {b_linear:.4f}")

Y_pred_f2 = a_linear * t + b_linear
SSE_f2 = np.sum((Y - Y_pred_f2)**2)
r2_f2 = r2_score(Y, Y_pred_f2)
adj_r2_f2 = adjusted_r2(r2_f2, n, p=1)

print(f"SSE для f2: {SSE_f2:.4f}")
print(f"R² для f2: {r2_f2:.4f}")
print(f"Скорректированный R² для f2: {adj_r2_f2:.4f}")

# 3. Функциональная модель f3(t) = a * (∛(t+1) + 1) + b
print("\n3. ФУНКЦИОНАЛЬНАЯ МОДЕЛЬ f3(t) = a * (∛(t+1) + 1) + b")
def f3_base(t):
    return np.cbrt(t + 1) + 1

X_f3 = np.column_stack([f3_base(t), np.ones(len(t))])
coeff_f3 = np.linalg.inv(X_f3.T @ X_f3) @ X_f3.T @ Y
a_scale, b_shift = coeff_f3

def f3_model(t):
    return a_scale * f3_base(t) + b_shift

Y_pred_f3 = f3_model(t)
SSE_f3 = np.sum((Y - Y_pred_f3)**2)
r2_f3 = r2_score(Y, Y_pred_f3)
adj_r2_f3 = adjusted_r2(r2_f3, n, p=1)

print(f"Модель: f3(t) = {a_scale:.4f} * (∛(t+1) + 1) + {b_shift:.4f}")
print(f"SSE для f3: {SSE_f3:.4f}")
print(f"R² для f3: {r2_f3:.4f}")
print(f"Скорректированный R² для f3: {adj_r2_f3:.4f}")

# 4. Сравнение моделей с использованием SSE и скорректированного R²
print("\n4. СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 50)

models = {
    'Квадратичная (f1)': {'sse': SSE_f1, 'adj_r2': adj_r2_f1, 'predictions': Y_pred_f1, 'p': 2},
    'Линейная (f2)': {'sse': SSE_f2, 'adj_r2': adj_r2_f2, 'predictions': Y_pred_f2, 'p': 1},
    'Функциональная (f3)': {'sse': SSE_f3, 'adj_r2': adj_r2_f3, 'predictions': Y_pred_f3, 'p': 1}
}

print("Таблица сравнения:")
print(f"{'Модель':<20} {'SSE':<12} {'R²':<10} {'Скорр. R²':<12}")
print("-" * 55)
for model_name, model_data in models.items():
    r2 = r2_score(Y, model_data['predictions'])
    print(f"{model_name:<20} {model_data['sse']:<12.4f} {r2:<10.4f} {model_data['adj_r2']:<12.4f}")

# Найти лучшую модель по SSE (минимум)
best_model_sse = min(models.keys(), key=lambda x: models[x]['sse'])
best_sse = models[best_model_sse]['sse']

# Найти лучшую модель по скорректированному R² (максимум)
valid_models = {name: data for name, data in models.items() if not np.isinf(data['adj_r2'])}
if valid_models:
    best_model_adj_r2 = max(valid_models.keys(), key=lambda x: valid_models[x]['adj_r2'])
    best_adj_r2 = valid_models[best_model_adj_r2]['adj_r2']
else:
    best_model_adj_r2 = 'Линейная (f2)'
    best_adj_r2 = models['Линейная (f2)']['adj_r2']

print(f"\nЛучшая модель по SSE: {best_model_sse} (SSE = {best_sse:.4f})")
print(f"Лучшая модель по скорр. R²: {best_model_adj_r2} (Скорр. R² = {best_adj_r2:.4f})")

# Финальный выбор модели
if best_model_sse == best_model_adj_r2:
    final_model = best_model_sse
    print(f"\n✓ Критерии согласованы: {final_model} - лучшая модель")
else:
    # Если критерии не согласованы, предпочитаем скорректированный R²
    final_model = best_model_adj_r2
    print(f"\n⚠ Критерии не согласованы. Предпочтение скорректированному R² (учитывает сложность модели): {final_model} выбрана")

# 5. Прогноз с использованием выбранной модели
print("\n5. АНАЛИЗ ПРОГНОЗА")
t_forecast = 8

if final_model == 'Квадратичная (f1)':
    forecast = a2*t_forecast**2 + a1*t_forecast + a0
    model_predictions = Y_pred_f1
elif final_model == 'Линейная (f2)':
    forecast = a_linear * t_forecast + b_linear
    model_predictions = Y_pred_f2
else:  # Функциональная
    forecast = f3_model(t_forecast)
    model_predictions = Y_pred_f3

# Расчет стандартной ошибки
residuals = Y - model_predictions
std_error = np.std(residuals)

print(f"Выбранная модель: {final_model}")
print(f"Прогноз на месяц {t_forecast}: {forecast:.2f} млн. тонн")
print(f"Стандартная ошибка: {std_error:.2f}")
print(f"95% интервал прогноза: {forecast:.2f} ± {2*std_error:.2f}")

# 6. Визуализация
print("\n6. ПОСТРОЕНИЕ ГРАФИКОВ...")
plt.figure(figsize=(12, 8))

# Исходные данные
plt.scatter(t, Y, color='black', s=100, zorder=5, label='Исходные данные')

# Сглаженные кривые
t_smooth = np.linspace(0.8, 8.2, 100)

# Построение всех моделей
Y_f1_smooth = a2*t_smooth**2 + a1*t_smooth + a0
plt.plot(t_smooth, Y_f1_smooth, 'r-', linewidth=2, 
         label=f'Квадратичная (SSE={SSE_f1:.1f}, Скорр. R²={adj_r2_f1:.3f})')

Y_f2_smooth = a_linear * t_smooth + b_linear
plt.plot(t_smooth, Y_f2_smooth, 'g-', linewidth=2, 
         label=f'Линейная (SSE={SSE_f2:.1f}, Скорр. R²={adj_r2_f2:.3f})')

Y_f3_smooth = f3_model(t_smooth)
plt.plot(t_smooth, Y_f3_smooth, 'b-', linewidth=2, 
         label=f'Функциональная (SSE={SSE_f3:.1f}, Скорр. R²={adj_r2_f3:.3f})')

# Выделение выбранной модели
if final_model == 'Квадратичная (f1)':
    plt.plot(t_smooth, Y_f1_smooth, 'r-', linewidth=4, alpha=0.3)
elif final_model == 'Линейная (f2)':
    plt.plot(t_smooth, Y_f2_smooth, 'g-', linewidth=4, alpha=0.3)
else:
    plt.plot(t_smooth, Y_f3_smooth, 'b-', linewidth=4, alpha=0.3)

# Прогноз
plt.axvline(x=t_forecast, color='gray', linestyle='--', alpha=0.7)
plt.scatter([t_forecast], [forecast], color='red', s=150, zorder=5, 
           label=f'Прогноз: {forecast:.1f} ± {2*std_error:.1f}')

plt.xlabel('Время (месяцы)')
plt.ylabel('Производство стали (млн. тонн)')
plt.title(f'Динамика производства стали - Вариант 15\nЛучшая модель: {final_model}', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim(0.8, 8.2)
plt.tight_layout()
plt.show()

# Финальное резюме
print("\n" + "="*60)
print("ФИНАЛЬНЫЙ АНАЛИЗ")
print("="*60)
print("Данные: Производство стали (млн. тонн), месяцы 1-7, 2017")
print(f"Выбранная модель: {final_model}")
print(f"Уравнение модели: f(t) = {a_linear:.4f}t + {b_linear:.4f}")
print(f"Критерии выбора:")
print(f"  - SSE: {models[final_model]['sse']:.4f} (меньше - лучше)")
print(f"  - Скорректированный R²: {models[final_model]['adj_r2']:.4f} (больше - лучше)")
print(f"Прогноз на месяц 8: {forecast:.2f} ± {2*std_error:.2f} млн. тонн")

# Интерпретация модели
print("\nИнтерпретация:")
if models[final_model]['adj_r2'] > 0.7:
    print("  ✓ Отличное соответствие модели")
elif models[final_model]['adj_r2'] > 0.5:
    print("  ○ Хорошее соответствие модели")
elif models[final_model]['adj_r2'] > 0.3:
    print("  ○ Умеренное соответствие модели")
elif models[final_model]['adj_r2'] > 0:
    print("  ⚠ Слабое соответствие модели")
else:
    print("  ❌ Плохое соответствие модели - хуже простого среднего")

print("="*60)