#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа №1
"Принципы построения моделирующих алгоритмов: Δt и δz"
Вариант 19 (функция из варианта 3 по таблице)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import deque

# ============================================================
# ЧАСТЬ 1: Численное дифференцирование (принцип Δt)
# X(t) = 4·tg²(t)
# ============================================================

dt = 0.01
K = 1 / dt
N_steps = 50
t_vals = np.arange(0, N_steps * dt + dt/2, dt)

def X_func(t):
    return 4 * np.tan(t)**2

def X_analytical_derivative(t):
    return 8 * np.tan(t) / (np.cos(t)**2)

Z = np.zeros_like(t_vals)
X = X_func(t_vals)
X_imit = np.zeros_like(t_vals)

Z[0] = X[0]
X_imit[0] = 0

for i in range(1, len(t_vals)):
    Z[i] = Z[i-1] + K * (X[i-1] - Z[i-1]) * dt
    X_imit[i-1] = K * (X[i-1] - Z[i-1])

X_imit[-1] = K * (X[-1] - Z[-1])
X_anal = X_analytical_derivative(t_vals)
rel_error = np.abs(X_imit - X_anal) / np.abs(X_anal + 1e-15)
rel_error[0] = np.nan

df_part1 = pd.DataFrame({
    't': np.round(t_vals, 2),
    'X(t)': np.round(X, 5),
    'X(t+dt)': np.round(np.concatenate([X[1:], [X[-1]]]), 5),
    "X'_имит": np.round(X_imit, 5),
    "X'_аналит": np.round(X_anal, 5),
    'Ошибка (отн.)': np.round(rel_error, 7)
})

print("=" * 70)
print("ЧАСТЬ 1: Численное дифференцирование")
print("X(t) = 4·tg²(t), Δt = 0.01, t ∈ [0, 0.50]")
print("=" * 70)
print(df_part1.to_string(index=False))
print(f"\nСредняя отн. ошибка: {np.nanmean(rel_error):.6f}")

# Графики части 1
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1 = axes[0]
ax1.plot(t_vals, X, 'b-', linewidth=2, label='X(t)')
ax1.plot(t_vals, X_imit, 'r--', linewidth=1.5, label="X'(t) имитац.")
ax1.plot(t_vals, X_anal, 'g:', linewidth=2, label="X'(t) аналит.")
ax1.set_xlabel('t', fontsize=12)
ax1.set_ylabel('Значение', fontsize=12)
ax1.set_title('Функция и производные (Δt = 0.01)')
ax1.legend()
ax1.grid(True)

ax2 = axes[1]
ax2.plot(t_vals[1:], rel_error[1:], 'm-', linewidth=1.5)
ax2.set_xlabel('t', fontsize=12)
ax2.set_ylabel('Относительная ошибка')
ax2.set_title('Погрешность имитационного метода')
ax2.grid(True)
plt.tight_layout()
plt.savefig('part1_derivative.png', dpi=150)
plt.show()


# ============================================================
# ЧАСТЬ 2: Имитационное моделирование СМО (принцип δz)
# ============================================================

def run_smo_simulation(seed=None):
    if seed is not None:
        random.seed(seed)

    ARRIVAL_MIN, ARRIVAL_MAX = 1, 7
    SERVICE_MIN, SERVICE_MAX = 1, 16
    SWITCH_TIME = 70
    N_JOBS = 1000

    comp1_free = 0.0
    comp2_available = False
    comp2_free = 0.0

    queue = deque()
    total_wait_time = 0.0
    total_system_time = 0.0
    max_queue_length = 0
    queue_length_sum = 0.0

    comp1_idle_time = 0.0
    comp2_idle_time = 0.0

    served_no_queue = 0
    total_served = 0
    total_arrivals = 0

    next_arrival = random.uniform(ARRIVAL_MIN, ARRIVAL_MAX)
    next_departure1 = float('inf')
    next_departure2 = float('inf')

    comp1_busy = False
    comp2_busy = False

    last_event_time = 0.0
    last_queue_length = 0

    while total_served < N_JOBS:
        events = [(next_arrival, 'arrival')]
        if comp1_busy:
            events.append((next_departure1, 'departure1'))
        if comp2_busy and comp2_available:
            events.append((next_departure2, 'departure2'))

        next_time, event_type = min(events, key=lambda x: x[0])

        time_diff = next_time - last_event_time
        queue_length_sum += last_queue_length * time_diff
        last_event_time = next_time
        clock = next_time

        if clock >= SWITCH_TIME:
            comp2_available = True

        if event_type == 'arrival':
            total_arrivals += 1
            arrival_time = clock
            assigned = False

            if not comp1_busy:
                service_time = random.uniform(SERVICE_MIN, SERVICE_MAX)
                next_departure1 = clock + service_time
                comp1_busy = True
                if clock > comp1_free:
                    comp1_idle_time += clock - comp1_free
                comp1_free = next_departure1
                total_system_time += service_time
                served_no_queue += 1
                assigned = True
            elif comp2_available and not comp2_busy:
                service_time = random.uniform(SERVICE_MIN, SERVICE_MAX)
                next_departure2 = clock + service_time
                comp2_busy = True
                if clock > comp2_free:
                    comp2_idle_time += clock - comp2_free
                comp2_free = next_departure2
                total_system_time += service_time
                served_no_queue += 1
                assigned = True

            if not assigned:
                queue.append(arrival_time)
                last_queue_length = len(queue)
                if len(queue) > max_queue_length:
                    max_queue_length = len(queue)

            next_arrival = clock + random.uniform(ARRIVAL_MIN, ARRIVAL_MAX)

        elif event_type == 'departure1':
            total_served += 1
            comp1_busy = False
            next_departure1 = float('inf')

            if queue:
                wait_time = clock - queue.popleft()
                total_wait_time += wait_time
                last_queue_length = len(queue)
                service_time = random.uniform(SERVICE_MIN, SERVICE_MAX)
                next_departure1 = clock + service_time
                comp1_busy = True
                comp1_free = next_departure1
                total_system_time += wait_time + service_time
            else:
                comp1_free = clock

        elif event_type == 'departure2':
            total_served += 1
            comp2_busy = False
            next_departure2 = float('inf')

            if queue:
                wait_time = clock - queue.popleft()
                total_wait_time += wait_time
                last_queue_length = len(queue)
                service_time = random.uniform(SERVICE_MIN, SERVICE_MAX)
                next_departure2 = clock + service_time
                comp2_busy = True
                comp2_free = next_departure2
                total_system_time += wait_time + service_time
            else:
                comp2_free = clock

    total_time = clock

    if comp1_free < total_time:
        comp1_idle_time += total_time - comp1_free
    if comp2_available and comp2_free < total_time:
        comp2_idle_time += total_time - comp2_free

    comp1_total_time = total_time
    comp2_total_time = max(0, total_time - SWITCH_TIME)

    avg_wait = total_wait_time / total_served if total_served > 0 else 0
    avg_system_time = total_system_time / total_served if total_served > 0 else 0
    avg_queue_length = queue_length_sum / total_time if total_time > 0 else 0
    p_idle1 = comp1_idle_time / comp1_total_time if comp1_total_time > 0 else 0
    p_idle2 = comp2_idle_time / comp2_total_time if comp2_total_time > 0 else 0
    p_no_queue = served_no_queue / total_served if total_served > 0 else 0

    return {
        'total_time': total_time,
        'total_served': total_served,
        'total_arrivals': total_arrivals,
        'max_queue_length': max_queue_length,
        'avg_queue_length': avg_queue_length,
        'avg_wait_time': avg_wait,
        'avg_system_time': avg_system_time,
        'p_idle1': p_idle1,
        'p_idle2': p_idle2,
        'p_no_queue': p_no_queue,
        'served_no_queue': served_no_queue,
        'comp1_util': 1 - p_idle1,
        'comp2_util': 1 - p_idle2 if comp2_available else 0,
    }


# Запуск 100 экспериментов
N_EXPERIMENTS = 100
all_results = []
for exp in range(N_EXPERIMENTS):
    res = run_smo_simulation(seed=1000 + exp)
    all_results.append(res)

def collect_stats(key, results):
    vals = [r[key] for r in results]
    return {'mean': np.mean(vals), 'std': np.std(vals), 'min': np.min(vals), 'max': np.max(vals)}

keys = ['total_time', 'total_arrivals', 'max_queue_length', 'avg_queue_length',
        'avg_wait_time', 'avg_system_time', 'p_idle1', 'p_idle2', 
        'p_no_queue', 'comp1_util', 'comp2_util', 'served_no_queue']

stats = {k: collect_stats(k, all_results) for k in keys}

print("\n" + "=" * 70)
print("ЧАСТЬ 2: Имитационное моделирование СМО (принцип δz)")
print("100 экспериментов, 1000 заявок каждый")
print("=" * 70)
print(f"{'Параметр':<35} {'Среднее':>12} {'Стд.откл.':>12} {'Мин':>10} {'Макс':>10}")
print("-" * 80)
for k in keys:
    s = stats[k]
    print(f"{k:<35} {s['mean']:>12.4f} {s['std']:>12.4f} {s['min']:>10.4f} {s['max']:>10.4f}")

# Графики части 2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

wait_times = [r['avg_wait_time'] for r in all_results]
axes[0,0].hist(wait_times, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
axes[0,0].axvline(np.mean(wait_times), color='r', linestyle='--', label=f'Средн.: {np.mean(wait_times):.1f}')
axes[0,0].set_title('Среднее время ожидания')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

queue_lens = [r['avg_queue_length'] for r in all_results]
axes[0,1].hist(queue_lens, bins=20, color='coral', edgecolor='black', alpha=0.8)
axes[0,1].axvline(np.mean(queue_lens), color='r', linestyle='--', label=f'Средн.: {np.mean(queue_lens):.1f}')
axes[0,1].set_title('Средняя длина очереди')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

util1 = [r['comp1_util']*100 for r in all_results]
util2 = [r['comp2_util']*100 for r in all_results]
axes[1,0].scatter(range(len(util1)), util1, s=8, color='blue', alpha=0.6, label='Comp1')
axes[1,0].scatter(range(len(util2)), util2, s=8, color='green', alpha=0.6, label='Comp2')
axes[1,0].set_title('Загрузка компьютеров')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

max_q = [r['max_queue_length'] for r in all_results]
axes[1,1].hist(max_q, bins=20, color='mediumpurple', edgecolor='black', alpha=0.8)
axes[1,1].axvline(np.mean(max_q), color='r', linestyle='--', label=f'Средн.: {np.mean(max_q):.1f}')
axes[1,1].set_title('Максимальная длина очереди')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('part2_smo.png', dpi=150)
plt.show()
