import numpy as np
import matplotlib.pyplot as plt

# Нормализация данных
def normalize_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Инициализация весов для 2D-сетки
def initialize_weights(input_size, grid_size):
    return np.random.rand(grid_size[0], grid_size[1], input_size)

# Обучение сети Кохонена
def train_kohonen_network(input_data, weights, learning_rate_init, epochs):
    grid_size = weights.shape[:2]
    for epoch in range(epochs):
        learning_rate = learning_rate_init * (1 - epoch / epochs)

        for input_vector in input_data:
            distances = np.linalg.norm(weights - input_vector, axis=2)
            winner_index = np.unravel_index(np.argmin(distances), grid_size)

            weights[winner_index] += learning_rate * (input_vector - weights[winner_index])

    return weights

def plot_feature_heatmap(weights, feature_index, grid_size, title):
    feature_map = weights[:, :, feature_index]

    plt.figure(figsize=(10, 10))
    plt.imshow(feature_map, cmap='plasma', interpolation='bilinear')
    plt.colorbar()
    plt.title(title)
    plt.show()

np.random.seed(0)  # для воспроизводимости
ages = np.random.randint(25, 70, 20)  # Возраст от 25 до 70
salaries = np.random.randint(50000, 150000, 20)  # Зарплата от 50000 до 150000
purchases = np.random.randint(20, 100, 20)  # Количество покупок от 20 до 100

input_data = np.array(list(zip(ages, salaries, purchases)))

# Нормализация данных
input_data = normalize_data(input_data)

# Параметры сети
input_size = 3
grid_size = (20, 20)
learning_rate_init = 0.1
epochs = 10000

# Инициализация весов
weights = initialize_weights(input_size, grid_size)

# Обучение сети
trained_weights = train_kohonen_network(input_data, weights, learning_rate_init, epochs)

# Визуализация каждого признака на отдельной карте
plot_feature_heatmap(trained_weights, 0, grid_size, 'Тепловая карта возраста')
plot_feature_heatmap(trained_weights, 1, grid_size, 'Тепловая карта зарплаты')
plot_feature_heatmap(trained_weights, 2, grid_size, 'Тепловая карта количества покупок')
