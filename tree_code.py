import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.

    Критерий Джини определяется следующим образом:
    .. math::
        Q(R) = -\\frac {|R_l|}{|R|}H(R_l) -\\frac {|R_r|}{|R|}H(R_r),

    где:
    * :math:`R` — множество всех объектов,
    * :math:`R_l` и :math:`R_r` — объекты, попавшие в левое и правое поддерево соответственно.

    Функция энтропии :math:`H(R)`:
    .. math::
        H(R) = 1 - p_1^2 - p_0^2,

    где:
    * :math:`p_1` и :math:`p_0` — доля объектов класса 1 и 0 соответственно.

    Указания:
    - Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    - В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    - Поведение функции в случае константного признака может быть любым.
    - При одинаковых приростах Джини нужно выбирать минимальный сплит.
    - Для оптимизации рекомендуется использовать векторизацию вместо циклов.

    Parameters
    ----------
    feature_vector : np.ndarray
        Вектор вещественнозначных значений признака.
    target_vector : np.ndarray
        Вектор классов объектов (0 или 1), длина `feature_vector` равна длине `target_vector`.

    Returns
    -------
    thresholds : np.ndarray
        Отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на
        два различных поддерева.
    ginis : np.ndarray
        Вектор со значениями критерия Джини для каждого порога в `thresholds`.
    threshold_best : float
        Оптимальный порог для разбиения.
    gini_best : float
        Оптимальное значение критерия Джини.

    """
        
    sorted_indices = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_indices]
    target_vector = target_vector[sorted_indices]
    
    unique_values = np.unique(feature_vector)
    if len(unique_values) == 1:
        return np.array([]), np.array([]), None, None
    
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    
    ginis = []
    for threshold in thresholds:
        left_mask = feature_vector <= threshold
        right_mask = feature_vector > threshold
        
        y_left = target_vector[left_mask]
        y_right = target_vector[right_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            continue
        
        gini_left = 1 - np.sum((np.bincount(y_left, minlength=2) / len(y_left)) ** 2)
        gini_right = 1 - np.sum((np.bincount(y_right, minlength=2) / len(y_right)) ** 2)
        
        gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(target_vector)
        ginis.append(gini)
    
    if len(ginis) == 0:
        return np.array([]), np.array([]), None, None
    
    ginis = np.array(ginis)
    
    best_index = np.argmin(ginis)
    threshold_best = thresholds[best_index]
    gini_best = ginis[best_index]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is an unknown feature type")
        if not isinstance(feature_types, list) or not feature_types:
            raise ValueError("`feature_types` must be a non-empty list")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._num_features = len(feature_types)

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if sub_X.shape[1] != self._num_features:
            raise ValueError("Number of features in sub_X does not match the number of feature types")

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if (self._max_depth is not None and depth >= self._max_depth) or \
           (len(sub_y) < self._min_samples_split) or \
           (len(sub_y) < self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, float('inf'), None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: clicks.get(key, 0) / count for key, count in counts.items()}
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {category: i for i, category in enumerate(sorted_categories)}
                feature_vector = np.vectorize(categories_map.get)(sub_X[:, feature])
            else:
                raise ValueError("Invalid feature type")

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini < gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [k for k, v in categories_map.items() if v < threshold]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError("Invalid feature type")

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_index = node["feature_split"]

        if self._feature_types[feature_index] == "real":
            if x[feature_index] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_index] == "categorical":
            if x[feature_index] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        return node["class"]

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_indices]
    target_vector = target_vector[sorted_indices]
    
    unique_values = np.unique(feature_vector)
    if len(unique_values) == 1:
        return np.array([]), np.array([]), None, None
    
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    ginis = []
    for threshold in thresholds:
        left_mask = feature_vector <= threshold
        right_mask = feature_vector > threshold
        
        y_left = target_vector[left_mask]
        y_right = target_vector[right_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            continue
        
        gini_left = 1 - np.sum((np.bincount(y_left, minlength=2) / len(y_left)) ** 2)
        gini_right = 1 - np.sum((np.bincount(y_right, minlength=2) / len(y_right)) ** 2)
        
        gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(target_vector)
        ginis.append(gini)
    
    if len(ginis) == 0:
        return np.array([]), np.array([]), None, None
    
    thresholds, ginis = np.array(thresholds), np.array(ginis)
    best_index = np.argmin(ginis)
    threshold_best = thresholds[best_index]
    gini_best = ginis[best_index]
    
    return thresholds, ginis, threshold_best, gini_best

