"""Punto 1. Distribuciones discretas y decisiones de respuesta."""
import math


def normalize(weights: dict) -> dict:
    """Retorna un diccionario nuevo cuyas probabilidades suman uno.

    Entrada: weights = {resultado: peso}.
    Salida: {resultado: probabilidad}, con las mismas claves.

    Tips:
    - Conserve las claves y su orden; no modifique weights.
    - Rechace pesos negativos, no finitos, diccionarios vacíos y suma cero.
    """

    if len(weights) <= 0:
        raise ValueError("The weight dictionary can't be empty.")
    
    normalized_weights = {}
    normalizer = 0
    for i in weights:

        current_weight = weights[i]
        if current_weight < 0:
            raise ValueError("Weights must be non-negative numbers.")
        if not math.isfinite(current_weight):
            raise ValueError("Weights must be finite numbers.")
        
        normalizer += weights[i]

    if normalizer <= 0:
        raise ValueError("Normalizer can't be zero. Avoid zero division error.")
        
    for i in weights:
        normalized_weights[i] = weights[i]/normalizer
    return normalized_weights
    


def expected_utility(distribution: dict, utilities: dict) -> float:
    """Calcula sum_s P(s) U(s) para una acción.

    Entradas: distribution = {resultado: probabilidad} y
    utilities = {resultado: utilidad} para una sola acción.
    Salida: un número con la utilidad esperada.

    Tips:
    - distribution ya debe estar normalizada; no normalice utilidades.
    - Ambas tablas deben contener los mismos resultados posibles.
    """
    if set(distribution) != set(utilities) or not distribution:
        raise ValueError("Probabilidades y utilidades deben tener las mismas claves")
    if any(not math.isfinite(p) or p < 0 for p in distribution.values()):
        raise ValueError("Distribución inválida")
    if not math.isclose(sum(distribution.values()), 1, abs_tol=1e-9):
        raise ValueError("La distribución debe sumar uno")
    if any(not math.isfinite(u) for u in utilities.values()):
        raise ValueError("Utilidades no finitas")

    expected_utility_val = 0
    for i in distribution:
        expected_utility_val += distribution[i]*utilities[i]
    return expected_utility_val


def best_action(distribution: dict, actions: dict) -> tuple[str, dict]:
    """Retorna (nombre de acción, utilidades esperadas por acción).

    Entradas: distribution = {resultado: probabilidad} y
    actions = {acción: {resultado: utilidad}}.
    Salida: (acción elegida, {acción: utilidad esperada}).

    Tips:
    - actions asocia cada acción con su tabla de utilidades.
    - Reutilice expected_utility y conserve la primera acción en los empates.
    """
    if not actions:
        raise ValueError("Se requiere al menos una acción")

    max_eu = float('-inf')
    best_action_key = None
    actions_utilities = {}

    for key in actions:
        world_utility = actions[key]
        eu = expected_utility(distribution, world_utility)
        actions_utilities[key] = eu
        if best_action_key is None or eu > max_eu:
            max_eu = eu
            best_action_key = key

    return best_action_key, actions_utilities
