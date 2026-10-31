"""Puntos 4 y 5. Bellman, iteración de valores y política."""
from dataclasses import dataclass
import math




@dataclass
class PlanningResult:
    """Valores e historial; no modificar."""
    values: dict
    history: list
    deltas: list
    converged: bool




def action_value(mdp, state, action, values):
    """Punto 4: suma T(s,a,s') [R(s,a,s') + gamma V(s')].


    Entradas: state es (x, y), action es N/E/S/W y values es {(x, y): valor}.
    Salida: un número con el valor Q de esa acción.


    Tips:
    - mdp.transitions(state, action) retorna pares (sucesor, probabilidad).
    - Use mdp.reward(state, action, sucesor) y mdp.discount.
    - No vuelva a sumar la recompensa del terminal en su valor:
      values[terminal] es cero.
    """
    # TODO: Add your code here
    q = 0.0
    for successor, prob in mdp.transitions(state, action):
        r = mdp.reward(state, action, successor)
        q += prob * (r + mdp.discount * values[successor])
    return q
    # FIN DEL BLOQUE DEL ESTUDIANTE




def value_iteration(mdp, tolerance=1e-8, max_iterations=1000):
    """Punto 4: barridos SINCRÓNICOS, desde V0=0.


    Entradas: mdp es el modelo, tolerance el umbral de parada
    y max_iterations el máximo de barridos.
    Estructura: values = {(x, y): valor}; mdp.states contiene los estados y
    mdp.actions(estado) sus acciones, o una secuencia vacía en terminales.
    El retorno PlanningResult se construye fuera del bloque TODO.


    Tips:
    - Cada actualización usa exclusivamente el diccionario del barrido anterior.
    - Mantenga terminales en cero. delta=max_s |Vnuevo(s)-Vanterior(s)|.
    - Pare con delta < tolerance o al completar max_iterations barridos.
    - Complete solo los bloques TODO. El historial y el empaquetado de
      resultados ya están implementados; no implemente código de interfaz.
    - Al terminar el bloque, values debe contener el nuevo barrido, delta
      su cambio máximo y converged el resultado de delta < tolerance.
    """
    # Validación e inicialización desde V0=0.
    if not math.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError('Tolerancia y límite de iteraciones deben ser positivos')
    values = dict.fromkeys(mdp.states, 0.0)
    history, deltas = [values.copy()], []
    for _ in range(max_iterations):
        # TODO: Add your code here
        previous = values
        new_values = {}
        for s in mdp.states:
            acts = mdp.actions(s)
            if not acts:
                new_values[s] = 0.0
            else:
                new_values[s] = max(action_value(mdp, s, a, previous) for a in acts)
        delta = max(abs(new_values[s] - previous[s]) for s in mdp.states)
        values = new_values
        converged = delta < tolerance
        # FIN DEL BLOQUE DEL ESTUDIANTE
        # Copias para la animación y empaquetado del resultado.
        history.append(values.copy())
        deltas.append(delta)
        if converged:
            return PlanningResult(values, history, deltas, True)
    return PlanningResult(values, history, deltas, False)




def extract_policy(mdp, values):
    """Punto 5: retorna acción por estado; terminales -> None.


    Entradas: mdp es el modelo y values = {(x, y): valor} viene del punto 4.
    Salida: {(x, y): acción}, usando N/E/S/W o None para terminales.


    Tips:
    - Reutilice action_value: no basta elegir el vecino de mayor valor.
    - Respete mdp.actions: desempate N, E, S, W conservando el primero.
    """
    # TODO: Add your code here
    policy = {}
    for s in mdp.states:
        acts = mdp.actions(s)
        if not acts:
            policy[s] = None
            continue
        best_a, best_q = None, -math.inf
        for a in acts:  # orden N, E, S, W
            q = action_value(mdp, s, a, values)
            # Mejora estricta con un margen mínimo: diferencias de redondeo
            # (~1e-16) cuentan como empate y se conserva la primera acción.
            if q > best_q + 1e-12:
                best_a, best_q = a, q
        policy[s] = best_a
    return policy
    # FIN DEL BLOQUE DEL ESTUDIANTE



