"""Punto 2. Inferencia exacta por enumeración."""
from algorithms.probability import normalize


def enumerate_all(network, variables: list, event: dict) -> float:

    if len(variables) == 0:
        return 1.0
        
    var = variables[0]
    rest = variables[1:]
    
    if var in event:
        return network.probability(var, event[var], event) * enumerate_all(network, rest, event)

    total = 0.0

    for val in (0, 1):
    
        extended = dict(event)
        extended[var] = val
        
        total += network.probability(var, val, extended) * enumerate_all(network, rest, extended)
    return total


def enumeration_ask(network, query: str, evidence: dict) -> dict:
    """Calcula P(query | evidence) sobre la conjunta factorizada.

    Entradas: network es la red; query es el nombre de la
    variable consultada; evidence es {nombre de variable: 0 o 1}.
    Salida: {0: probabilidad, 1: probabilidad}.

    Tips:
    - Recorra network.variables en orden topológico.
    - Para una variable fijada multiplique su probabilidad; para una oculta
      sume las dos alternativas 0 y 1. Normalice al terminar.
    - Use copias del evento y no modifique evidence.
    - network.probability(variable, valor, evento) consulta la CPT;
      el evento debe contener los valores de todos los padres.
    - Puede usar una función auxiliar recursiva. Al terminar el recorrido,
      el producto sin factores restantes vale 1.
    - Evidencia imposible: ValueError. Consulta observada: distribución puntual
      solo si la evidencia es posible.
    """
    # Valida nombres de variables y dominio de la evidencia.
    network.validate_query(query, evidence)

    weights = {}
    for val in (0, 1):
    
        if query in evidence and evidence[query] != val:
        
            weights[val] = 0.0
        else:
            extended = dict(evidence)
            extended[query] = val
            
            weights[val] = enumerate_all(network, network.variables, extended)

    return normalize(weights)
    # FIN DEL BLOQUE DEL ESTUDIANTE
