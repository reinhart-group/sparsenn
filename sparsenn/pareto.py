def is_dominated(point, pareto_points):
    """
    Check if a given point is dominated by any point in the current Pareto front.
    """
    for p in pareto_points:
        if all(p <= point) and any(p < point):
            return True
    return False


def update_pareto_front(new_point, pareto_points):
    """
    Update the Pareto front with a new point.
    """
    non_dominated_points = []
    for p in pareto_points:
        if all(p <= new_point) and any(p < new_point):
            continue  # Point p is dominated by new_point
        if all(new_point <= p) and any(new_point < p):
            continue  # new_point is dominated by point p
        non_dominated_points.append(p)
    non_dominated_points.append(new_point)
    return non_dominated_points
