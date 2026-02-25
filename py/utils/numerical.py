

def has_exact_root(number, root_degree=2):
    """
    Verify if a number has an exact root.
    
    Args:
        number: The number to check
        root_degree: The degree of the root (2 for square root, 3 for cube root, etc.)
    
    Returns:
        tuple: (bool, float) - (True if exact root exists, the root value)
    
    Examples:
        >>> has_exact_root(16, 2)
        (True, 4.0)
        >>> has_exact_root(27, 3)
        (True, 3.0)
        >>> has_exact_root(10, 2)
        (False, 3.1622776601683795)
    """
    if number < 0 and root_degree % 2 == 0:
        return (False, None)  # Even roots of negative numbers are not real
    
    # Handle negative numbers with odd roots
    if number < 0:
        root = -(abs(number) ** (1 / root_degree))
    else:
        root = number ** (1 / root_degree)
    
    # Round to nearest integer
    nearest_int = round(root)
    
    # Check if the integer raised to the power equals the original number
    is_exact = (nearest_int ** root_degree) == number
    
    return (is_exact, root if is_exact else root)


# Alternative implementation using integer arithmetic (more accurate)
def has_exact_integer_root(number, root_degree=2):
    """
    Verify if a number has an exact integer root using integer arithmetic.
    More accurate for large numbers.
    """
    if number < 0 and root_degree % 2 == 0:
        return (False, None)
    
    # Handle negative numbers with odd roots
    if number < 0:
        candidate = -int(abs(number) ** (1 / root_degree) + 0.5)
    else:
        candidate = int(number ** (1 / root_degree) + 0.5)
    
    # Check candidates around the computed value
    for offset in [0, -1, 1]:
        test_val = candidate + offset
        if test_val ** root_degree == number:
            return (True, test_val)
    
    return (False, None)
