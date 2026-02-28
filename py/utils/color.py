import ast

def rgb2hex(rgb_color):
    # If string, try to parse safely
    if isinstance(rgb_color, str):
        try:
            rgb_color = ast.literal_eval(rgb_color)
        except (ValueError, SyntaxError):
            raise ValueError("Invalid string format. Expected '(R,G,B)'")

    # Validate type
    if not isinstance(rgb_color, (tuple, list)):
        raise TypeError("rgb_color must be tuple, list, or string '(R,G,B)'")

    # Validate length
    if len(rgb_color) != 3:
        raise ValueError("rgb_color must have exactly 3 values")

    # Validate values
    if not all(isinstance(c, int) for c in rgb_color):
        raise TypeError("RGB values must be integers")

    if not all(0 <= c <= 255 for c in rgb_color):
        raise ValueError("RGB values must be between 0 and 255")

    return "#{:02x}{:02x}{:02x}".format(*rgb_color)
