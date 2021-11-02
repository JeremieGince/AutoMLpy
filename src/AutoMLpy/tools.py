
plotly_colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def multiplyIsOverflow(a, b):
    """
    Function to check whether there is
    overflow in a * b or not. It returns
    true if there is overflow.

    Returns
    -------
        True if overflow else False.
    """
    # Check if either of them is zero
    if a == 0 or b == 0:
        return False

    result = a * b
    if (result >= 9223372036854775807 or
            result <= -9223372036854775808):
        result = 0
    if a == (result // b):
        return False
    else:
        return True


def try_prod_overflow(iterable):
    result = 1
    for i in range(len(iterable)):
        if multiplyIsOverflow(result, iterable[i]):
            raise ValueError("Overflow detected.")
        else:
            result *= iterable[i]
    return result


