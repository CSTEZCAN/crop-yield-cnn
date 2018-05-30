def allow_multi_index(idx, get_item_function):
    """
    Parse input index arguments and return either a single item or list of items accordingly.

    Args:

        idx: Either a single index, a slice or a container of indices.
        get_item_function: The function with which the items are to be retrieved.

    Returns:

        Either a single item or a list of items produced by the function.
    """
    items = []

    if isinstance(idx, (list, tuple, set)):

        for i in idx:

            items.append(get_item_function(i))

        return items

    if isinstance(idx, slice):

        start, end, step = idx.start, idx.stop, idx.step

        if start is None:

            start = 0

        if end is None:

            end = start + 1

        if step is None:

            step = 1

        for i in range(start, end, step):

            items.append(get_item_function(i))

        return items

    return get_item_function(idx)
