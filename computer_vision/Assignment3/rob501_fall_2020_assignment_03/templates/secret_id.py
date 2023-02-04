def secret_id():
    """
    Return a unique secret identifier.

    The fuction should return your unique secret identifier (a string).
    The identifier must be 32 characters or less in length.

    Returns:
    --------
    id  - String identifier (class must String).
    """
    #--- FILL ME IN ---

    id =  "GODFATHER"  # Update with your ID!

    #------------------

    correct = isinstance(id, str) and len(id) <= 32 and len(id) >= 4

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return id