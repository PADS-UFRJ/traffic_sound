
# ---------------------------

# [REF] Stack Overflow: How do you sort files numerically? - https://stackoverflow.com/a/4623518

from re import split

def tryint(s):
    return int(s) if s.isdigit() else s

def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in split('([0-9]+)', str(s)) ]

def sort_nicely(l):
    """
    Sort the given list in the way that humans expect.
        sort_nicely( ['f_2.png', 'f_1.png', 'f_10.png'] ) -> ['f_1.png', 'f_2.png', 'f_10.png']
    """
    return sorted(l, key=alphanum_key)

# ---------------------------