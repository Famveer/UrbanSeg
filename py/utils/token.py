import random
import re

"""
Acceptable characters:
0-9, A-Z, a-z, _
"""

symbols = (
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    + 'abcdefghijklmnopqrstuvwxyz'
    + '0123456789'
    + '_'
)

nSymbols = len(symbols)

def generateToken(size = 40):

    """ Function to generate a token
        
    Args:
        size: token size
        
    Returns:
        str: random token
    
    """
    tokenSymbols = []
    for i in range(size):
        randIx = random.randint(0, nSymbols - 1)
        tokenSymbols.append(symbols[randIx])
    return ''.join(tokenSymbols)

def isToken(token):

    """ Function to verify is a token
        
    Args:
        token: token 
        
    Returns:
        bool: True is token. Otherwise, false
    
    """
    try:
        int_token = int(token)
        return False
    except:
        return not re.match(r"[0-9]{6,}-[0-9]{2,}.[0-9]{4}.[0-9]{1,}.[0-9]{2,}.[0-9]{4,}", token)
