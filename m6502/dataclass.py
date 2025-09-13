from m6502.uneditble import uneditable  # Corrected module name
from typing import Dict, ClassVar

@uneditable
class token:
    TOKENS: ClassVar[Dict[str, int]] = {
        # Core keywords
        'PRINT': 101, 'LET': 102, 'GOTO': 103, 'IF': 104, 'THEN': 105,
        'INPUT': 106, 'END': 107, 'DIM': 108, 'READ': 109, 'RESTORE': 110,
        'DATA': 111, 'REM': 112, 'FOR': 113, 'TO': 114, 'STEP': 115, 'NEXT': 116,
        'GOSUB': 117, 'RETURN': 118,
        
        # Operators and punctuation
        '+': 201, '-': 202, '*': 203, '/': 204, '=': 205, '<': 206, '>': 207,
        ',': 208, ';': 209, '(': 210, ')': 211, ':': 212,
        
        # String functions
        'LEFT$': 301, 'RIGHT$': 302, 'MID$': 303, 'LEN': 304, 'CHR$': 305, 'ASC': 306,
        
        # Math functions
        'SIN': 401, 'COS': 402, 'ABS': 403, 'SQR': 404, 'RND': 405,
        
        # Memory operations
        'PEEK': 501, 'POKE': 502
    }
    
    KEYWORDS: ClassVar[Dict[int, str]] = {v: k for k, v in TOKENS.items() if k.isalpha() or k.endswith('$')}
    SYMBOLS: ClassVar[Dict[int, str]] = {v: k for k, v in TOKENS.items() if not (isinstance(k, str) and k.isupper())}
    
    @staticmethod
    def get_token_name(code: int) -> str:
        if code in token.KEYWORDS:
            return token.KEYWORDS[code]
        elif code in token.SYMBOLS:
            return token.SYMBOLS[code]
        else:
            raise ValueError(f"Invalid token code: {code}")

        
    @staticmethod
    def get_token_code(name: str | int) -> int:
        if isinstance(name, int):
            return name
        if name in token.TOKENS:
            return token.TOKENS[name]
        if name.isdigit():
            return int(name)
        raise ValueError(f"Invalid token name: {name}")
