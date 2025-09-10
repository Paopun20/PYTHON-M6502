import re
import sys
import math
import os
import array
import logging

from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install
install()
Console().clear()

file_handler = logging.FileHandler("m6502.log", mode="w")  # overwrite each run
file_handler.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=True,
        ),
        file_handler
    ]
)

logger = logging.getLogger("M6502")

VARS = {chr(65 + i): 0.0 for i in range(26)}  # Numeric variables
STR_VARS = {chr(65 + i) + '$': '' for i in range(26)}  # String variables
ARRAYS = {}  # Numeric arrays
STR_ARRAYS = {}  # String arrays
LINES = {}  # line_num -> tokenized line
DATA_PTR = 0  # For RESTORE and READ
DATA_LINES = []  # Flattened data tokens
GOSUB_STACK = []
FOR_STACK = []
MAX_STACK_DEPTH = 100
try:
    TERM_WIDTH = os.get_terminal_size().columns
except OSError:
    TERM_WIDTH = 80  # Default width if terminal size is unavailable

TOKENS = {
    'PRINT': 160, 'LET': 161, 'GOTO': 162, 'IF': 163, 'THEN': 164,
    'INPUT': 165, 'END': 166, '+': 167, '-': 168, '*': 169, '/': 170,
    '=': 171, '<': 172, '>': 173, ',': 174, ';': 175, '(': 176, ')': 177,
    'DIM': 178, 'READ': 179, 'RESTORE': 180, 'DATA': 181, 'REM': 182,
    'FOR': 183, 'TO': 184, 'STEP': 185, 'NEXT': 186, 'GOSUB': 187, 'RETURN': 188,
    ':': 189, 'LEFT$': 190, 'RIGHT$': 191, 'MID$': 192, 'LEN': 193, 'CHR$': 194,
    'ASC': 195, 'SIN': 196, 'COS': 197, 'ABS': 198, 'SQR': 199, 'RND': 200,
    'PEEK': 201, 'POKE': 202
}
KEYWORDS = {v: k for k, v in TOKENS.items() if isinstance(k, str) and k.isupper()}
SYMBOLS = {v: k for k, v in TOKENS.items() if not (isinstance(k, str) and k.isupper())}
MEMORY = array.array('B', [0] * 65536)  # Simulated 6502 memory

class BasicError(Exception):
    def __init__(self, message):
        logger.error(message)
        super().__init__(message)

def tokenize(line):
    parts = []
    i = 0
    while i < len(line):
        if line[i] == '"':
            j = i + 1
            while j < len(line) and line[j] != '"':
                j += 1
            if j >= len(line):
                raise BasicError("Unterminated string literal")
            parts.append(line[i:j+1])
            i = j + 1
        else:
            j = i
            while j < len(line) and line[j] not in ' \t"+-*/=<>;,():':
                j += 1
            if i < j:
                parts.append(line[i:j])
            if j < len(line) and line[j] not in ' \t':
                parts.append(line[j])
            i = max(j + 1, i + 1)
    
    parts = [p for p in parts if p.strip()]
    
    tokens = []
    for p in parts:
        if p.startswith('"') and p.endswith('"') and len(p) >= 2:
            tokens.append(p[1:-1])  # Store as string literal without quotes
        elif p.upper() in TOKENS:
            tokens.append(TOKENS[p.upper()])
        elif len(p) == 1 and p.upper().isalpha():
            tokens.append(ord(p.upper()))
        elif p.upper().endswith('$') and len(p) == 2 and p[0].isalpha():
            tokens.append(p.upper())
        elif re.match(r'^-?\d+(\.\d+)?$', p):
            tokens.append(float(p))
        elif p in '+-*/=<>;,():':
            tokens.append(TOKENS[p])
        else:
            tokens.append(p.upper())
    return tokens

def detokenize(tokens):
    s = ''
    for t in tokens:
        if isinstance(t, str) and t not in TOKENS and not t.endswith('$'):
            s += f'"{t}" '
        elif isinstance(t, str) and t.endswith('$'):
            s += t + ' '
        elif isinstance(t, int) and 65 <= t <= 90:
            s += chr(t) + ' '
        elif t in KEYWORDS:
            s += KEYWORDS[t] + ' '
        elif t in SYMBOLS:
            s += SYMBOLS[t] + ' '
        elif isinstance(t, float):
            if t.is_integer():
                s += str(int(t)) + ' '
            else:
                s += str(t) + ' '
        else:
            s += str(t) + ' '
    return s.strip()

def parse_line(line):
    global DATA_LINES
    m = re.match(r'^(\d+)\s*(.*)$', line.strip())
    if not m:
        raise BasicError("Line must start with number")
    num = int(m.group(1))
    if num <= 0:
        raise BasicError("Line number must be positive")
    
    if not m.group(2).strip():
        if num in LINES:
            del LINES[num]
            rebuild_data_lines()
        return
    
    code = tokenize(m.group(2))
    LINES[num] = code
    rebuild_data_lines()

def rebuild_data_lines():
    """Rebuild DATA_LINES from all DATA statements in the program."""
    global DATA_LINES
    DATA_LINES = []
    for line_num in sorted(LINES.keys()):
        tokens = LINES[line_num]
        if tokens and tokens[0] == TOKENS['DATA']:
            for token in tokens[1:]:
                if token == TOKENS[',']:
                    continue  # Skip comma tokens
                # Determine if this is a string literal or numeric value
                is_string_literal = isinstance(token, str) and not token.endswith('$') and not token.isupper()
                DATA_LINES.append((token, is_string_literal))

def parse_factor(tokens, pos):
    if pos[0] >= len(tokens):
        raise BasicError("Unexpected end of expression")
        
    t = tokens[pos[0]]
    pos[0] += 1
    
    if isinstance(t, float):
        return t
    elif isinstance(t, int) and 65 <= t <= 90:
        var = chr(t)
        if pos[0] < len(tokens) and tokens[pos[0]] == TOKENS['(']:
            pos[0] += 1
            idx_pos = [pos[0]]
            idx = int(parse_expr(tokens, idx_pos))
            pos[0] = idx_pos[0]
            if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
                raise BasicError("Expected )")
            pos[0] += 1
            if idx < 0:
                raise BasicError("Negative array index")
            if var in ARRAYS and idx < len(ARRAYS[var]):
                return ARRAYS[var][idx]
            else:
                raise BasicError(f"Array {var}({idx}) out of bounds")
        return VARS.get(var, 0.0)
    elif isinstance(t, str) and t.endswith('$'):
        var = t
        if pos[0] < len(tokens) and tokens[pos[0]] == TOKENS['(']:
            pos[0] += 1
            idx_pos = [pos[0]]
            idx = int(parse_expr(tokens, idx_pos))
            pos[0] = idx_pos[0]
            if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
                raise BasicError("Expected )")
            pos[0] += 1
            if idx < 0:
                raise BasicError("Negative array index")
            if var in STR_ARRAYS and idx < len(STR_ARRAYS[var]):
                return STR_ARRAYS[var][idx]
            else:
                raise BasicError(f"String array {var}({idx}) out of bounds")
        return STR_VARS.get(var, '')
    elif t == TOKENS['-']:
        return -parse_factor(tokens, pos)
    elif t == TOKENS['+']:
        return parse_factor(tokens, pos)
    elif t == TOKENS['(']:
        val = parse_expr(tokens, pos)
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
            raise BasicError("Expected )")
        pos[0] += 1
        return val
    elif t in (TOKENS['SIN'], TOKENS['COS'], TOKENS['ABS'], TOKENS['SQR'], TOKENS['RND']):
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS['(']:
            raise BasicError("Expected (")
        pos[0] += 1
        val = parse_expr(tokens, pos)
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
            raise BasicError("Expected )")
        pos[0] += 1
        if t == TOKENS['SIN']:
            return math.sin(val)
        elif t == TOKENS['COS']:
            return math.cos(val)
        elif t == TOKENS['ABS']:
            return abs(val)
        elif t == TOKENS['SQR']:
            if val < 0:
                raise BasicError("Negative square root")
            return math.sqrt(val)
        elif t == TOKENS['RND']:
            return (val * 1103515245 + 12345) & 0x7fffffff / 0x7fffffff
    elif t == TOKENS['LEN']:
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS['(']:
            raise BasicError("Expected (")
        pos[0] += 1
        val = parse_string(tokens, pos)
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
            raise BasicError("Expected )")
        pos[0] += 1
        return float(len(val))
    elif t == TOKENS['ASC']:
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS['(']:
            raise BasicError("Expected (")
        pos[0] += 1
        val = parse_string(tokens, pos)
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
            raise BasicError("Expected )")
        pos[0] += 1
        if not val:
            raise BasicError("Empty string for ASC")
        return float(ord(val[0]))
    elif t == TOKENS['PEEK']:
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS['(']:
            raise BasicError("Expected (")
        pos[0] += 1
        addr = int(parse_expr(tokens, pos))
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
            raise BasicError("Expected )")
        pos[0] += 1
        if 0 <= addr < 65536:
            return float(MEMORY[addr])
        raise BasicError("Invalid memory address")
    else:
        raise BasicError(f"Invalid token in expression: {t}")

def parse_string(tokens, pos):
    if pos[0] >= len(tokens):
        raise BasicError("Expected string expression")
    t = tokens[pos[0]]
    if isinstance(t, str) and not t.endswith('$'):
        pos[0] += 1
        return t
    elif isinstance(t, str) and t.endswith('$'):
        pos[0] += 1
        if pos[0] < len(tokens) and tokens[pos[0]] == TOKENS['(']:
            pos[0] += 1
            idx_pos = [pos[0]]
            idx = int(parse_expr(tokens, idx_pos))
            pos[0] = idx_pos[0]
            if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
                raise BasicError("Expected )")
            pos[0] += 1
            if idx < 0:
                raise BasicError("Negative array index")
            if t in STR_ARRAYS and idx < len(STR_ARRAYS[t]):
                return STR_ARRAYS[t][idx]
            raise BasicError(f"String array {t}({idx}) out of bounds")
        return STR_VARS.get(t, '')
    elif t == TOKENS['CHR$']:
        pos[0] += 1
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS['(']:
            raise BasicError("Expected (")
        pos[0] += 1
        val = int(parse_expr(tokens, pos))
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
            raise BasicError("Expected )")
        pos[0] += 1
        if 0 <= val <= 255:
            return chr(val)
        raise BasicError("Invalid CHR$ value")
    elif t == TOKENS['LEFT$']:
        pos[0] += 1
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS['(']:
            raise BasicError("Expected (")
        pos[0] += 1
        s = parse_string(tokens, pos)
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[',']:
            raise BasicError("Expected ,")
        pos[0] += 1
        n = int(parse_expr(tokens, pos))
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
            raise BasicError("Expected )")
        pos[0] += 1
        if n < 0:
            raise BasicError("Negative length")
        return s[:n]
    elif t == TOKENS['RIGHT$']:
        pos[0] += 1
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS['(']:
            raise BasicError("Expected (")
        pos[0] += 1
        s = parse_string(tokens, pos)
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[',']:
            raise BasicError("Expected ,")
        pos[0] += 1
        n = int(parse_expr(tokens, pos))
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
            raise BasicError("Expected )")
        pos[0] += 1
        if n < 0:
            raise BasicError("Negative length")
        return s[-n:] if n > 0 else ''
    elif t == TOKENS['MID$']:
        pos[0] += 1
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS['(']:
            raise BasicError("Expected (")
        pos[0] += 1
        s = parse_string(tokens, pos)
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[',']:
            raise BasicError("Expected ,")
        pos[0] += 1
        start = int(parse_expr(tokens, pos))
        if pos[0] < len(tokens) and tokens[pos[0]] == TOKENS[',']:
            pos[0] += 1
            length = int(parse_expr(tokens, pos))
        else:
            length = None
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
            raise BasicError("Expected )")
        pos[0] += 1
        if start < 1:
            raise BasicError("Invalid start position")
        start -= 1
        if length is None:
            return s[start:]
        if length < 0:
            raise BasicError("Negative length")
        return s[start:start+length]
    else:
        raise BasicError("Expected string expression")

def parse_term(tokens, pos):
    val = parse_factor(tokens, pos)
    while pos[0] < len(tokens):
        t = tokens[pos[0]]
        if t == TOKENS['*']:
            pos[0] += 1
            val *= parse_factor(tokens, pos)
        elif t == TOKENS['/']:
            pos[0] += 1
            divisor = parse_factor(tokens, pos)
            if divisor == 0:
                raise BasicError("Division by zero")
            val /= divisor
        else:
            break
    return val

def parse_expr(tokens, pos):
    val = parse_term(tokens, pos)
    while pos[0] < len(tokens):
        t = tokens[pos[0]]
        if t == TOKENS['+']:
            pos[0] += 1
            val += parse_term(tokens, pos)
        elif t == TOKENS['-']:
            pos[0] += 1
            val -= parse_term(tokens, pos)
        else:
            break
    return val

def parse_rel(tokens, pos):
    val = parse_expr(tokens, pos)
    if pos[0] < len(tokens):
        op_t = tokens[pos[0]]
        if op_t not in (TOKENS['='], TOKENS['<'], TOKENS['>']):
            return val
        pos[0] += 1
        op = None
        if op_t == TOKENS['<']:
            if pos[0] < len(tokens):
                next_t = tokens[pos[0]]
                if next_t == TOKENS['=']:
                    pos[0] += 1
                    op = '<='
                elif next_t == TOKENS['>']:
                    pos[0] += 1
                    op = '<>'
                else:
                    op = '<'
            else:
                op = '<'
        elif op_t == TOKENS['>']:
            if pos[0] < len(tokens) and tokens[pos[0]] == TOKENS['=']:
                pos[0] += 1
                op = '>='
            else:
                op = '>'
        elif op_t == TOKENS['=']:
            op = '='
        
        right = parse_expr(tokens, pos)
        cond = False
        if op == '=':
            cond = val == right
        elif op == '<':
            cond = val < right
        elif op == '>':
            cond = val > right
        elif op == '<=':
            cond = val <= right
        elif op == '>=':
            cond = val >= right
        elif op == '<>':
            cond = val != right
        val = -1.0 if cond else 0.0
    return val

def execute_statement(tokens, i_pos):
    """Execute a single BASIC statement with full logging."""
    global DATA_PTR, GOSUB_STACK, FOR_STACK, TERM_WIDTH
    i = i_pos[0]
    if i >= len(tokens):
        return None

    tok = tokens[i]

    if tok == TOKENS['REM']:
        logger.debug("REM statement, ignored")
        i_pos[0] = len(tokens)
        return None

    elif tok == TOKENS['PRINT']:
        i += 1
        output = ""
        suppress_newline = False
        zone_width = max(TERM_WIDTH // 5, 14)
        while i < len(tokens) and tokens[i] != TOKENS[':']:
            t = tokens[i]
            if t == TOKENS[',']:
                output += " " * (zone_width - (len(output) % zone_width))
                i += 1
                continue
            if t == TOKENS[';']:
                suppress_newline = True
                i += 1
                continue
            suppress_newline = False
            if isinstance(t, str) and not t.endswith('$'):
                output += t
                i += 1
            elif isinstance(t, str) and t.endswith('$'):
                output += STR_VARS.get(t, '')
                i += 1
            elif isinstance(t, int) and 65 <= t <= 90:
                val = VARS.get(chr(t), 0.0)
                output += str(int(val)) if val.is_integer() else str(val)
                i += 1
            else:
                pos = [i]
                val = parse_expr(tokens, pos)
                i = pos[0]
                output += str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
        print(output, end="" if suppress_newline else "\n")
        logger.info("PRINT: %r", output)
        i_pos[0] = i
        return None

    elif tok == TOKENS['LET']:
        i += 1
        if i >= len(tokens):
            raise BasicError("LET missing variable")
        var_tok = tokens[i]
        var = None

        # Numeric variable
        if isinstance(var_tok, int) and 65 <= var_tok <= 90:
            var = chr(var_tok)
            i += 1
            if i < len(tokens) and tokens[i] == TOKENS['(']:
                # Array element
                i += 1
                pos = [i]
                idx = int(parse_expr(tokens, pos))
                i = pos[0]
                if i >= len(tokens) or tokens[i] != TOKENS[')']:
                    raise BasicError("Expected )")
                i += 1
                if i >= len(tokens) or tokens[i] != TOKENS['=']:
                    raise BasicError("Expected =")
                i += 1
                pos = [i]
                val = parse_expr(tokens, pos)
                i = pos[0]
                ARRAYS[var][idx] = val
                logger.debug("LET array numeric: %s(%d) = %s", var, idx, val)
            else:
                # Simple variable
                if i >= len(tokens) or tokens[i] != TOKENS['=']:
                    raise BasicError("Expected =")
                i += 1
                pos = [i]
                val = parse_expr(tokens, pos)
                i = pos[0]
                VARS[var] = val
                logger.debug("LET numeric: %s = %s", var, val)

        # String variable
        elif isinstance(var_tok, str) and var_tok.endswith('$'):
            var = var_tok
            i += 1
            if i < len(tokens) and tokens[i] == TOKENS['(']:
                # String array element
                i += 1
                pos = [i]
                idx = int(parse_expr(tokens, pos))
                i = pos[0]
                if i >= len(tokens) or tokens[i] != TOKENS[')']:
                    raise BasicError("Expected )")
                i += 1
                if i >= len(tokens) or tokens[i] != TOKENS['=']:
                    raise BasicError("Expected =")
                i += 1
                pos = [i]
                val = parse_string(tokens, pos)
                i = pos[0]
                STR_ARRAYS[var][idx] = val
                logger.debug("LET array string: %s(%d) = %r", var, idx, val)
            else:
                if i >= len(tokens) or tokens[i] != TOKENS['=']:
                    raise BasicError("Expected =")
                i += 1
                pos = [i]
                val = parse_string(tokens, pos)
                i = pos[0]
                STR_VARS[var] = val
                logger.debug("LET string: %s = %r", var, val)
        else:
            raise BasicError(f"Invalid variable in LET: {var_tok}")

        i_pos[0] = i
        return None

    elif tok == TOKENS['INPUT']:
        i += 1
        prompt = "?"
        if i < len(tokens) and isinstance(tokens[i], str) and not tokens[i].endswith('$'):
            prompt = tokens[i]
            i += 1
            if i < len(tokens) and tokens[i] == TOKENS[';']:
                i += 1
        if i >= len(tokens):
            raise BasicError("INPUT requires variable")
        var_tok = tokens[i]
        i += 1
        if isinstance(var_tok, int) and 65 <= var_tok <= 90:
            var = chr(var_tok)
            try:
                user_val = input(prompt + " ")
                VARS[var] = float(user_val)
                logger.info("INPUT numeric: %s = %s", var, VARS[var])
            except ValueError:
                raise BasicError("Invalid numeric input")
        elif isinstance(var_tok, str) and var_tok.endswith('$'):
            var = var_tok
            user_val = input(prompt + " ")
            STR_VARS[var] = user_val
            logger.info("INPUT string: %s = %r", var, user_val)
        else:
            raise BasicError("Invalid variable for INPUT")
        i_pos[0] = i
        return None

    elif tok == TOKENS['DIM']:
        i += 1
        while i < len(tokens) and tokens[i] != TOKENS[':']:
            var = tokens[i]
            i += 1
            if i >= len(tokens) or tokens[i] != TOKENS['(']:
                raise BasicError("Expected ( after array variable")
            i += 1
            pos = [i]
            size = int(parse_expr(tokens, pos))
            i = pos[0]
            if i >= len(tokens) or tokens[i] != TOKENS[')']:
                raise BasicError("Expected )")
            i += 1
            if isinstance(var, str) and var.endswith('$'):
                STR_ARRAYS[var] = [''] * (size + 1)
                logger.debug("DIM string array: %s(%d)", var, size)
            elif isinstance(var, int) and 65 <= var <= 90:
                var_name = chr(var)
                ARRAYS[var_name] = [0.0] * (size + 1)
                logger.debug("DIM numeric array: %s(%d)", var_name, size)
            else:
                raise BasicError("Invalid array variable")
            if i < len(tokens) and tokens[i] == TOKENS[',']:
                i += 1
            else:
                break
        i_pos[0] = i
        return None

    elif tok == TOKENS['READ']:
        i += 1
        while i < len(tokens) and tokens[i] != TOKENS[':']:
            var_tok = tokens[i]
            if isinstance(var_tok, int) and 65 <= var_tok <= 90:
                var = chr(var_tok)
                is_string_var = False
            elif isinstance(var_tok, str) and var_tok.endswith('$'):
                var = var_tok
                is_string_var = True
            else:
                raise BasicError("Invalid variable for READ")
            i += 1

            if DATA_PTR >= len(DATA_LINES):
                raise BasicError("Out of DATA")

            # Extract value and is_string flag
            data_val, is_string_literal = DATA_LINES[DATA_PTR]

            if is_string_var:
                # For string variables, use the data as-is if it's a string literal,
                # otherwise convert numbers to string representation
                if is_string_literal:
                    STR_VARS[var] = data_val
                else:
                    # Convert numeric data to string
                    if isinstance(data_val, float) and data_val.is_integer():
                        STR_VARS[var] = str(int(data_val))
                    else:
                        STR_VARS[var] = str(data_val)
                logger.debug("READ string: %s = %r", var, STR_VARS[var])
            else:
                # For numeric variables, convert string literals to numbers if possible
                if is_string_literal:
                    try:
                        VARS[var] = float(data_val)
                    except ValueError:
                        raise BasicError(f"Expected numeric DATA for {var}, got '{data_val}'")
                else:
                    VARS[var] = float(data_val)
                logger.debug("READ numeric: %s = %s", var, VARS[var])

            DATA_PTR += 1

            if i < len(tokens) and tokens[i] == TOKENS[',']:
                i += 1
            else:
                break
        i_pos[0] = i
        return None

    elif tok == TOKENS['RESTORE']:
        DATA_PTR = 0
        logger.debug("RESTORE: DATA_PTR reset to 0")
        i_pos[0] = len(tokens)
        return None

    # FOR/NEXT logging
    elif tok == TOKENS['FOR']:
        i += 1
        var = chr(tokens[i])
        i += 1
        i += 1  # Skip '='
        pos = [i]
        start_val = parse_expr(tokens, pos)
        i = pos[0]
        i += 1  # Skip TO
        pos = [i]
        end_val = parse_expr(tokens, pos)
        i = pos[0]
        step_val = 1.0
        if i < len(tokens) and tokens[i] == TOKENS['STEP']:
            i += 1
            pos = [i]
            step_val = parse_expr(tokens, pos)
            i = pos[0]
        VARS[var] = start_val
        FOR_STACK.append({'var': var, 'end': end_val, 'step': step_val, 'line_num': None})
        logger.debug("FOR loop: %s = %s TO %s STEP %s", var, start_val, end_val, step_val)
        i_pos[0] = i
        return None

    elif tok == TOKENS['NEXT']:
        if not FOR_STACK:
            raise BasicError("NEXT without FOR")
        i += 1
        next_var = chr(tokens[i]) if i < len(tokens) and isinstance(tokens[i], int) and 65 <= tokens[i] <= 90 else None
        loop_info = FOR_STACK[-1]
        var = loop_info['var']
        VARS[var] += loop_info['step']
        if ((loop_info['step'] > 0 and VARS[var] <= loop_info['end']) or
            (loop_info['step'] < 0 and VARS[var] >= loop_info['end'])):
            i_pos[0] = i
            logger.debug("NEXT %s: continue loop, new value = %s", var, VARS[var])
            return ('FOR_CONTINUE', loop_info['line_num'])
        else:
            FOR_STACK.pop()
            i_pos[0] = i
            logger.debug("NEXT %s: loop finished", var)
            return None

    # GOTO/GOSUB/RETURN logging
    elif tok == TOKENS['GOTO']:
        i += 1
        target = int(tokens[i])
        logger.debug("GOTO line %s", target)
        i_pos[0] = len(tokens)
        return ('GOTO', target)

    elif tok == TOKENS['GOSUB']:
        i += 1
        target = int(tokens[i])
        GOSUB_STACK.append(None)
        logger.debug("GOSUB line %s", target)
        i_pos[0] = len(tokens)
        return ('GOSUB', target)

    elif tok == TOKENS['RETURN']:
        if not GOSUB_STACK:
            raise BasicError("RETURN without GOSUB")
        return_line = GOSUB_STACK.pop()
        logger.debug("RETURN to line %s", return_line)
        i_pos[0] = len(tokens)
        return ('RETURN', return_line)

    elif tok == TOKENS['END']:
        logger.debug("END statement")
        i_pos[0] = len(tokens)
        return ('END',)

    elif tok == TOKENS['DATA']:
        logger.debug("DATA statement ignored during execution")
        i_pos[0] = len(tokens)
        return None

    elif tok == TOKENS['POKE']:
        i += 1
        pos = [i]
        addr = int(parse_expr(tokens, pos))
        i = pos[0]
        if i < len(tokens) and tokens[i] == TOKENS[',']:
            i += 1
        pos = [i]
        val = int(parse_expr(tokens, pos))
        i = pos[0]
        if 0 <= addr < 65536 and 0 <= val <= 255:
            MEMORY[addr] = val
            logger.debug("POKE memory[%d] = %d", addr, val)
        else:
            raise BasicError("POKE address or value out of range")
        i_pos[0] = i
        return None

    elif tok == TOKENS['IF']:
        i += 1
        pos = [i]
        condition = parse_rel(tokens, pos)
        i = pos[0]
        if i >= len(tokens) or tokens[i] != TOKENS['THEN']:
            raise BasicError("Expected THEN")
        i += 1
        if condition != 0:  # True condition
            # Execute the THEN part
            i_pos[0] = i
            return None
        else:
            # Skip to end of statement
            i_pos[0] = len(tokens)
            return None

    else:
        raise BasicError(f"Invalid statement starting with {tok}")

def execute_line(tokens):
    i_pos = [0]
    while i_pos[0] < len(tokens):
        result = execute_statement(tokens, i_pos)
        if result is not None:
            return result
        if i_pos[0] < len(tokens) and tokens[i_pos[0]] == TOKENS[':']:
            i_pos[0] += 1
    return None

def run_program():
    global DATA_PTR, GOSUB_STACK, FOR_STACK, TERM_WIDTH
    DATA_PTR = 0
    GOSUB_STACK = []
    FOR_STACK = []
    
    try:
        TERM_WIDTH = os.get_terminal_size().columns
    except:
        TERM_WIDTH = 80
    
    lines_sorted = sorted(LINES.keys())
    if not lines_sorted: 
        return
    
    pc = 0
    while pc < len(lines_sorted):
        line_num = lines_sorted[pc]
        tokens = LINES[line_num]
        
        for loop in FOR_STACK:
            if loop['line_num'] is None:
                loop['line_num'] = line_num
        
        try:
            result = execute_line(tokens)
        except BasicError as e:
            print(f"ERROR IN LINE {line_num}: {e}")
            return
        
        if isinstance(result, tuple):
            cmd = result[0]
            
            if cmd == 'END':
                return
                
            elif cmd == 'GOTO':
                target = result[1]
                if target not in LINES:
                    print(f"ERROR: Line {target} not found")
                    return
                pc = lines_sorted.index(target)
                continue
                
            elif cmd == 'GOSUB':
                target = result[1]
                if target not in LINES:
                    print(f"ERROR: Line {target} not found")
                    return
                GOSUB_STACK[-1] = lines_sorted[pc]
                pc = lines_sorted.index(target)
                continue
                
            elif cmd == 'RETURN':
                return_line = result[1]
                if return_line is None:
                    print("ERROR: RETURN without GOSUB")
                    return
                pc = lines_sorted.index(return_line) + 1
                continue
                
            elif cmd == 'FOR_CONTINUE':
                target = result[1]
                pc = lines_sorted.index(target) + 1
                continue
        
        pc += 1

def main():
    global DATA_LINES, DATA_PTR, GOSUB_STACK, FOR_STACK
    print("PYTHON-M6502 Interpreter")
    print("Commands: RUN, LIST, NEW, SAVE, LOAD, QUIT")
    print("Ready.")
    
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if line.strip() == "":
            continue

        upper = line.upper().strip()
        
        if upper == "RUN":
            try: 
                run_program()
                print("Ready.")
            except Exception as e: 
                print(f"RUNTIME ERROR: {e}")
                
        elif upper == "LIST":
            if not LINES:
                print("No program in memory.")
            else:
                for num in sorted(LINES.keys()):
                    print(f"{num} {detokenize(LINES[num])}")
                    
        elif upper == "NEW":
            LINES.clear()
            for k in VARS: VARS[k] = 0.0
            for k in STR_VARS: STR_VARS[k] = ''
            ARRAYS.clear()
            STR_ARRAYS.clear()
            DATA_LINES = []
            DATA_PTR = 0
            GOSUB_STACK = []
            FOR_STACK = []
            print("Ready.")
            
        elif upper.startswith("SAVE"):
            parts = line.split(None, 1)
            filename = parts[1] if len(parts) > 1 else input("Filename: ")
            try:
                with open(f"{filename}.bas", "w") as f:
                    for num in sorted(LINES.keys()):
                        f.write(f"{num} {detokenize(LINES[num])}\n")
                print(f"Saved as {filename}.bas")
            except PermissionError:
                print("SAVE ERROR: Permission denied")
            except Exception as e:
                print(f"SAVE ERROR: {e}")
                
        elif upper.startswith("LOAD"):
            parts = line.split(None, 1)
            filename = parts[1] if len(parts) > 1 else input("Filename: ")
            if not filename.endswith('.bas'):
                filename += '.bas'
            try:
                LINES.clear()
                for k in VARS: 
                    VARS[k] = 0.0
                for k in STR_VARS:
                    STR_VARS[k] = ''
                ARRAYS.clear()
                STR_ARRAYS.clear()
                DATA_LINES = []
                DATA_PTR = 0
                GOSUB_STACK = []
                FOR_STACK = []
                
                with open(filename, "r") as f:
                    for line_text in f:
                        line_text = line_text.strip()
                        if line_text: parse_line(line_text)
                print(f"Loaded {filename}")
            except FileNotFoundError: print(f"FILE NOT FOUND: {filename}")
            except PermissionError:
                print("LOAD ERROR: Permission denied")
            except Exception as e: print(f"LOAD ERROR: {e}")
        
        elif upper == "QUIT":
            print("Exiting.")
            break
        
        elif re.match(r'^\d+', line):
            try:  parse_line(line)
            except BasicError as e: print(f"SYNTAX ERROR: {e}")
                
        else:
            try: 
                tokens = tokenize(line)
                result = execute_line(tokens)
                if isinstance(result, tuple) and result[0] == 'END':
                    print("Ready.")
            except BasicError as e:  print(f"ERROR: {e}")
            except Exception as e: print(f"RUNTIME ERROR: {e}")

if __name__ == "__main__": main()