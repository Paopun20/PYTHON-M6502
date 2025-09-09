import re
import sys

VARS = {chr(65 + i): 0.0 for i in range(26)}
ARRAYS = {}  # For DIM arrays
LINES = {}  # line_num -> tokenized line
DATA_PTR = 0  # For RESTORE and READ
DATA_LINES = []  # Flattened data tokens

TOKENS = {
    'PRINT': 160, 'LET': 161, 'GOTO': 162, 'IF': 163, 'THEN': 164,
    'INPUT': 165, 'END': 166, '+': 167, '-': 168, '*': 169, '/': 170,
    '=': 171, '<': 172, '>': 173, ',': 174, ';': 175, '(': 176, ')': 177,
    'DIM': 178, 'READ': 179, 'RESTORE': 180, 'DATA': 181, 'REM': 182,
    'FOR': 183, 'TO': 184, 'STEP': 185, 'NEXT': 186, 'GOSUB': 187, 'RETURN': 188
}
KEYWORDS = {v: k for k, v in TOKENS.items() if isinstance(k, str) and k.isupper()}
SYMBOLS = {v: k for k, v in TOKENS.items() if not (isinstance(k, str) and k.isupper())}
GOSUB_STACK = []
FOR_STACK = []

class BasicError(Exception): pass

def tokenize(line):
    # Handle string literals properly and preserve case for strings
    parts = []
    i = 0
    while i < len(line):
        if line[i] == '"':
            # Find closing quote
            j = i + 1
            while j < len(line) and line[j] != '"':
                j += 1
            if j < len(line):
                parts.append(line[i:j+1])  # Include quotes
                i = j + 1
            else:
                parts.append(line[i:])  # Unterminated string
                break
        else:
            # Find next delimiter
            j = i
            while j < len(line) and line[j] not in ' \t"+-*/=<>;,()':
                j += 1
            if i < j:
                parts.append(line[i:j])
            if j < len(line) and line[j] not in ' \t':
                parts.append(line[j])
            i = max(j + 1, i + 1)
    
    # Remove empty parts and spaces
    parts = [p for p in parts if p.strip()]
    
    tokens = []
    for p in parts:
        if p.startswith('"') and p.endswith('"') and len(p) >= 2:
            tokens.append(p[1:-1])  # Remove quotes, preserve original case
        elif p.upper() in TOKENS:
            tokens.append(TOKENS[p.upper()])
        elif len(p) == 1 and p.upper().isalpha():
            tokens.append(ord(p.upper()))
        elif re.match(r'^-?\d+(\.\d+)?$', p):
            tokens.append(float(p))
        elif p in '+-*/=<>;,()':
            tokens.append(TOKENS[p])
        else:
            # Try to parse as identifier or keep as string
            tokens.append(p.upper())
    return tokens

def detokenize(tokens):
    s = ''
    for t in tokens:
        # Strings
        if isinstance(t, str) and t not in TOKENS:
            s += f'"{t}" '
        # Variables A-Z
        elif isinstance(t, int) and 65 <= t <= 90:
            s += chr(t) + ' '
        # Keywords
        elif t in KEYWORDS:
            s += KEYWORDS[t] + ' '
        # Symbols
        elif t in SYMBOLS:
            s += SYMBOLS[t] + ' '
        # Numeric constants
        elif isinstance(t, float):
            if t.is_integer():
                s += str(int(t)) + ' '
            else:
                s += str(t) + ' '
        # Other tokens
        else:
            s += str(t) + ' '
    return s.strip()

def parse_line(line):
    global DATA_LINES
    m = re.match(r'^(\d+)\s*(.*)$', line.strip())
    if not m: 
        raise BasicError("Line must start with number")
    num = int(m.group(1))
    
    # Handle line deletion
    if not m.group(2).strip():
        if num in LINES:
            del LINES[num]
            # Rebuild DATA_LINES
            DATA_LINES = []
            for line_num in sorted(LINES.keys()):
                tokens = LINES[line_num]
                if tokens and tokens[0] == TOKENS['DATA']:
                    DATA_LINES.extend(tokens[1:])
        return
    
    code = tokenize(m.group(2))
    LINES[num] = code
    
    # Rebuild DATA_LINES completely
    DATA_LINES = []
    for line_num in sorted(LINES.keys()):
        tokens = LINES[line_num]
        if tokens and tokens[0] == TOKENS['DATA']:
            DATA_LINES.extend(tokens[1:])

# --------------------
# Expression evaluation with precedence
# --------------------
def parse_factor(tokens, pos):
    if pos[0] >= len(tokens):
        raise BasicError("Unexpected end of expression")
        
    t = tokens[pos[0]]
    pos[0] += 1
    
    if isinstance(t, float):
        return t
    elif isinstance(t, int) and 65 <= t <= 90:
        var = chr(t)
        # Check for array access
        if pos[0] < len(tokens) and tokens[pos[0]] == TOKENS['(']:
            pos[0] += 1  # Skip (
            idx_pos = [pos[0]]
            idx = int(parse_expr(tokens, idx_pos))
            pos[0] = idx_pos[0]
            if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
                raise BasicError("Expected )")
            pos[0] += 1  # Skip )
            if var in ARRAYS and 0 <= idx < len(ARRAYS[var]):
                return ARRAYS[var][idx]
            else:
                raise BasicError(f"Array {var}({idx}) out of bounds")
        return VARS.get(var, 0.0)
    elif t == TOKENS['-']:
        return -parse_factor(tokens, pos)
    elif t == TOKENS['+']:  # Unary plus
        return parse_factor(tokens, pos)
    elif t == TOKENS['(']:
        val = parse_expr(tokens, pos)
        if pos[0] >= len(tokens) or tokens[pos[0]] != TOKENS[')']:
            raise BasicError("Expected )")
        pos[0] += 1
        return val
    else:
        raise BasicError(f"Invalid token in expression: {t}")

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

def execute_line(tokens):
    global DATA_PTR, GOSUB_STACK, FOR_STACK
    i = 0
    
    if not tokens:
        return None
        
    while i < len(tokens):
        tok = tokens[i]

        # REM
        if tok == TOKENS['REM']:
            return None

        # PRINT
        if tok == TOKENS['PRINT']:
            i += 1
            output = ""
            tab_next = False
            suppress_newline = False
            
            while i < len(tokens):
                t = tokens[i]
                if t == TOKENS[',']:
                    # Tab to next print zone
                    current_len = len(output) % 14
                    spaces_needed = 14 - current_len if current_len > 0 else 0
                    output += " " * spaces_needed
                    i += 1
                    continue
                if t == TOKENS[';']:
                    # No spacing, just continue
                    suppress_newline = True
                    i += 1
                    continue
                
                suppress_newline = False
                
                if isinstance(t, str) and t not in TOKENS and not (isinstance(t, int) and 65 <= t <= 90):
                    # String literal
                    output += t
                    i += 1
                else:
                    # Expression
                    pos = [i]
                    val = parse_expr(tokens, pos)
                    i = pos[0]
                    if isinstance(val, float) and val.is_integer():
                        output += str(int(val))
                    else:
                        output += str(val)
            
            print(output, end="" if suppress_newline else "\n")
            return None

        # LET (explicit or implicit)
        elif tok == TOKENS['LET']:
            i += 1
        
        # Check for variable assignment (implicit LET)
        if i < len(tokens):
            var_tok = tokens[i]
            if isinstance(var_tok, int) and 65 <= var_tok <= 90:
                var = chr(var_tok)
                i += 1
                
                # Check for array assignment
                if i < len(tokens) and tokens[i] == TOKENS['(']:
                    i += 1  # Skip (
                    pos = [i]
                    idx = int(parse_expr(tokens, pos))
                    i = pos[0]
                    if i >= len(tokens) or tokens[i] != TOKENS[')']:
                        raise BasicError("Expected )")
                    i += 1  # Skip )
                    if i >= len(tokens) or tokens[i] != TOKENS['=']:
                        raise BasicError("Expected =")
                    i += 1  # Skip =
                    pos = [i]
                    val = parse_expr(tokens, pos)
                    i = pos[0]
                    
                    if var not in ARRAYS:
                        raise BasicError(f"Array {var} not dimensioned")
                    if idx < 0 or idx >= len(ARRAYS[var]):
                        raise BasicError(f"Array {var}({idx}) out of bounds")
                    ARRAYS[var][idx] = val
                    return None
                
                elif i < len(tokens) and tokens[i] == TOKENS['=']:
                    i += 1  # Skip =
                    pos = [i]
                    val = parse_expr(tokens, pos)
                    i = pos[0]
                    VARS[var] = val
                    return None

        # INPUT
        if tok == TOKENS['INPUT']:
            i += 1
            # Handle optional prompt string
            if i < len(tokens) and isinstance(tokens[i], str) and tokens[i] not in TOKENS:
                prompt = tokens[i]
                print(prompt, end="")
                i += 1
                if i < len(tokens) and tokens[i] == TOKENS[';']:
                    i += 1
            
            if i >= len(tokens):
                raise BasicError("INPUT requires variable")
                
            var_tok = tokens[i]
            if not (isinstance(var_tok, int) and 65 <= var_tok <= 90):
                raise BasicError("Expected variable after INPUT")
            var = chr(var_tok)
            i += 1
            
            try:
                user_val = input("? " if 'prompt' not in locals() else "")
                VARS[var] = float(user_val)
            except ValueError:
                raise BasicError("Invalid numeric input")
            except EOFError:
                raise BasicError("Input interrupted")
            return None

        # DIM
        elif tok == TOKENS['DIM']:
            i += 1
            while i < len(tokens):
                if not (isinstance(tokens[i], int) and 65 <= tokens[i] <= 90):
                    raise BasicError("Expected array variable")
                var = chr(tokens[i])
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
                
                ARRAYS[var] = [0.0] * (size + 1)  # 0-based indexing, size+1 elements
                
                # Check for multiple arrays in one DIM statement
                if i < len(tokens) and tokens[i] == TOKENS[',']:
                    i += 1
                else:
                    break
            return None

        # READ
        elif tok == TOKENS['READ']:
            i += 1
            while i < len(tokens):
                if not (isinstance(tokens[i], int) and 65 <= tokens[i] <= 90):
                    raise BasicError("Expected variable")
                var = chr(tokens[i])
                i += 1
                
                if DATA_PTR >= len(DATA_LINES):
                    raise BasicError("Out of DATA")
                    
                data_val = DATA_LINES[DATA_PTR]
                if isinstance(data_val, str):
                    try:
                        VARS[var] = float(data_val)
                    except ValueError:
                        VARS[var] = 0.0
                else:
                    VARS[var] = float(data_val)
                DATA_PTR += 1
                
                if i < len(tokens) and tokens[i] == TOKENS[',']:
                    i += 1
                else:
                    break
            return None

        # RESTORE
        elif tok == TOKENS['RESTORE']:
            DATA_PTR = 0
            return None

        # FOR loop
        elif tok == TOKENS['FOR']:
            i += 1
            if not (isinstance(tokens[i], int) and 65 <= tokens[i] <= 90):
                raise BasicError("Expected variable after FOR")
            var = chr(tokens[i])
            i += 1
            
            if i >= len(tokens) or tokens[i] != TOKENS['=']:
                raise BasicError("Expected = in FOR statement")
            i += 1
            
            pos = [i]
            start_val = parse_expr(tokens, pos)
            i = pos[0]
            
            if i >= len(tokens) or tokens[i] != TOKENS['TO']:
                raise BasicError("Expected TO in FOR statement")
            i += 1
            
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
            FOR_STACK.append({
                'var': var,
                'end': end_val,
                'step': step_val,
                'line_num': None  # Will be set by the interpreter
            })
            return None

        # NEXT
        elif tok == TOKENS['NEXT']:
            i += 1
            next_var = None
            if i < len(tokens) and isinstance(tokens[i], int) and 65 <= tokens[i] <= 90:
                next_var = chr(tokens[i])
                i += 1
            
            if not FOR_STACK:
                raise BasicError("NEXT without FOR")
            
            loop_info = FOR_STACK[-1]
            if next_var and next_var != loop_info['var']:
                raise BasicError(f"NEXT {next_var} doesn't match FOR {loop_info['var']}")
            
            var = loop_info['var']
            VARS[var] += loop_info['step']
            
            # Check if loop should continue
            if ((loop_info['step'] > 0 and VARS[var] <= loop_info['end']) or
                (loop_info['step'] < 0 and VARS[var] >= loop_info['end'])):
                # Continue loop - return to line after FOR
                return ('FOR_CONTINUE', loop_info['line_num'])
            else:
                # End loop
                FOR_STACK.pop()
                return None

        # GOSUB
        elif tok == TOKENS['GOSUB']:
            i += 1
            if i >= len(tokens):
                raise BasicError("GOSUB requires line number")
            target = int(tokens[i])
            GOSUB_STACK.append(None)  # Will be set by interpreter
            return ('GOSUB', target)

        # RETURN
        elif tok == TOKENS['RETURN']:
            if not GOSUB_STACK:
                raise BasicError("RETURN without GOSUB")
            return_line = GOSUB_STACK.pop()
            return ('RETURN', return_line)

        # GOTO
        elif tok == TOKENS['GOTO']:
            i += 1
            if i >= len(tokens):
                raise BasicError("GOTO requires line number")
            target = int(tokens[i])
            return ('GOTO', target)

        # IF ... THEN
        elif tok == TOKENS['IF']:
            i += 1
            # Collect condition tokens
            cond_tokens = []
            while i < len(tokens) and tokens[i] != TOKENS['THEN']:
                cond_tokens.append(tokens[i])
                i += 1
            
            if i >= len(tokens) or tokens[i] != TOKENS['THEN']:
                raise BasicError("Expected THEN")
            i += 1
            
            if i >= len(tokens):
                raise BasicError("Expected line number or statement after THEN")
            
            pos = [0]
            cond_val = parse_rel(cond_tokens, pos)
            
            if cond_val != 0:  # Condition is true
                # Check if it's a line number or a statement
                if isinstance(tokens[i], float) and tokens[i].is_integer():
                    return ('GOTO', int(tokens[i]))
                else:
                    # Execute the rest of the line
                    return execute_line(tokens[i:])
            return None

        # END
        elif tok == TOKENS['END']:
            return ('END',)

        # DATA (skip - already processed)
        elif tok == TOKENS['DATA']:
            return None

        else:
            i += 1
    
    return None

def run_program():
    global DATA_PTR, GOSUB_STACK, FOR_STACK
    DATA_PTR = 0  # Reset on RUN
    GOSUB_STACK = []
    FOR_STACK = []
    
    lines_sorted = sorted(LINES.keys())
    if not lines_sorted: 
        return
    
    pc = 0
    while pc < len(lines_sorted):
        line_num = lines_sorted[pc]
        tokens = LINES[line_num]
        
        # Set current line for FOR loops
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
                GOSUB_STACK[-1] = line_num  # Set return address
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
                pc = lines_sorted.index(target)
                continue
        
        pc += 1

def main():
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
        
        # Commands
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
            ARRAYS.clear()
            
            # Reset all interpreter state
            global DATA_LINES, DATA_PTR, GOSUB_STACK, FOR_STACK
            DATA_LINES = []
            DATA_PTR = 0
            GOSUB_STACK = []
            FOR_STACK = []
            
            print("Ready.")
            
        elif upper == "QUIT" or upper == "EXIT":
            print("Goodbye!")
            break
            
        elif upper.startswith("SAVE"):
            parts = line.split(None, 1)
            filename = parts[1] if len(parts) > 1 else input("Filename: ")
            try:
                with open(f"{filename}.bas", "w") as f:
                    for num in sorted(LINES.keys()):
                        f.write(f"{num} {detokenize(LINES[num])}\n")
                print(f"Saved as {filename}.bas")
            except Exception as e:
                print(f"SAVE ERROR: {e}")
                
        elif upper.startswith("LOAD"):
            parts = line.split(None, 1)
            filename = parts[1] if len(parts) > 1 else input("Filename: ")
            if not filename.endswith('.bas'):
                filename += '.bas'
            try:
                # Clear current program
                LINES.clear()
                for k in VARS: 
                    VARS[k] = 0.0
                ARRAYS.clear()
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
            except Exception as e: print(f"LOAD ERROR: {e}")
        
        # Program line
        elif re.match(r'^\d+', line):
            try:  parse_line(line)
            except BasicError as e: print(f"SYNTAX ERROR: {e}")
                
        # Direct mode execution
        else:
            try: 
                result = execute_line(tokenize(line))
                if isinstance(result, tuple) and result[0] == 'END':
                    print("Ready.")
            except BasicError as e:  print(f"ERROR: {e}")
            except Exception as e: print(f"RUNTIME ERROR: {e}")

if __name__ == "__main__": main()