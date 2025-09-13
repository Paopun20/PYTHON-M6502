import re
import logging
import json
from m6502.logging import Logger
from m6502.dataclass import token

# Initialize logger
Logger = Logger(__name__, level=logging.DEBUG)
logger = Logger.get_logger()

class Visual6502:
    def __init__(self, code_lines: dict[int, list]):
        self.code = dict(sorted(code_lines.items()))  # keep lines ordered
        self.IsEnd = False
        self.pc = min(self.code.keys())  # start at first line

    def step(self):
        if self.pc not in self.code:
            self.IsEnd = True
            return

        line_tokens = self.code[self.pc]
        logger.debug(f"line: {self.pc} | code: {line_tokens}")

        # TODO: decode/execute statement here
        if any(tok["val"] == token.TOKENS["END"] for tok in line_tokens):
            self.IsEnd = True
            return

        # advance to next line
        keys = list(self.code.keys())
        idx = keys.index(self.pc)
        if idx + 1 < len(keys):
            self.pc = keys[idx + 1]
        else:
            self.IsEnd = True

    def run(self):
        while not self.IsEnd:
            self.step()


def tokenize(line: str) -> dict[int, list]:
    """
    Tokenizes a BASIC-like line and returns as {line_number: [tokens...]}
    Tokens are wrapped as {type: ..., val: ...}.
    """
    
    def ctoken(val: any, type: str) -> dict[str, any]:
        """FACK CODE AS SHIT"""
        if type == "token": return {"type": "token", "val": val}
        elif type == "str": return {"type": "str", "val": val}
        elif type == "int": return {"type": "int", "val": val}
        elif type == "float": return {"type": "float", "val": val}
    
    line_match = re.match(r'^\s*(\d+)\s*(.*)$', line)
    if not line_match:
        raise ValueError("Line must start with a number")
    
    line_number = int(line_match.group(1))
    content = line_match.group(2)
    
    # split into parts
    parts = []
    i = 0
    while i < len(content):
        if content[i] == '"':
            j = i + 1
            while j < len(content) and content[j] != '"':
                j += 1
            if j >= len(content):
                raise ValueError("Unterminated string literal")
            parts.append(content[i:j+1])
            i = j + 1
        else:
            j = i
            while j < len(content) and content[j] not in ' \t"+-*/=<>;,():':
                j += 1
            if i < j:
                parts.append(content[i:j])
            if j < len(content) and content[j] not in ' \t':
                parts.append(content[j])
            i = max(j + 1, i + 1)

    parts = [p for p in parts if p.strip()]

    tokens = []
    i = 0
    while i < len(parts):
        p = parts[i]

        if p.startswith('"') and p.endswith('"'):
            tokens.append(ctoken(p[1:-1], "str"))

        elif p.upper() in token.TOKENS:
            code = token.TOKENS[p.upper()]
            if p.upper() == "DATA":
                data_items = []
                i += 1
                while i < len(parts):
                    part = parts[i]
                    if part.startswith('"') and part.endswith('"'):
                        data_items.append(ctoken(part[1:-1], "str"))
                    elif part in '+-*/=<>;,()':
                        data_items.append(ctoken(token.TOKENS[part], "token"))
                    elif re.match(r'^-?\d+\.\d+$', part):
                        data_items.append(ctoken(float(part), "float"))
                    elif re.match(r'^-?\d+$', part):
                        data_items.append(ctoken(int(part), "int"))
                    elif len(part) == 1 and part.isalpha():
                        data_items.append(ctoken(ord(part.upper()), "int"))
                    elif part.upper().endswith('$') and len(part) == 2:
                        data_items.append(ctoken(part.upper(), "str"))
                    else:
                        data_items.append(ctoken(part.upper(), "str"))
                    i += 1
                tokens.append(ctoken(code, "token"))
                tokens.extend(data_items)
                break
            else:
                tokens.append(ctoken(code, "token"))

        elif len(p) == 1 and p.isalpha():
            tokens.append(ctoken(ord(p.upper()), "int"))
        elif p.upper().endswith('$') and len(p) == 2:
            tokens.append(ctoken(p.upper(), "str"))
        elif re.match(r'^-?\d+$', p):
            tokens.append(ctoken(int(p), "int"))
        elif re.match(r'^-?\d+\.\d+$', p):
            tokens.append(ctoken(float(p), "float"))
        elif p in '+-*/=<>;,():':
            tokens.append(ctoken(token.TOKENS[p], "token"))
        else:
            tokens.append(ctoken(p.upper(), "str"))
        i += 1

    return {line_number: tokens}

# Example BASIC lines
basic_lines = [
    '01 DATA 5+5, "HELLO"',
    '02 DATA 10, 20, "WORLD"',
    '03 READ A',
    '04 READ B',
    '05 PRINT A',
    '06 PRINT B',
    '12 LET X = A + B',
    '15 IF X > 20 THEN 25',
    '16 PRINT "SKIP"',
    '25 PRINT "DONE"',
    '30 END',
    '40 DATA 1+2, 3+4, "MORE"',
    '45 READ C, D$',
    '50 PRINT C, D$'
]

if __name__ == "__main__":
    program_dict = {}
    for line in basic_lines: program_dict.update(tokenize(line))
    with open("NEWTEST.json", "w") as f:
        f.write(json.dumps(program_dict, indent=4, sort_keys=True, ensure_ascii=False))
    try:
        vm = Visual6502(program_dict)
        vm.run()
    except InterruptedError as e:
        print(e)