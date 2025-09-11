import re
import sys
import math
import os
import array
import logging
import random
import traceback
from typing import List, Tuple, Union, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install
from rich import print as rprint
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table

install()
Console().clear()

# Enhanced logging configuration
class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: "dim blue",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "bold red"
    }

    def format(self, record):
        level_color = self.LEVEL_COLORS.get(record.levelno, "white")
        message = super().format(record)
        return f"[{level_color}]{message}[/{level_color}]"

# Setup logging
file_handler = logging.FileHandler("m6502.log", mode="w")
file_handler.setLevel(logging.DEBUG)

rich_handler = RichHandler(
    rich_tracebacks=True,
    show_time=True,
    show_level=True,
    show_path=True,
    markup=True
)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[rich_handler, file_handler],
    format="%(message)s",
    datefmt="[%X]"
)

logger = logging.getLogger("M6502")

# Token types
class TokenType(Enum):
    # Keywords
    PRINT = "PRINT"
    LET = "LET"
    GOTO = "GOTO"
    IF = "IF"
    THEN = "THEN"
    INPUT = "INPUT"
    END = "END"
    DIM = "DIM"
    READ = "READ"
    RESTORE = "RESTORE"
    DATA = "DATA"
    REM = "REM"
    FOR = "FOR"
    TO = "TO"
    STEP = "STEP"
    NEXT = "NEXT"
    GOSUB = "GOSUB"
    RETURN = "RETURN"
    
    # Functions
    LEFT = "LEFT$"
    RIGHT = "RIGHT$"
    MID = "MID$"
    LEN = "LEN"
    CHR = "CHR$"
    ASC = "ASC"
    SIN = "SIN"
    COS = "COS"
    ABS = "ABS"
    SQR = "SQR"
    RND = "RND"
    PEEK = "PEEK"
    POKE = "POKE"
    
    # Operators
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    EQUALS = "="
    LESS = "<"
    GREATER = ">"
    COMMA = ","
    SEMICOLON = ";"
    LPAREN = "("
    RPAREN = ")"
    COLON = ":"
    
    # Literals
    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    LINE_NUMBER = "LINE_NUMBER"
    
    # Special
    EOF = "EOF"
    NEWLINE = "NEWLINE"

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int
    length: int

    def __str__(self):
        return f"{self.type.value}({repr(self.value)}) at {self.line}:{self.column}"

class BasicError(Exception):
    def __init__(self, message: str, token: Token = None, line: str = None):
        self.message = message
        self.token = token
        self.line = line
        super().__init__(message)
    
    def __str__(self):
        if self.token and self.line:
            pointer = " " * (self.token.column - 1) + "^" * self.token.length
            return f"Line {self.token.line}: {self.message}\n{self.line}\n{pointer}"
        return self.message

class Scanner:
    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self.tokens: List[Token] = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.column = 1
    
    def scan_tokens(self) -> List[Token]:
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        
        self.add_token(TokenType.EOF, None)
        return self.tokens
    
    def scan_token(self):
        char = self.advance()
        
        # Single character tokens
        if char == '+': self.add_token(TokenType.PLUS, char)
        elif char == '-': self.add_token(TokenType.MINUS, char)
        elif char == '*': self.add_token(TokenType.MULTIPLY, char)
        elif char == '/': self.add_token(TokenType.DIVIDE, char)
        elif char == '=': self.add_token(TokenType.EQUALS, char)
        elif char == '<': self.add_token(TokenType.LESS, char)
        elif char == '>': self.add_token(TokenType.GREATER, char)
        elif char == ',': self.add_token(TokenType.COMMA, char)
        elif char == ';': self.add_token(TokenType.SEMICOLON, char)
        elif char == '(': self.add_token(TokenType.LPAREN, char)
        elif char == ')': self.add_token(TokenType.RPAREN, char)
        elif char == ':': self.add_token(TokenType.COLON, char)
        
        # Whitespace
        elif char in ' \t':
            pass  # Ignore whitespace
        
        # Newline
        elif char == '\n':
            self.line += 1
            self.column = 1
            self.add_token(TokenType.NEWLINE, char)
        
        # Numbers
        elif char.isdigit():
            self.number()
        
        # Strings
        elif char == '"':
            self.string()
        
        # Identifiers and keywords
        elif char.isalpha() or char == '$':
            self.identifier()
        
        else:
            self.error(f"Unexpected character: {char}")
    
    def number(self):
        while self.peek().isdigit():
            self.advance()
        
        # Decimal point
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()
            while self.peek().isdigit():
                self.advance()
        
        value = float(self.source[self.start:self.current])
        self.add_token(TokenType.NUMBER, value)
    
    def string(self):
        while self.peek() != '"' and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
                self.column = 0
            self.advance()
        
        if self.is_at_end():
            self.error("Unterminated string")
        
        # Closing quote
        self.advance()
        
        # Remove quotes
        value = self.source[self.start + 1:self.current - 1]
        self.add_token(TokenType.STRING, value)
    
    def identifier(self):
        while self.peek().isalnum() or self.peek() in ('$', '_'):
            self.advance()
        
        text = self.source[self.start:self.current].upper()
        
        # Check if it's a keyword
        keyword_types = {
            'PRINT': TokenType.PRINT, 'LET': TokenType.LET, 'GOTO': TokenType.GOTO,
            'IF': TokenType.IF, 'THEN': TokenType.THEN, 'INPUT': TokenType.INPUT,
            'END': TokenType.END, 'DIM': TokenType.DIM, 'READ': TokenType.READ,
            'RESTORE': TokenType.RESTORE, 'DATA': TokenType.DATA, 'REM': TokenType.REM,
            'FOR': TokenType.FOR, 'TO': TokenType.TO, 'STEP': TokenType.STEP,
            'NEXT': TokenType.NEXT, 'GOSUB': TokenType.GOSUB, 'RETURN': TokenType.RETURN,
            'LEFT$': TokenType.LEFT, 'RIGHT$': TokenType.RIGHT, 'MID$': TokenType.MID,
            'LEN': TokenType.LEN, 'CHR$': TokenType.CHR, 'ASC': TokenType.ASC,
            'SIN': TokenType.SIN, 'COS': TokenType.COS, 'ABS': TokenType.ABS,
            'SQR': TokenType.SQR, 'RND': TokenType.RND, 'PEEK': TokenType.PEEK,
            'POKE': TokenType.POKE
        }
        
        if text in keyword_types:
            self.add_token(keyword_types[text], text)
        else:
            self.add_token(TokenType.IDENTIFIER, text)
    
    def advance(self):
        char = self.source[self.current]
        self.current += 1
        self.column += 1
        return char
    
    def peek(self):
        if self.is_at_end():
            return '\0'
        return self.source[self.current]
    
    def peek_next(self):
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]
    
    def is_at_end(self):
        return self.current >= len(self.source)
    
    def add_token(self, type: TokenType, value: Any):
        text = self.source[self.start:self.current]
        length = self.current - self.start
        self.tokens.append(Token(type, value, self.line, self.column - length, length))
    
    def error(self, message: str):
        current_line = self.get_current_line()
        token = Token(TokenType.EOF, None, self.line, self.column, 1)
        raise BasicError(message, token, current_line)
    
    def get_current_line(self):
        lines = self.source.split('\n')
        if self.line - 1 < len(lines):
            return lines[self.line - 1]
        return ""

class Parser:
    def __init__(self, tokens: List[Token], filename: str = "<input>"):
        self.tokens = tokens
        self.filename = filename
        self.current = 0
        self.lines: Dict[int, List[Token]] = {}
    
    def parse(self) -> Dict[int, List[Token]]:
        try:
            while not self.is_at_end():
                self.parse_line()
            return self.lines
        except BasicError as e:
            # Enhance error with context
            if not e.token:
                raise e
            raise BasicError(
                f"{e.message} in {self.filename}",
                e.token,
                self.get_line_content(e.token.line)
            )
    
    def parse_line(self):
        if self.match(TokenType.NUMBER):
            line_number = int(self.previous().value)
            if not self.match(TokenType.NEWLINE):
                statements = []
                while not self.check(TokenType.NEWLINE) and not self.is_at_end():
                    if self.match(TokenType.COLON):
                        continue  # Skip colons between statements
                    statements.extend(self.parse_statement())
                self.lines[line_number] = statements
            if self.match(TokenType.NEWLINE):
                pass  # Consume newline
        else:
            self.error("Line must start with a number")
    
    def parse_statement(self) -> List[Token]:
        statement_tokens = []
        
        if self.match(TokenType.PRINT, TokenType.LET, TokenType.INPUT, TokenType.DIM,
                     TokenType.READ, TokenType.RESTORE, TokenType.FOR, TokenType.NEXT,
                     TokenType.GOTO, TokenType.IF, TokenType.GOSUB, TokenType.RETURN,
                     TokenType.END, TokenType.REM, TokenType.DATA):
            statement_tokens.append(self.previous())
            
            # Parse the rest of the statement
            while (not self.check(TokenType.NEWLINE) and 
                   not self.check(TokenType.COLON) and 
                   not self.is_at_end()):
                statement_tokens.append(self.advance())
                
        else:
            # Parse as expression
            while (not self.check(TokenType.NEWLINE) and 
                   not self.check(TokenType.COLON) and 
                   not self.is_at_end()):
                statement_tokens.append(self.advance())
        
        return statement_tokens
    
    def match(self, *types: TokenType) -> bool:
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False
    
    def check(self, type: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.peek().type == type
    
    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def previous(self) -> Token:
        return self.tokens[self.current - 1]
    
    def peek(self) -> Token:
        return self.tokens[self.current]
    
    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF
    
    def error(self, message: str, token: Token = None):
        if not token:
            token = self.peek()
        raise BasicError(message, token, self.get_line_content(token.line))
    
    def get_line_content(self, line_num: int) -> str:
        # Reconstruct line from tokens
        line_tokens = [t for t in self.tokens if t.line == line_num and t.type != TokenType.NEWLINE]
        return " ".join(str(t.value) for t in line_tokens)

class Interpreter:
    def __init__(self):
        self.vars = {chr(65 + i): 0.0 for i in range(26)}
        self.str_vars = {chr(65 + i) + '$': '' for i in range(26)}
        self.arrays = {}
        self.str_arrays = {}
        self.lines = {}
        self.data_ptr = 0
        self.data_lines = []
        self.gosub_stack = []
        self.for_stack = []
        
        # Runtime context for better error messages
        self.current_line = 0
        self.current_token = None
        self.console = Console()
        
        try:
            self.term_width = os.get_terminal_size().columns
        except:
            self.term_width = 80
    
    def execute(self, parsed_lines: Dict[int, List[Token]]):
        self.lines = parsed_lines
        self.rebuild_data_lines()
        line_numbers = sorted(self.lines.keys())
        
        if not line_numbers:
            logger.info("No program to execute")
            return
        
        pc = 0
        while pc < len(line_numbers):
            self.current_line = line_numbers[pc]
            tokens = self.lines[self.current_line]
            
            try:
                result = self.execute_line(tokens)
                if result and isinstance(result, tuple):
                    cmd = result[0]
                    if cmd == 'END':
                        break
                    elif cmd == 'GOTO':
                        target = result[1]
                        if target in line_numbers:
                            pc = line_numbers.index(target)
                            continue
                        else:
                            self.error(f"Line {target} not found")
                    elif cmd == 'GOSUB':
                        target = result[1]
                        if target in line_numbers:
                            self.gosub_stack.append(line_numbers[pc])
                            pc = line_numbers.index(target)
                            continue
                        else:
                            self.error(f"Line {target} not found")
                    elif cmd == 'RETURN':
                        if self.gosub_stack:
                            return_line = self.gosub_stack.pop()
                            pc = line_numbers.index(return_line) + 1
                            continue
                        else:
                            self.error("RETURN without GOSUB")
                    elif cmd == 'FOR_CONTINUE':
                        target_line = result[1]
                        pc = line_numbers.index(target_line)
                        continue
                
                pc += 1
                
            except BasicError as e:
                self.console.print(f"[red]Runtime error at line {self.current_line}:[/red] {e}")
                logger.error(f"Runtime error at line {self.current_line}: {e}")
                break
            except Exception as e:
                self.console.print(f"[red]Unexpected error at line {self.current_line}:[/red] {e}")
                logger.exception(f"Unexpected error at line {self.current_line}")
                break
    
    def rebuild_data_lines(self):
        """Rebuild DATA_LINES from all DATA statements."""
        self.data_lines = []
        for line_num in sorted(self.lines.keys()):
            tokens = self.lines[line_num]
            if tokens and tokens[0].type == TokenType.DATA:
                i = 1
                while i < len(tokens):
                    if tokens[i].type == TokenType.COMMA:
                        i += 1
                        continue
                    
                    if tokens[i].type == TokenType.STRING:
                        self.data_lines.append((tokens[i].value, True))
                        i += 1
                    else:
                        # Try to parse as number
                        try:
                            value = self.parse_expression(tokens[i:])
                            self.data_lines.append((value, False))
                            i += 1  # Simplified - should advance properly
                        except:
                            self.data_lines.append((str(tokens[i].value), True))
                            i += 1
    
    def execute_line(self, tokens: List[Token]):
        pos = 0
        while pos < len(tokens):
            self.current_token = tokens[pos]
            result = self.execute_statement(tokens, pos)
            if result is not None:
                return result
            
            # Move to next statement (skip colons)
            pos += 1
            while pos < len(tokens) and tokens[pos].type == TokenType.COLON:
                pos += 1
        
        return None
    
    def execute_statement(self, tokens: List[Token], pos: int):
        token = tokens[pos]
        
        try:
            if token.type == TokenType.PRINT:
                return self.execute_print(tokens, pos)
            elif token.type == TokenType.LET:
                return self.execute_let(tokens, pos)
            elif token.type == TokenType.INPUT:
                return self.execute_input(tokens, pos)
            elif token.type == TokenType.GOTO:
                return self.execute_goto(tokens, pos)
            elif token.type == TokenType.IF:
                return self.execute_if(tokens, pos)
            elif token.type == TokenType.END:
                return ('END',)
            elif token.type == TokenType.DIM:
                return self.execute_dim(tokens, pos)
            elif token.type == TokenType.READ:
                return self.execute_read(tokens, pos)
            elif token.type == TokenType.RESTORE:
                self.data_ptr = 0
                logger.debug("RESTORE: DATA_PTR reset to 0")
                return None
            elif token.type == TokenType.FOR:
                return self.execute_for(tokens, pos)
            elif token.type == TokenType.NEXT:
                return self.execute_next(tokens, pos)
            elif token.type == TokenType.GOSUB:
                return self.execute_gosub(tokens, pos)
            elif token.type == TokenType.RETURN:
                return self.execute_return(tokens, pos)
            elif token.type == TokenType.REM:
                return None  # Skip comments
            elif token.type == TokenType.DATA:
                return None  # DATA statements are handled during rebuild
            else:
                self.error(f"Unknown statement: {token.type.value}")
                
        except Exception as e:
            if isinstance(e, BasicError):
                raise e
            raise BasicError(f"Error in {token.type.value}: {e}", token)
    
    def execute_print(self, tokens: List[Token], pos: int):
        output = []
        pos += 1
        suppress_newline = False
        zone_width = max(self.term_width // 5, 14)
        
        while pos < len(tokens) and tokens[pos].type != TokenType.COLON:
            token = tokens[pos]
            
            if token.type == TokenType.COMMA:
                output.append(" " * (zone_width - (len(''.join(output)) % zone_width)))
                pos += 1
                continue
                
            if token.type == TokenType.SEMICOLON:
                suppress_newline = True
                pos += 1
                continue
            
            # Handle expressions
            expr_result = self.parse_expression(tokens[pos:])
            if isinstance(expr_result, str):
                output.append(expr_result)
            else:
                output.append(str(expr_result))
            
            # Advance position (simplified)
            pos += 1
        
        result = ''.join(output)
        print(result, end="" if suppress_newline else "\n")
        logger.info(f"PRINT: {repr(result)}")
        return None
    
    def execute_let(self, tokens: List[Token], pos: int):
        pos += 1
        if pos >= len(tokens):
            self.error("LET missing variable")
        
        var_token = tokens[pos]
        pos += 1
        
        # Check for assignment operator
        if pos >= len(tokens) or tokens[pos].type != TokenType.EQUALS:
            self.error("Expected = in LET statement")
        pos += 1
        
        # Parse the expression
        value = self.parse_expression(tokens[pos:])
        
        # Handle variable assignment
        if var_token.type == TokenType.IDENTIFIER:
            var_name = var_token.value
            if var_name.endswith('$'):
                self.str_vars[var_name] = str(value)
                logger.debug(f"LET string: {var_name} = {repr(value)}")
            else:
                self.vars[var_name] = float(value)
                logger.debug(f"LET numeric: {var_name} = {value}")
        else:
            self.error("Invalid variable in LET")
        
        return None
    
    def execute_input(self, tokens: List[Token], pos: int):
        pos += 1
        prompt = "?"
        
        # Check for prompt string
        if pos < len(tokens) and tokens[pos].type == TokenType.STRING:
            prompt = tokens[pos].value
            pos += 1
            if pos < len(tokens) and tokens[pos].type == TokenType.SEMICOLON:
                pos += 1
        
        if pos >= len(tokens):
            self.error("INPUT requires variable")
        
        var_token = tokens[pos]
        
        if var_token.type == TokenType.IDENTIFIER:
            var_name = var_token.value
            try:
                user_input = input(prompt + " ")
                if var_name.endswith('$'):
                    self.str_vars[var_name] = user_input
                    logger.info(f"INPUT string: {var_name} = {repr(user_input)}")
                else:
                    self.vars[var_name] = float(user_input)
                    logger.info(f"INPUT numeric: {var_name} = {self.vars[var_name]}")
            except ValueError:
                self.error("Invalid numeric input")
        else:
            self.error("Invalid variable for INPUT")
        
        return None
    
    def execute_goto(self, tokens: List[Token], pos: int):
        pos += 1
        if pos >= len(tokens) or tokens[pos].type != TokenType.NUMBER:
            self.error("GOTO requires line number")
        
        target = int(tokens[pos].value)
        logger.debug(f"GOTO line {target}")
        return ('GOTO', target)
    
    def execute_if(self, tokens: List[Token], pos: int):
        pos += 1
        # Parse condition
        condition = self.parse_expression(tokens[pos:])
        
        # Find THEN
        then_pos = -1
        for i in range(pos, len(tokens)):
            if tokens[i].type == TokenType.THEN:
                then_pos = i
                break
        
        if then_pos == -1:
            self.error("IF without THEN")
        
        # Check condition
        if condition:  # True condition
            # Execute THEN part
            return self.execute_statement(tokens, then_pos + 1)
        
        return None
    
    def execute_dim(self, tokens: List[Token], pos: int):
        pos += 1
        while pos < len(tokens) and tokens[pos].type != TokenType.COLON:
            var_token = tokens[pos]
            pos += 1
            
            if pos >= len(tokens) or tokens[pos].type != TokenType.LPAREN:
                self.error("Expected ( after array variable")
            pos += 1
            
            # Parse array size
            size = int(self.parse_expression(tokens[pos:]))
            
            if pos >= len(tokens) or tokens[pos].type != TokenType.RPAREN:
                self.error("Expected )")
            pos += 1
            
            if var_token.type == TokenType.IDENTIFIER:
                var_name = var_token.value
                if var_name.endswith('$'):
                    self.str_arrays[var_name] = [''] * (size + 1)
                    logger.debug(f"DIM string array: {var_name}({size})")
                else:
                    self.arrays[var_name] = [0.0] * (size + 1)
                    logger.debug(f"DIM numeric array: {var_name}({size})")
            else:
                self.error("Invalid array variable")
            
            if pos < len(tokens) and tokens[pos].type == TokenType.COMMA:
                pos += 1
        
        return None
    
    def execute_read(self, tokens: List[Token], pos: int):
        pos += 1
        while pos < len(tokens) and tokens[pos].type != TokenType.COLON:
            var_token = tokens[pos]
            
            if var_token.type != TokenType.IDENTIFIER:
                self.error("Invalid variable for READ")
            
            var_name = var_token.value
            pos += 1
            
            if self.data_ptr >= len(self.data_lines):
                self.error("Out of DATA")
            
            data_val, is_string = self.data_lines[self.data_ptr]
            
            if var_name.endswith('$'):
                if is_string:
                    self.str_vars[var_name] = data_val
                else:
                    self.str_vars[var_name] = str(data_val)
                logger.debug(f"READ string: {var_name} = {repr(self.str_vars[var_name])}")
            else:
                if is_string:
                    try:
                        self.vars[var_name] = float(data_val)
                    except ValueError:
                        self.error(f"Expected numeric DATA for {var_name}, got '{data_val}'")
                else:
                    self.vars[var_name] = data_val
                logger.debug(f"READ numeric: {var_name} = {self.vars[var_name]}")
            
            self.data_ptr += 1
            
            if pos < len(tokens) and tokens[pos].type == TokenType.COMMA:
                pos += 1
        
        return None
    
    def execute_for(self, tokens: List[Token], pos: int):
        pos += 1
        if pos >= len(tokens) or tokens[pos].type != TokenType.IDENTIFIER:
            self.error("FOR requires variable")
        
        var_token = tokens[pos]
        var_name = var_token.value
        pos += 1
        
        if pos >= len(tokens) or tokens[pos].type != TokenType.EQUALS:
            self.error("Expected = in FOR")
        pos += 1
        
        start_val = self.parse_expression(tokens[pos:])
        pos += 1  # Simplified
        
        if pos >= len(tokens) or tokens[pos].type != TokenType.TO:
            self.error("Expected TO in FOR")
        pos += 1
        
        end_val = self.parse_expression(tokens[pos:])
        pos += 1  # Simplified
        
        step_val = 1.0
        if pos < len(tokens) and tokens[pos].type == TokenType.STEP:
            pos += 1
            step_val = self.parse_expression(tokens[pos:])
        
        # Initialize variable
        if var_name.endswith('$'):
            self.str_vars[var_name] = str(start_val)
        else:
            self.vars[var_name] = float(start_val)
        
        # Store loop info
        self.for_stack.append({
            'var': var_name,
            'end': end_val,
            'step': step_val,
            'line': self.current_line
        })
        
        logger.debug(f"FOR loop: {var_name} = {start_val} TO {end_val} STEP {step_val}")
        return None
    
    def execute_next(self, tokens: List[Token], pos: int):
        if not self.for_stack:
            self.error("NEXT without FOR")
        
        pos += 1
        loop_info = self.for_stack[-1]
        var_name = loop_info['var']
        
        # Update variable
        if var_name.endswith('$'):
            # String loops not supported
            self.error("NEXT with string variable not supported")
        else:
            self.vars[var_name] += loop_info['step']
        
        # Check condition
        if ((loop_info['step'] > 0 and self.vars[var_name] <= loop_info['end']) or
            (loop_info['step'] < 0 and self.vars[var_name] >= loop_info['end'])):
            logger.debug(f"NEXT {var_name}: continue loop, new value = {self.vars[var_name]}")
            return ('FOR_CONTINUE', loop_info['line'])
        else:
            self.for_stack.pop()
            logger.debug(f"NEXT {var_name}: loop finished")
            return None
    
    def execute_gosub(self, tokens: List[Token], pos: int):
        pos += 1
        if pos >= len(tokens) or tokens[pos].type != TokenType.NUMBER:
            self.error("GOSUB requires line number")
        
        target = int(tokens[pos].value)
        logger.debug(f"GOSUB line {target}")
        return ('GOSUB', target)
    
    def execute_return(self, tokens: List[Token], pos: int):
        if not self.gosub_stack:
            self.error("RETURN without GOSUB")
        
        return_line = self.gosub_stack.pop()
        logger.debug(f"RETURN to line {return_line}")
        return ('RETURN', return_line)
    
    def parse_expression(self, tokens: List[Token]) -> Any:
        """Simplified expression parser - in real implementation would be more complex"""
        if not tokens:
            self.error("Empty expression")
        
        # Simple implementation for demo
        token = tokens[0]
        if token.type == TokenType.NUMBER:
            return token.value
        elif token.type == TokenType.STRING:
            return token.value
        elif token.type == TokenType.IDENTIFIER:
            var_name = token.value
            if var_name.endswith('$'):
                return self.str_vars.get(var_name, '')
            else:
                return self.vars.get(var_name, 0.0)
        else:
            self.error(f"Unexpected token in expression: {token.type.value}")
    
    def error(self, message: str, token: Token = None):
        if not token:
            token = self.current_token
        raise BasicError(message, token)

def main():
    interpreter = Interpreter()
    console = Console()
    
    console.print(Panel.fit("PYTHON-M6502 Enhanced Interpreter", style="bold blue"))
    console.print("Commands: RUN, LIST, NEW, SAVE, LOAD, QUIT, HELP")
    console.print("Ready.")
    
    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue
            
            if line.upper() in ("QUIT", "EXIT"):
                console.print("Goodbye!")
                break
            
            elif line.upper() == "HELP":
                table = Table(title="Available Commands")
                table.add_column("Command", style="cyan")
                table.add_column("Description", style="green")
                
                table.add_row("RUN", "Execute the program")
                table.add_row("LIST", "List all program lines")
                table.add_row("NEW", "Clear current program")
                table.add_row("SAVE", "Save program to file")
                table.add_row("LOAD", "Load program from file")
                table.add_row("QUIT", "Exit the interpreter")
                table.add_row("HELP", "Show this help")
                
                console.print(table)
                console.print("\nEnter BASIC code with line numbers to add to program")
                continue
            
            # Handle BASIC lines
            if re.match(r'^\d+', line):
                try:
                    scanner = Scanner(line + '\n')
                    tokens = scanner.scan_tokens()
                    parser = Parser(tokens)
                    parsed_line = parser.parse()
                    interpreter.lines.update(parsed_line)
                    logger.debug(f"Added line: {line}")
                except BasicError as e:
                    console.print(f"[red]Syntax Error:[/red] {e}")
                except Exception as e:
                    console.print(f"[red]Unexpected error:[/red] {e}")
            
            # Handle commands
            elif line.upper() == "RUN":
                try:
                    interpreter.execute(interpreter.lines)
                    console.print("Ready.")
                except BasicError as e:
                    console.print(f"[red]Runtime Error:[/red] {e}")
                except Exception as e:
                    console.print(f"[red]Unexpected error:[/red] {e}")
                    logger.exception("Unexpected error during execution")
            
            elif line.upper() == "LIST":
                if not interpreter.lines:
                    console.print("No program in memory.")
                else:
                    for num in sorted(interpreter.lines.keys()):
                        line_tokens = interpreter.lines[num]
                        line_text = " ".join(str(t.value) for t in line_tokens)
                        console.print(f"[cyan]{num}[/cyan] {line_text}")
            
            elif line.upper() == "NEW":
                interpreter = Interpreter()
                console.print("Program cleared. Ready.")
            
            elif line.upper().startswith("SAVE"):
                console.print("SAVE functionality not yet implemented")
            
            elif line.upper().startswith("LOAD"):
                console.print("LOAD functionality not yet implemented")
            
            else:
                console.print("[yellow]Unknown command. Type HELP for available commands.[/yellow]")
                
        except KeyboardInterrupt:
            console.print("\nInterrupted. Type QUIT to exit.")
        except EOFError:
            console.print("\nGoodbye!")
            break
        except Exception as e:
            console.print(f"[red]Fatal error:[/red] {e}")
            logger.exception("Fatal error in main loop")

if __name__ == "__main__":
    main()
