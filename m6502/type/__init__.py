# Single token
class Token:
    def __init__(self, type_: str, val: any):
        self.type = type_
        self.val = val

    def __repr__(self):
        return f"Token(type={self.type!r}, val={self.val!r})"


# Behaves like a list of tokens (TokenList)
class TokenList(list):
    def __init__(self, iterable=None):
        super().__init__(iterable or [])

    def add(self, token):
        self.append(token)

    def __repr__(self):
        return f"TokenList({list.__repr__(self)})"


# Group of tokens (TokenGroup)
class TokenGroup(list):
    def __init__(self, iterable=None):
        super().__init__(iterable or [])

    def add(self, token):
        self.append(token)

    def __repr__(self):
        return f"TokenGroup({list.__repr__(self)})"


# Wrapper for compilation/execution
class TokenCompile:
    def __init__(self, token_list: TokenList):
        self.token_list = token_list

    def __repr__(self):
        return f"TokenCompile({self.token_list})"