class system:
    def __init__(self):
        self.VARS = {chr(i): 0.0 for i in range(65, 91)}  # A-Z
        self.STR_VARS = {f"{chr(i)}$": '' for i in range(65, 91)}
        self.ARRAYS = {}
        self.STR_ARRAYS = {}
        self.LINES = {}
        self.DATA_LINES = []
        self.DATA_PTR = 0
        self.FOR_STACK = []
        self.GOSUB_STACK = []

    def reset(self):
        # Resets everything without touching token class
        for k in self.VARS: self.VARS[k] = 0.0
        for k in self.STR_VARS: self.STR_VARS[k] = ''
        self.ARRAYS.clear()
        self.STR_ARRAYS.clear()
        self.LINES.clear()
        self.DATA_LINES.clear()
        self.DATA_PTR = 0
        self.FOR_STACK.clear()
        self.GOSUB_STACK.clear()
        return self