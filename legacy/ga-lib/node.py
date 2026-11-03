class Node:
    def __init__(self, value, left=None, right=None, is_operator=False):
        self.value = value
        self.left = left
        self.right = right
        self.is_operator = is_operator

    # All operators that consume only the left child.
    UNARY_OPS = {'sin', 'cos', 'exp', 'log', 'tan', 'sinh', 'cosh', 'tanh', 'atan', 'sqrt'}

    def to_formula(self):
        if self.is_operator:
            if self.value in Node.UNARY_OPS:
                left_str = self.left.to_formula() if self.left else ""
                return f"{self.value}({left_str})"
            left_str  = self.left.to_formula()  if self.left  else ""
            right_str = self.right.to_formula() if self.right else ""
            return f"({left_str} {self.value} {right_str})"
        else:
            return str(self.value)

    def evaluate(self, **kwargs):
        import cmath
        if not self.is_operator:
            if isinstance(self.value, str):
                return complex(kwargs.get(self.value, 0.0)) # variable
            return complex(self.value) # number
        
        left_val = self.left.evaluate(**kwargs) if self.left else 0j
        right_val = self.right.evaluate(**kwargs) if self.right else 0j
        
        if self.value == '+': return left_val + right_val
        if self.value == '-': return left_val - right_val
        if self.value == '*': return left_val * right_val
        if self.value == '/': 
            return left_val / right_val if right_val != 0j else 1.0 + 0j
        if self.value == 'sin':
            try: return cmath.sin(left_val)
            except: return complex(float('inf'), float('inf'))
        if self.value == 'cos':
            try: return cmath.cos(left_val)
            except: return complex(float('inf'), float('inf'))
        if self.value == 'tan':
            try: return cmath.tan(left_val)
            except: return complex(float('inf'), float('inf'))
        if self.value == 'sinh':
            try: return cmath.sinh(left_val)
            except: return complex(float('inf'), float('inf'))
        if self.value == 'cosh':
            try: return cmath.cosh(left_val)
            except: return complex(float('inf'), float('inf'))
        if self.value == 'tanh':
            try: return cmath.tanh(left_val)
            except: return complex(float('inf'), float('inf'))
        if self.value == 'atan':
            try: return cmath.atan(left_val)
            except: return complex(float('inf'), float('inf'))
        if self.value == 'sqrt':
            try: return cmath.sqrt(left_val)
            except: return complex(float('inf'), float('inf'))
        if self.value == 'exp':
            try: return cmath.exp(left_val)
            except: return complex(float('inf'), float('inf'))
        if self.value == 'log':
            try: return cmath.log(left_val) if left_val != 0j else complex(float('-inf'), 0)
            except: return complex(float('inf'), float('inf'))
        if self.value == 'eml':
            try:
                # Handle extended reals for log(0)
                if right_val == 0j or right_val == 0:
                    log_val = complex(float('-inf'), 0.0)
                else:
                    log_val = cmath.log(right_val)
                    
                exp_val = cmath.exp(left_val)
                return exp_val - log_val
            except Exception:
                return complex(float('inf'), float('inf'))
            
        raise ValueError(f"Unknown operator: {self.value}")
        
    def copy(self):
        return Node(
            value=self.value,
            left=self.left.copy() if self.left else None,
            right=self.right.copy() if self.right else None,
            is_operator=self.is_operator
        )
